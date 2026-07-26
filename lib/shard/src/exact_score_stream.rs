// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Shared pull/merge machinery for exact Segment score streams in one Shard.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::sync_channel;

use ahash::AHashSet;
use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::types::{PointIdType, ScoredPoint};
use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactSourceMode {
    Exact,
    ExhaustiveFallback,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExactShardStreamTelemetry {
    pub sources: usize,
    pub visible_point_copies: usize,
    pub exact_sources: usize,
    pub exhaustive_fallback_sources: usize,
    pub points_pulled: usize,
    pub points_emitted: usize,
    pub duplicates_suppressed: usize,
    /// Physical batch replies fetched from Segment sessions.
    pub batch_requests: usize,
    /// Physical scored points returned by Segment sessions, including points
    /// buffered ahead of the channel-global consumer.
    pub points_received: usize,
}

pub(crate) struct BatchReply {
    pub(crate) points: Vec<ScoredPoint>,
    pub(crate) eof: bool,
}

type BoxedBatchTask = Box<dyn FnOnce() + Send + 'static>;
type DispatchBatchTask =
    dyn Fn(String, BoxedBatchTask) -> OperationResult<()> + Send + Sync + 'static;

#[derive(Clone)]
enum BatchExecution {
    Inline,
    Dispatch(Arc<DispatchBatchTask>),
}

/// Qdrant-runtime hook used by exact Shard sessions.
///
/// Production runs bounded reads inline on its reserved Qdrant coordinator.
/// Tests may dispatch the same owned task to another thread to verify that no
/// Segment guard or borrowed cursor survives a batch boundary.
#[derive(Clone)]
pub struct ExactBatchExecutor {
    execution: BatchExecution,
}

impl ExactBatchExecutor {
    pub fn new(
        dispatch: impl Fn(String, BoxedBatchTask) -> OperationResult<()> + Send + Sync + 'static,
    ) -> Self {
        Self {
            execution: BatchExecution::Dispatch(Arc::new(dispatch)),
        }
    }

    pub(crate) fn run_batch(
        &self,
        name: String,
        task: impl FnOnce() -> OperationResult<BatchReply> + Send + 'static,
    ) -> OperationResult<BatchReply> {
        match &self.execution {
            BatchExecution::Inline => task(),
            BatchExecution::Dispatch(dispatch) => {
                let (reply_tx, reply_rx) = sync_channel(1);
                dispatch(
                    name,
                    Box::new(move || {
                        let _ = reply_tx.send(task());
                    }),
                )?;
                reply_rx.recv().map_err(|_| {
                    OperationError::service_error_light(
                        "exact Segment reader task stopped before returning",
                    )
                })?
            }
        }
    }

    /// Execute one bounded reader task on the current Qdrant search worker.
    ///
    /// The exact coordinator already runs on a reserved Qdrant blocking
    /// runtime. Running the task inline avoids a second Tokio enqueue and a
    /// channel wake-up for every batch while preserving the same owned-state
    /// and transient-read-view boundary.
    pub fn inline_on_current_worker() -> Self {
        Self {
            execution: BatchExecution::Inline,
        }
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn dedicated_threads_for_tests() -> Self {
        Self::new(|name, task| {
            std::thread::Builder::new()
                .name(name)
                .spawn(task)
                .map(|_| ())
                .map_err(|error| {
                    OperationError::service_error_light(format!(
                        "failed to start native test thread: {error}"
                    ))
                })
        })
    }
}

/// One logical Segment score source backed by short Qdrant-runtime tasks.
pub(crate) struct SegmentScoreSource {
    pull: Arc<PullBatch>,
    buffer: VecDeque<ScoredPoint>,
    eof: bool,
    mode: ExactSourceMode,
    previous: Option<ScoredPoint>,
}

type PullBatch = dyn Fn(usize) -> OperationResult<BatchReply> + Send + Sync + 'static;

struct SourcePop {
    point: Option<ScoredPoint>,
    batch_requests: usize,
    points_received: usize,
}

impl SegmentScoreSource {
    pub(crate) fn on_demand(
        pull: impl Fn(usize) -> OperationResult<BatchReply> + Send + Sync + 'static,
        mode: ExactSourceMode,
        _channel: &'static str,
    ) -> Self {
        Self {
            pull: Arc::new(pull),
            buffer: VecDeque::new(),
            eof: false,
            mode,
            previous: None,
        }
    }

    fn pop(&mut self, batch_size: usize, source: usize) -> OperationResult<SourcePop> {
        let mut batch_requests = 0usize;
        let mut points_received = 0usize;
        while self.buffer.is_empty() && !self.eof {
            let batch = (self.pull)(batch_size)?;
            batch_requests = batch_requests.saturating_add(1);
            points_received = points_received.saturating_add(batch.points.len());
            self.eof = batch.eof;
            validate_source_batch(source, self.previous.as_ref(), &batch.points)?;
            self.buffer = VecDeque::from(batch.points);
        }
        let point = self.buffer.pop_front();
        if let Some(point) = &point {
            self.previous = Some(point.clone());
        }
        Ok(SourcePop {
            point,
            batch_requests,
            points_received,
        })
    }
}

fn exact_score_order(left: &ScoredPoint, right: &ScoredPoint) -> Ordering {
    OrderedFloat(right.score)
        .cmp(&OrderedFloat(left.score))
        .then_with(|| left.id.cmp(&right.id))
}

fn validate_source_batch(
    source: usize,
    previous: Option<&ScoredPoint>,
    points: &[ScoredPoint],
) -> OperationResult<()> {
    let mut previous = previous;
    for point in points {
        if !point.score.is_finite() {
            return Err(OperationError::validation_error(format!(
                "exact Segment source {source} produced a non-finite score"
            )));
        }
        if let Some(previous) = previous
            && exact_score_order(previous, point).is_gt()
        {
            return Err(OperationError::validation_error(format!(
                "exact Segment source {source} violated descending score/identity order"
            )));
        }
        previous = Some(point);
    }
    Ok(())
}

#[derive(Debug)]
struct PendingPoint {
    source: usize,
    point: ScoredPoint,
}

impl Eq for PendingPoint {}

impl PartialEq for PendingPoint {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.point.id == other.point.id
            && self.point.version == other.point.version
            && self.point.score == other.point.score
    }
}

impl Ord for PendingPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.point.score)
            .cmp(&OrderedFloat(other.point.score))
            .then_with(|| other.point.id.cmp(&self.point.id))
            .then_with(|| self.point.version.cmp(&other.point.version))
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl PartialOrd for PendingPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Exact score stream merged across every frozen Segment in one Shard.
pub(crate) struct ExactShardMergeState {
    sources: Vec<SegmentScoreSource>,
    pending: BinaryHeap<PendingPoint>,
    seen: AHashSet<PointIdType>,
    batch_size: usize,
    stopped: Arc<AtomicBool>,
    channel: &'static str,
    telemetry: ExactShardStreamTelemetry,
    terminal_error: Option<OperationError>,
}

impl ExactShardMergeState {
    pub(crate) fn open(
        sources: Vec<SegmentScoreSource>,
        visible_point_copies: usize,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
        channel: &'static str,
    ) -> OperationResult<Self> {
        if batch_size == 0 {
            return Err(OperationError::validation_error(format!(
                "native {channel} Shard batch size must be positive"
            )));
        }
        let mut telemetry = ExactShardStreamTelemetry {
            sources: sources.len(),
            visible_point_copies,
            ..Default::default()
        };
        for source in &sources {
            match source.mode {
                ExactSourceMode::Exact => telemetry.exact_sources += 1,
                ExactSourceMode::ExhaustiveFallback => {
                    telemetry.exhaustive_fallback_sources += 1;
                }
            }
        }
        let mut stream = Self {
            sources,
            pending: BinaryHeap::new(),
            seen: AHashSet::new(),
            batch_size,
            stopped,
            channel,
            telemetry,
            terminal_error: None,
        };
        for source in 0..stream.sources.len() {
            // A k-way merge needs only one head from each Segment. Pulling the
            // full continuation batch here would perform batch_size × Segment
            // work before dynamic WRRF has requested a single global rank.
            stream.pull_source(source, 1)?;
        }
        Ok(stream)
    }

    pub(crate) fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        let result = self.next_inner();
        if let Err(error) = &result {
            self.pending.clear();
            self.terminal_error = Some(error.clone());
        }
        result
    }

    pub(crate) fn next_batch(&mut self, max_results: usize) -> OperationResult<Vec<ScoredPoint>> {
        if max_results == 0 {
            return Err(OperationError::validation_error(format!(
                "exact {} Shard output batch size must be positive",
                self.channel
            )));
        }
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }

        let mut output = Vec::with_capacity(max_results);
        while output.len() < max_results {
            match self.next_inner() {
                Ok(Some(point)) => output.push(point),
                Ok(None) => break,
                Err(error) => {
                    self.pending.clear();
                    self.terminal_error = Some(error.clone());
                    return Err(error);
                }
            }
        }
        Ok(output)
    }

    fn next_inner(&mut self) -> OperationResult<Option<ScoredPoint>> {
        if self.stopped.load(AtomicOrdering::Relaxed) {
            return Err(OperationError::cancelled(format!(
                "exact {} Shard stream was cancelled",
                self.channel
            )));
        }
        let Some(pending) = self.pending.pop() else {
            return Ok(None);
        };

        let point = pending.point;
        let mut contributing_sources = SmallVec::<[usize; 4]>::new();
        contributing_sources.push(pending.source);
        while self.pending.peek().is_some_and(|candidate| {
            candidate.point.id == point.id && candidate.point.score == point.score
        }) {
            let duplicate = self.pending.pop().expect("peeked pending head exists");
            if duplicate.point.version != point.version {
                return Err(OperationError::inconsistent_storage(format!(
                    "exact {} Shard generation contains point {} at conflicting versions {} and {}",
                    self.channel, point.id, point.version, duplicate.point.version,
                )));
            }
            contributing_sources.push(duplicate.source);
            self.telemetry.duplicates_suppressed += 1;
        }

        for source in contributing_sources {
            self.pull_source(source, self.batch_size)?;
        }

        if !self.seen.insert(point.id) {
            return Err(OperationError::inconsistent_storage(format!(
                "exact {} Shard generation emitted point {} more than once",
                self.channel, point.id,
            )));
        }
        self.telemetry.points_emitted += 1;
        Ok(Some(point))
    }

    pub(crate) fn telemetry(&self) -> ExactShardStreamTelemetry {
        self.telemetry
    }

    fn pull_source(&mut self, source: usize, limit: usize) -> OperationResult<()> {
        let result = self.sources[source].pop(limit, source)?;
        self.telemetry.batch_requests = self
            .telemetry
            .batch_requests
            .saturating_add(result.batch_requests);
        self.telemetry.points_received = self
            .telemetry
            .points_received
            .saturating_add(result.points_received);
        let Some(point) = result.point else {
            return Ok(());
        };
        self.telemetry.points_pulled += 1;
        self.pending.push(PendingPoint { source, point });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::*;

    fn scored(id: u64, score: f32) -> ScoredPoint {
        ScoredPoint {
            id: id.into(),
            version: 0,
            score,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    #[test]
    fn producer_failure_is_sticky_and_never_becomes_eof() {
        let calls = Arc::new(Mutex::new(0usize));
        let source = SegmentScoreSource::on_demand(
            move |_| {
                let mut calls = calls.lock();
                *calls += 1;
                if *calls == 1 {
                    Ok(BatchReply {
                        points: vec![scored(7, 1.0)],
                        eof: false,
                    })
                } else {
                    Err(OperationError::service_error_light(
                        "synthetic producer failure",
                    ))
                }
            },
            ExactSourceMode::Exact,
            "Test",
        );
        let stopped = Arc::new(AtomicBool::new(false));
        let mut stream = ExactShardMergeState::open(vec![source], 0, 8, stopped, "Test").unwrap();

        let first = stream.next_result().unwrap_err();
        let second = stream.next_result().unwrap_err();
        assert!(first.to_string().contains("synthetic producer failure"));
        assert_eq!(first.to_string(), second.to_string());
    }

    #[test]
    fn batch_merge_suppresses_equivalent_physical_copies() {
        let source = |points: Vec<ScoredPoint>| {
            let points = Arc::new(Mutex::new(VecDeque::from(points)));
            SegmentScoreSource::on_demand(
                move |limit| {
                    let mut points = points.lock();
                    let mut batch = Vec::with_capacity(limit);
                    while batch.len() < limit
                        && let Some(point) = points.pop_front()
                    {
                        batch.push(point);
                    }
                    Ok(BatchReply {
                        eof: points.is_empty(),
                        points: batch,
                    })
                },
                ExactSourceMode::Exact,
                "Test",
            )
        };
        let mut duplicate = scored(7, 9.0);
        duplicate.version = 3;
        let mut first = duplicate.clone();
        let mut second = duplicate;
        first.payload = None;
        second.payload = None;

        let stopped = Arc::new(AtomicBool::new(false));
        let mut stream = ExactShardMergeState::open(
            vec![
                source(vec![first, scored(8, 7.0)]),
                source(vec![second, scored(9, 8.0)]),
            ],
            4,
            4,
            stopped,
            "Test",
        )
        .unwrap();

        let points = stream.next_batch(8).unwrap();
        assert_eq!(
            points.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![7.into(), 9.into(), 8.into()]
        );
        assert_eq!(stream.telemetry().duplicates_suppressed, 1);
    }

    #[test]
    fn conflicting_versions_fail_before_duplicate_emission() {
        let source = |point: ScoredPoint| {
            let point = Arc::new(Mutex::new(Some(point)));
            SegmentScoreSource::on_demand(
                move |_| {
                    let point = point.lock().take();
                    Ok(BatchReply {
                        points: point.into_iter().collect(),
                        eof: true,
                    })
                },
                ExactSourceMode::Exact,
                "Test",
            )
        };
        let mut old = scored(7, 9.0);
        old.version = 3;
        let mut new = old.clone();
        new.version = 4;

        let stopped = Arc::new(AtomicBool::new(false));
        let mut stream =
            ExactShardMergeState::open(vec![source(old), source(new)], 2, 8, stopped, "Test")
                .unwrap();

        assert!(
            stream
                .next_batch(1)
                .unwrap_err()
                .to_string()
                .contains("conflicting versions")
        );
    }

    #[test]
    fn every_output_batch_size_preserves_the_exhaustive_order() {
        fn source(points: Vec<ScoredPoint>) -> SegmentScoreSource {
            let points = Arc::new(Mutex::new(VecDeque::from(points)));
            SegmentScoreSource::on_demand(
                move |limit| {
                    let mut points = points.lock();
                    let mut batch = Vec::with_capacity(limit);
                    while batch.len() < limit
                        && let Some(point) = points.pop_front()
                    {
                        batch.push(point);
                    }
                    Ok(BatchReply {
                        eof: points.is_empty(),
                        points: batch,
                    })
                },
                ExactSourceMode::Exact,
                "Test",
            )
        }

        let expected = (0..400u64).map(PointIdType::from).collect::<Vec<_>>();
        for output_batch_size in [1, 2, 7, 32, 64, 257] {
            let sources = (0..3u64)
                .map(|lane| {
                    source(
                        (lane..400)
                            .step_by(3)
                            .map(|id| scored(id, 1_000.0 - id as f32))
                            .collect(),
                    )
                })
                .collect();
            let mut state = ExactShardMergeState::open(
                sources,
                400,
                32,
                Arc::new(AtomicBool::new(false)),
                "Test",
            )
            .unwrap();

            let mut actual = Vec::new();
            loop {
                let batch = state.next_batch(output_batch_size).unwrap();
                if batch.is_empty() {
                    break;
                }
                actual.extend(batch.into_iter().map(|point| point.id));
            }

            assert_eq!(actual, expected, "output batch size {output_batch_size}");
        }
    }
}
