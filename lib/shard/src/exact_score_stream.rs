// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Shared pull/merge machinery for exact Segment score streams in one Shard.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::sync_channel;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::DeferredBehavior;
use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::types::{PointIdType, ScoredPoint, SeqNumberType};

use crate::locked_segment::LockedSegment;
use crate::read_segment_handle::ReadSegmentHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AuthoritativePointVersion {
    version: SeqNumberType,
    source: usize,
}

/// Authoritative visible point owner for one frozen Shard generation.
///
/// Qdrant may temporarily retain older point copies across Segments. Exact
/// channel streams must remove those copies before score ordering; seeing a
/// stale high score first and deduplicating later is not correct.
pub struct ExactShardPointVersions {
    points: HashMap<PointIdType, AuthoritativePointVersion>,
}

impl ExactShardPointVersions {
    pub fn build(segments: &[LockedSegment], stopped: &AtomicBool) -> OperationResult<Arc<Self>> {
        let hardware_counter = HardwareCounterCell::disposable();
        let mut points = HashMap::<PointIdType, AuthoritativePointVersion>::new();
        for (source, segment) in segments.iter().enumerate() {
            let segment = segment.read_segment();
            let ids = segment.read_filtered(
                None,
                None,
                None,
                stopped,
                &hardware_counter,
                DeferredBehavior::Exclude,
            )?;
            for id in ids {
                let version = segment.point_version(id).ok_or_else(|| {
                    OperationError::inconsistent_storage(format!(
                        "native exact Shard snapshot lost visible point {id} while resolving versions"
                    ))
                })?;
                match points.get_mut(&id) {
                    None => {
                        points.insert(id, AuthoritativePointVersion { version, source });
                    }
                    Some(current) if version > current.version => {
                        *current = AuthoritativePointVersion { version, source };
                    }
                    Some(current) if version == current.version && source != current.source => {
                        return Err(OperationError::inconsistent_storage(format!(
                            "native exact Shard snapshot contains point {id} at version {version} in multiple Segments"
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(Arc::new(Self { points }))
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn contains(&self, source: usize, point: &ScoredPoint) -> bool {
        self.points.get(&point.id).is_some_and(|authoritative| {
            authoritative.source == source && authoritative.version == point.version
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactSourceMode {
    Exact,
    ExhaustiveFallback,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExactShardStreamTelemetry {
    pub sources: usize,
    pub visible_points: usize,
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
}

type PullBatch = dyn Fn(usize) -> OperationResult<BatchReply> + Send + Sync + 'static;

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
        }
    }

    fn pop(&mut self, batch_size: usize) -> OperationResult<(Option<ScoredPoint>, usize)> {
        let mut fetched = 0;
        if self.buffer.is_empty() && !self.eof {
            let batch = (self.pull)(batch_size)?;
            fetched = batch.points.len();
            self.buffer = VecDeque::from(batch.points);
            self.eof = batch.eof;
        }
        Ok((self.buffer.pop_front(), fetched))
    }
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
pub(crate) struct ExactShardScoreStream {
    sources: Vec<SegmentScoreSource>,
    pending: BinaryHeap<PendingPoint>,
    seen: HashSet<PointIdType>,
    batch_size: usize,
    stopped: Arc<AtomicBool>,
    channel: &'static str,
    point_versions: Arc<ExactShardPointVersions>,
    telemetry: ExactShardStreamTelemetry,
    terminal_error: Option<OperationError>,
}

impl ExactShardScoreStream {
    pub(crate) fn open(
        sources: Vec<SegmentScoreSource>,
        point_versions: Arc<ExactShardPointVersions>,
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
            visible_points: point_versions.len(),
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
            seen: HashSet::new(),
            batch_size,
            stopped,
            channel,
            point_versions,
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

    fn next_inner(&mut self) -> OperationResult<Option<ScoredPoint>> {
        loop {
            if self.stopped.load(AtomicOrdering::Relaxed) {
                return Err(OperationError::cancelled(format!(
                    "native {} Shard stream was cancelled",
                    self.channel
                )));
            }
            let Some(pending) = self.pending.pop() else {
                return Ok(None);
            };
            self.pull_source(pending.source, self.batch_size)?;
            if !self.seen.insert(pending.point.id) {
                return Err(OperationError::inconsistent_storage(format!(
                    "native {} Shard stream emitted authoritative point {} more than once",
                    self.channel, pending.point.id,
                )));
            }
            self.telemetry.points_emitted += 1;
            return Ok(Some(pending.point));
        }
    }

    pub(crate) fn telemetry(&self) -> ExactShardStreamTelemetry {
        self.telemetry
    }

    fn pull_source(&mut self, source: usize, limit: usize) -> OperationResult<()> {
        loop {
            let (point, fetched) = self.sources[source].pop(limit)?;
            if fetched != 0 {
                self.telemetry.batch_requests += 1;
                self.telemetry.points_received += fetched;
            }
            let Some(point) = point else {
                return Ok(());
            };
            self.telemetry.points_pulled += 1;
            if !self.point_versions.contains(source, &point) {
                self.telemetry.duplicates_suppressed += 1;
                continue;
            }
            self.pending.push(PendingPoint { source, point });
            return Ok(());
        }
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
        let point_versions = Arc::new(ExactShardPointVersions {
            points: HashMap::from([(
                7_u64.into(),
                AuthoritativePointVersion {
                    version: 0,
                    source: 0,
                },
            )]),
        });
        let mut stream =
            ExactShardScoreStream::open(vec![source], point_versions, 8, stopped, "Test").unwrap();

        let first = stream.next_result().unwrap_err();
        let second = stream.next_result().unwrap_err();
        assert!(first.to_string().contains("synthetic producer failure"));
        assert_eq!(first.to_string(), second.to_string());
    }
}
