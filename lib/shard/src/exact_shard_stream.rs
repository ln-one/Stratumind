// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Shared pull/merge machinery for exact Segment score streams in one Shard.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use ahash::AHashSet;
use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::types::{PointIdType, ScoredPoint};
use smallvec::SmallVec;

use crate::locked_segment::LockedSegment;

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

/// One logical Segment score source backed by short Qdrant-runtime tasks.
pub struct SegmentScoreSource {
    pull: Arc<PullBatch>,
    buffer: VecDeque<ScoredPoint>,
    eof: bool,
    mode: ExactSourceMode,
    previous: Option<ScoredPoint>,
}

/// Channel-specific Segment initialization and bounded rank advancement.
///
/// The Shard merge, authoritative identity handling, EOF and error semantics
/// are shared by every exact channel.
pub trait SegmentRankPlan: Clone + Send + Sync + 'static {
    const CHANNEL: &'static str;

    fn open_segment(
        &self,
        source: usize,
        segment: LockedSegment,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<SegmentScoreSource>;
}

/// One exact channel rank stream merged across every frozen Segment in a Shard.
pub struct ExactShardStream<P: SegmentRankPlan> {
    inner: ExactShardMergeState,
    _plan: std::marker::PhantomData<P>,
}

impl<P: SegmentRankPlan> ExactShardStream<P> {
    pub fn open(
        segments: Vec<LockedSegment>,
        plan: P,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<Self> {
        let visible_point_copies = segments
            .iter()
            .map(|segment| {
                segment
                    .get()
                    .read()
                    .available_point_count_without_deferred()
            })
            .sum();
        let sources = segments
            .into_iter()
            .enumerate()
            .map(|(source, segment)| plan.open_segment(source, segment, stopped.clone()))
            .collect::<OperationResult<Vec<_>>>()?;
        Ok(Self {
            inner: ExactShardMergeState::open(
                sources,
                visible_point_copies,
                batch_size,
                stopped,
                P::CHANNEL,
            )?,
            _plan: std::marker::PhantomData,
        })
    }

    pub fn next_batch(&mut self, max_results: usize) -> OperationResult<Vec<ScoredPoint>> {
        self.inner.next_batch(max_results)
    }

    pub fn telemetry(&self) -> ExactShardStreamTelemetry {
        self.inner.telemetry()
    }
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
                "{channel} Shard batch size must be positive"
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
#[path = "exact_shard_stream/tests.rs"]
mod tests;
