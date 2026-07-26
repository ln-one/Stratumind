// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Fail-closed k-way merge for exact per-channel Segment or Shard streams.
//!
//! This is the physical scored-pull layer. Dense and Sparse scores are never
//! compared here; one `KWayExactScoreStream` combines sources belonging to one
//! channel. Only the fully merged channel order may be converted into an
//! identity-only `ExactRrfStream` for cross-channel fusion.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};

use ordered_float::OrderedFloat;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::reciprocal_rank_fusion::{ChannelRankBatch, ExactRrfStream};
use crate::types::ExtendedPointId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactScoredIdentity {
    pub id: ExtendedPointId,
    pub score: f32,
}

/// Exact, locally ordered scored results for one channel.
///
/// The type is deliberately synchronous and borrowing. Cross-thread and
/// remote producers must own their lifecycle outside this fusion contract and
/// adapt their pull replies at the coordinator boundary.
pub type ExactScoreStream<'a> = Box<dyn Iterator<Item = OperationResult<ExactScoredIdentity>> + 'a>;

#[derive(Clone, Debug, PartialEq)]
pub struct ExactScoreBatch {
    pub points: Vec<ExactScoredIdentity>,
    pub exhausted: bool,
}

pub type ExactScoreBatchStream<'a> = Box<dyn FnMut(usize) -> OperationResult<ExactScoreBatch> + 'a>;

enum ExactScoreSource<'a> {
    Point(ExactScoreStream<'a>),
    Batch {
        pull: ExactScoreBatchStream<'a>,
        buffer: VecDeque<ExactScoredIdentity>,
        exhausted: bool,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExactScoreMergeTelemetry {
    pub source_pulls: Vec<usize>,
    pub source_exhausted: Vec<bool>,
    pub points_emitted: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingExactScore {
    source: usize,
    point: ExactScoredIdentity,
}

impl Eq for PendingExactScore {}

impl Ord for PendingExactScore {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.point.score)
            .cmp(&OrderedFloat(other.point.score))
            .then_with(|| other.point.id.cmp(&self.point.id))
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl PartialOrd for PendingExactScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Lazily restores one global channel order from exact local score streams.
///
/// Every source is primed before the first output and refilled before the next
/// winner is selected. Therefore an unobserved local head can never overtake a
/// returned point. Local order violations, duplicate identities, non-finite
/// scores, cancellation, and storage failures all fail closed instead of being
/// mistaken for EOF.
pub struct KWayExactScoreStream<'a> {
    sources: Vec<ExactScoreSource<'a>>,
    source_refill_batch_size: usize,
    pending: BinaryHeap<PendingExactScore>,
    previous: Vec<Option<ExactScoredIdentity>>,
    refill_source: Option<usize>,
    seen: HashSet<ExtendedPointId>,
    telemetry: ExactScoreMergeTelemetry,
    terminal_error: Option<OperationError>,
    error_delivered: bool,
}

impl<'a> KWayExactScoreStream<'a> {
    pub fn new(sources: Vec<ExactScoreStream<'a>>) -> OperationResult<Self> {
        let sources = sources.into_iter().map(ExactScoreSource::Point).collect();
        Self::new_with_sources(sources, 1)
    }

    pub fn new_batched(
        sources: Vec<ExactScoreBatchStream<'a>>,
        source_refill_batch_size: usize,
    ) -> OperationResult<Self> {
        let sources = sources
            .into_iter()
            .map(|pull| ExactScoreSource::Batch {
                pull,
                buffer: VecDeque::new(),
                exhausted: false,
            })
            .collect();
        Self::new_with_sources(sources, source_refill_batch_size)
    }

    fn new_with_sources(
        sources: Vec<ExactScoreSource<'a>>,
        source_refill_batch_size: usize,
    ) -> OperationResult<Self> {
        if sources.is_empty() {
            return Err(OperationError::validation_error(
                "exact score merge requires at least one source",
            ));
        }
        if source_refill_batch_size == 0 {
            return Err(OperationError::validation_error(
                "exact score merge source refill batch size must be positive",
            ));
        }
        let source_count = sources.len();
        let mut stream = Self {
            sources,
            source_refill_batch_size,
            pending: BinaryHeap::with_capacity(source_count),
            previous: vec![None; source_count],
            refill_source: None,
            seen: HashSet::new(),
            telemetry: ExactScoreMergeTelemetry {
                source_pulls: vec![0; source_count],
                source_exhausted: vec![false; source_count],
                points_emitted: 0,
            },
            terminal_error: None,
            error_delivered: false,
        };
        for source in 0..source_count {
            stream.refill(source)?;
        }
        Ok(stream)
    }

    pub fn telemetry(&self) -> &ExactScoreMergeTelemetry {
        &self.telemetry
    }

    /// Pull one contiguous channel-global exact-rank batch.
    ///
    /// An error commits no identities to the caller. Internal failure is
    /// sticky, and a later call returns the same error instead of EOF.
    pub fn next_rank_batch(&mut self, max_results: usize) -> OperationResult<ChannelRankBatch> {
        if max_results == 0 {
            return Err(OperationError::validation_error(
                "exact score merge batch size must be positive",
            ));
        }
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }

        let start_rank = self.telemetry.points_emitted;
        let mut point_ids = Vec::with_capacity(max_results);
        let mut exhausted = false;
        while point_ids.len() < max_results {
            match self.next_inner() {
                Ok(Some(point)) => point_ids.push(point.id),
                Ok(None) => {
                    exhausted = true;
                    break;
                }
                Err(error) => {
                    self.fail(error.clone());
                    return Err(error);
                }
            }
        }
        Ok(ChannelRankBatch {
            start_rank,
            point_ids,
            exhausted,
        })
    }

    /// Erase scores only after every visible source for this channel has been
    /// included in the merge. A Segment- or Shard-local instance is not a
    /// global rank stream and must not be passed to WRRF.
    pub fn into_rank_stream(self) -> ExactRrfStream<'a> {
        Box::new(self.map(|result| result.map(|point| point.id)))
    }

    fn fail(&mut self, error: OperationError) {
        self.terminal_error = Some(error);
        self.pending.clear();
        self.refill_source = None;
    }

    fn refill(&mut self, source: usize) -> OperationResult<()> {
        if self.telemetry.source_exhausted[source] {
            return Ok(());
        }
        let previous_before_refill = self.previous[source];
        let point = match &mut self.sources[source] {
            ExactScoreSource::Point(stream) => stream.next().transpose()?,
            ExactScoreSource::Batch {
                pull,
                buffer,
                exhausted,
            } => {
                if buffer.is_empty() && !*exhausted {
                    let batch = pull(self.source_refill_batch_size)?;
                    if batch.points.len() > self.source_refill_batch_size {
                        return Err(OperationError::validation_error(format!(
                            "exact score source {source} returned {} points for a maximum refill size of {}",
                            batch.points.len(),
                            self.source_refill_batch_size,
                        )));
                    }
                    if batch.points.is_empty() && !batch.exhausted {
                        return Err(OperationError::validation_error(format!(
                            "exact score source {source} returned an empty non-exhausted batch"
                        )));
                    }
                    let mut previous = previous_before_refill;
                    let mut batch_ids = HashSet::with_capacity(batch.points.len());
                    for point in &batch.points {
                        if !point.score.is_finite() {
                            return Err(OperationError::validation_error(format!(
                                "exact score source {source} produced a non-finite score"
                            )));
                        }
                        if previous
                            .is_some_and(|previous| exact_score_order(previous, *point).is_gt())
                        {
                            return Err(OperationError::validation_error(format!(
                                "exact score source {source} violated descending score/identity order inside a batch"
                            )));
                        }
                        if self.seen.contains(&point.id) || !batch_ids.insert(point.id) {
                            return Err(OperationError::validation_error(format!(
                                "exact score source {source} repeated visible identity {} inside a batch",
                                point.id
                            )));
                        }
                        previous = Some(*point);
                    }
                    *buffer = VecDeque::from(batch.points);
                    *exhausted = batch.exhausted;
                }
                buffer.pop_front()
            }
        };
        let Some(point) = point else {
            self.telemetry.source_exhausted[source] = true;
            return Ok(());
        };
        if !point.score.is_finite() {
            return Err(OperationError::validation_error(format!(
                "exact score source {source} produced a non-finite score"
            )));
        }
        if let Some(previous) = self.previous[source]
            && exact_score_order(previous, point).is_gt()
        {
            return Err(OperationError::validation_error(format!(
                "exact score source {source} violated descending score/identity order"
            )));
        }
        self.telemetry.source_pulls[source] += 1;
        self.previous[source] = Some(point);
        self.pending.push(PendingExactScore { source, point });
        Ok(())
    }

    fn next_inner(&mut self) -> OperationResult<Option<ExactScoredIdentity>> {
        if let Some(source) = self.refill_source.take() {
            self.refill(source)?;
        }
        let Some(pending) = self.pending.pop() else {
            return Ok(None);
        };
        self.refill_source = Some(pending.source);
        let point = pending.point;
        if !self.seen.insert(point.id) {
            return Err(OperationError::validation_error(format!(
                "exact score merge observed duplicate visible identity {}",
                point.id
            )));
        }
        self.telemetry.points_emitted += 1;
        Ok(Some(point))
    }
}

impl Iterator for KWayExactScoreStream<'_> {
    type Item = OperationResult<ExactScoredIdentity>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.terminal_error.clone() {
            if self.error_delivered {
                return None;
            }
            self.error_delivered = true;
            return Some(Err(error));
        }
        match self.next_inner() {
            Ok(Some(point)) => Some(Ok(point)),
            Ok(None) => None,
            Err(error) => {
                self.fail(error.clone());
                self.error_delivered = true;
                Some(Err(error))
            }
        }
    }
}

fn exact_score_order(left: ExactScoredIdentity, right: ExactScoredIdentity) -> std::cmp::Ordering {
    OrderedFloat(right.score)
        .cmp(&OrderedFloat(left.score))
        .then_with(|| left.id.cmp(&right.id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: u64, score: f32) -> ExactScoredIdentity {
        ExactScoredIdentity {
            id: id.into(),
            score,
        }
    }

    fn stream(points: Vec<ExactScoredIdentity>) -> ExactScoreStream<'static> {
        Box::new(points.into_iter().map(Ok))
    }

    fn batch_stream(points: Vec<ExactScoredIdentity>) -> ExactScoreBatchStream<'static> {
        let mut points = VecDeque::from(points);
        Box::new(move |max_results| {
            let batch = (0..max_results)
                .filter_map(|_| points.pop_front())
                .collect();
            Ok(ExactScoreBatch {
                points: batch,
                exhausted: points.is_empty(),
            })
        })
    }

    #[test]
    fn merges_local_exact_orders_into_one_global_channel_order() {
        let merged = KWayExactScoreStream::new(vec![
            stream(vec![point(2, 9.0), point(3, 7.0)]),
            stream(vec![point(1, 9.0), point(4, 8.0)]),
        ])
        .unwrap()
        .collect::<OperationResult<Vec<_>>>()
        .unwrap();

        assert_eq!(
            merged.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![1.into(), 2.into(), 4.into(), 3.into()]
        );
    }

    #[test]
    fn emits_contiguous_channel_rank_batches_and_explicit_eof() {
        let mut merged = KWayExactScoreStream::new(vec![
            stream(vec![point(2, 9.0), point(3, 7.0)]),
            stream(vec![point(1, 9.0), point(4, 8.0)]),
        ])
        .unwrap();

        assert_eq!(
            merged.next_rank_batch(3).unwrap(),
            ChannelRankBatch {
                start_rank: 0,
                point_ids: vec![1.into(), 2.into(), 4.into()],
                exhausted: false,
            }
        );
        assert_eq!(
            merged.next_rank_batch(3).unwrap(),
            ChannelRankBatch {
                start_rank: 3,
                point_ids: vec![3.into()],
                exhausted: true,
            }
        );
    }

    #[test]
    fn batched_sources_preserve_global_score_order() {
        let mut merged = KWayExactScoreStream::new_batched(
            vec![
                batch_stream(vec![point(2, 9.0), point(3, 7.0)]),
                batch_stream(vec![point(1, 9.0), point(4, 8.0)]),
            ],
            2,
        )
        .unwrap();

        assert_eq!(
            merged.next_rank_batch(8).unwrap(),
            ChannelRankBatch {
                start_rank: 0,
                point_ids: vec![1.into(), 2.into(), 4.into(), 3.into()],
                exhausted: true,
            }
        );
    }

    #[test]
    fn batch_failure_is_sticky_and_commits_no_partial_reply() {
        let failed: ExactScoreStream<'_> = Box::new(
            vec![
                Ok(point(1, 9.0)),
                Err(OperationError::cancelled("segment stopped")),
            ]
            .into_iter(),
        );
        let mut merged = KWayExactScoreStream::new(vec![failed]).unwrap();

        assert!(matches!(
            merged.next_rank_batch(8),
            Err(OperationError::Cancelled { .. })
        ));
        assert!(matches!(
            merged.next_rank_batch(8),
            Err(OperationError::Cancelled { .. })
        ));
    }

    #[test]
    fn source_failure_is_not_treated_as_global_eof() {
        let failed: ExactScoreStream<'_> = Box::new(
            vec![
                Ok(point(1, 9.0)),
                Err(OperationError::cancelled("segment stopped")),
            ]
            .into_iter(),
        );
        let mut merged =
            KWayExactScoreStream::new(vec![failed, stream(vec![point(2, 8.0), point(3, 7.0)])])
                .unwrap();

        assert_eq!(merged.next().transpose().unwrap(), Some(point(1, 9.0)));
        assert!(matches!(
            merged.next(),
            Some(Err(OperationError::Cancelled { .. }))
        ));
        assert_eq!(merged.next(), None);
    }

    #[test]
    fn duplicate_visible_identity_fails_closed() {
        let mut merged = KWayExactScoreStream::new(vec![
            stream(vec![point(1, 9.0)]),
            stream(vec![point(1, 8.0)]),
        ])
        .unwrap();

        assert_eq!(merged.next().transpose().unwrap(), Some(point(1, 9.0)));
        assert!(merged.next().unwrap().is_err());
    }

    #[test]
    fn local_order_violation_fails_closed() {
        let mut merged =
            KWayExactScoreStream::new(vec![stream(vec![point(1, 8.0), point(2, 9.0)])]).unwrap();

        assert_eq!(merged.next().transpose().unwrap(), Some(point(1, 8.0)));
        assert!(merged.next().unwrap().is_err());
    }
}
