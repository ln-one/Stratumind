// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Fail-closed k-way merge for exact per-channel Segment or Shard streams.

use std::collections::HashSet;

use ordered_float::OrderedFloat;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::reciprocal_rank_fusion::ExactRrfStream;
use crate::types::ExtendedPointId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactScoredIdentity {
    pub id: ExtendedPointId,
    pub score: f32,
}

pub type ExactScoreStream<'a> = Box<dyn Iterator<Item = OperationResult<ExactScoredIdentity>> + 'a>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExactScoreMergeTelemetry {
    pub source_pulls: Vec<usize>,
    pub source_exhausted: Vec<bool>,
    pub points_emitted: usize,
}

/// Lazily restores one global channel order from exact local score streams.
///
/// Every source is primed before the first output and refilled before the next
/// winner is selected. Therefore an unobserved local head can never overtake a
/// returned point. Local order violations, duplicate identities, non-finite
/// scores, cancellation, and storage failures all fail closed instead of being
/// mistaken for EOF.
pub struct KWayExactScoreStream<'a> {
    sources: Vec<ExactScoreStream<'a>>,
    heads: Vec<Option<ExactScoredIdentity>>,
    previous: Vec<Option<ExactScoredIdentity>>,
    needs_refill: Vec<bool>,
    seen: HashSet<ExtendedPointId>,
    telemetry: ExactScoreMergeTelemetry,
    terminal_error: Option<OperationError>,
    error_delivered: bool,
}

impl<'a> KWayExactScoreStream<'a> {
    pub fn new(sources: Vec<ExactScoreStream<'a>>) -> OperationResult<Self> {
        if sources.is_empty() {
            return Err(OperationError::validation_error(
                "exact score merge requires at least one source",
            ));
        }
        let source_count = sources.len();
        Ok(Self {
            sources,
            heads: vec![None; source_count],
            previous: vec![None; source_count],
            needs_refill: vec![true; source_count],
            seen: HashSet::new(),
            telemetry: ExactScoreMergeTelemetry {
                source_pulls: vec![0; source_count],
                source_exhausted: vec![false; source_count],
                points_emitted: 0,
            },
            terminal_error: None,
            error_delivered: false,
        })
    }

    pub fn telemetry(&self) -> &ExactScoreMergeTelemetry {
        &self.telemetry
    }

    pub fn into_rank_stream(self) -> ExactRrfStream<'a> {
        Box::new(self.map(|result| result.map(|point| point.id)))
    }

    fn fail(&mut self, error: OperationError) {
        self.terminal_error = Some(error);
        self.heads.fill(None);
        self.needs_refill.fill(false);
    }

    fn refill(&mut self, source: usize) -> OperationResult<()> {
        if !self.needs_refill[source] || self.telemetry.source_exhausted[source] {
            return Ok(());
        }
        self.needs_refill[source] = false;
        let Some(point) = self.sources[source].next().transpose()? else {
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
        self.heads[source] = Some(point);
        Ok(())
    }

    fn next_inner(&mut self) -> OperationResult<Option<ExactScoredIdentity>> {
        for source in 0..self.sources.len() {
            self.refill(source)?;
        }
        let Some(source) = self
            .heads
            .iter()
            .enumerate()
            .filter_map(|(source, point)| point.map(|point| (source, point)))
            .min_by(|(_, left), (_, right)| exact_score_order(*left, *right))
            .map(|(source, _)| source)
        else {
            return Ok(None);
        };
        let point = self.heads[source]
            .take()
            .expect("selected exact score source has a head");
        self.needs_refill[source] = true;
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
