// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use crate::common::operation_error::OperationResult;
use crate::types::ExtendedPointId;

/// Recoverable channel-global exact rank stream.
///
/// Producer failures remain distinct from successful exhaustion, so a partial
/// prefix can never be certified after an I/O, snapshot, or cancellation error.
pub type ExactRrfStream<'a> = Box<dyn Iterator<Item = OperationResult<ExtendedPointId>> + 'a>;

/// One contiguous prefix extension from a channel-global exact rank stream.
///
/// `start_rank` is zero-based. `exhausted` proves that no later rank exists.
/// Empty non-exhausted batches are forbidden.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelRankBatch {
    pub start_rank: usize,
    pub point_ids: Vec<ExtendedPointId>,
    pub exhausted: bool,
}

/// Recoverable batch pull boundary for one channel-global exact rank stream.
pub type ExactRrfBatchStream<'a> = Box<dyn FnMut(usize) -> OperationResult<ChannelRankBatch> + 'a>;

pub(super) enum DynamicRrfSource<'a> {
    Point(ExactRrfStream<'a>),
    Batch(ExactRrfBatchStream<'a>),
}

/// Adapts an in-memory or otherwise infallible ordered identity iterator to
/// the production exact-stream contract.
pub fn infallible_exact_rrf_stream<'a, I>(source: I) -> ExactRrfStream<'a>
where
    I: Iterator<Item = ExtendedPointId> + 'a,
{
    Box::new(source.map(Ok))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicRrfStopReason {
    TopKFixed,
    AllSourcesExhausted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicRrfExecution {
    pub point_ids: Vec<ExtendedPointId>,
    pub source_pulls: Vec<usize>,
    pub source_exhausted: Vec<bool>,
    pub certification_checks: usize,
    pub stop_reason: DynamicRrfStopReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DynamicRrfAdvance {
    Fixed(DynamicRrfExecution),
    Paused,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DynamicRrfScheduler {
    /// Legacy exact baseline: advance the source with the largest next RRF
    /// contribution.
    MaxNextContribution,
    /// Prefer the source expected to remove the most live Top-K uncertainty
    /// per unit of physical work. This changes only pull order; certification
    /// remains the responsibility of [`super::DynamicRrfState::fixed_top_k`].
    #[default]
    CompetitorCostAware,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynamicRrfPolicy {
    /// `None` preserves the original one-check-per-source-round schedule.
    pub warmup_check_interval: Option<usize>,
    /// Drain already-open streams after the aggregate pull count reaches this
    /// value multiplied by the source count (a normalized per-source budget).
    pub exhaustive_after_pulls_per_source: Option<usize>,
    /// Selects an exact-equivalent source pull schedule.
    pub scheduler: DynamicRrfScheduler,
    /// Maximum scheduling actions a relevant unfinished source may wait,
    /// expressed as a multiple of the source count. `None` disables fairness.
    pub fairness_after_actions_per_source: Option<usize>,
}

impl Default for DynamicRrfPolicy {
    fn default() -> Self {
        Self {
            warmup_check_interval: None,
            exhaustive_after_pulls_per_source: None,
            scheduler: DynamicRrfScheduler::CompetitorCostAware,
            fairness_after_actions_per_source: Some(8),
        }
    }
}
