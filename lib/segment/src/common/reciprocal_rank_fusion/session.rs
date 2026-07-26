// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use ordered_float::OrderedFloat;

use super::types::{
    ChannelRankBatch, DynamicRrfAdvance, DynamicRrfExecution, DynamicRrfPolicy,
    DynamicRrfScheduler, DynamicRrfSource, DynamicRrfStopReason, ExactRrfBatchStream,
    ExactRrfStream,
};
use super::{DynamicRrfState, position_score};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::types::ExtendedPointId;

const MIN_CERTIFICATION_INTERVAL: usize = 64;
const MAX_CERTIFICATION_INTERVAL: usize = 1_024;

/// Recoverable exact fusion session. Streams may be paused, replaced by an
/// equivalent implementation after prefix replay, resumed, or cancelled by
/// dropping the session. Replacement never trusts a new stream until its whole
/// observed prefix has matched the original identity order.
pub struct DynamicRrfSession<'a> {
    sources: Vec<DynamicRrfSource<'a>>,
    source_batch_size: usize,
    state: DynamicRrfState,
    source_pulls: Vec<usize>,
    source_history: Vec<Vec<ExtendedPointId>>,
    source_costs: Vec<f64>,
    source_last_advanced: Vec<usize>,
    source_schedule: Vec<usize>,
    competitive_missing: Vec<usize>,
    policy: DynamicRrfPolicy,
    warmup_pulls: usize,
    total_pulls: usize,
    advance_actions: usize,
    pulls_since_check: usize,
    certification_checks: usize,
    force_check: bool,
    exhausted_order: Option<Vec<ExtendedPointId>>,
}

impl<'a> DynamicRrfSession<'a> {
    pub fn new(
        sources: Vec<ExactRrfStream<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        policy: DynamicRrfPolicy,
    ) -> OperationResult<Self> {
        let source_costs = vec![1.0; sources.len()];
        Self::new_with_source_costs(sources, top_k, rrf_k, weights, &source_costs, policy)
    }

    pub fn new_with_source_costs(
        sources: Vec<ExactRrfStream<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        source_costs: &[f64],
        policy: DynamicRrfPolicy,
    ) -> OperationResult<Self> {
        let sources = sources.into_iter().map(DynamicRrfSource::Point).collect();
        Self::new_with_inputs(sources, 1, top_k, rrf_k, weights, source_costs, policy)
    }

    pub fn new_batched(
        sources: Vec<ExactRrfBatchStream<'a>>,
        source_batch_size: usize,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        policy: DynamicRrfPolicy,
    ) -> OperationResult<Self> {
        let source_costs = vec![1.0; sources.len()];
        let sources = sources.into_iter().map(DynamicRrfSource::Batch).collect();
        Self::new_with_inputs(
            sources,
            source_batch_size,
            top_k,
            rrf_k,
            weights,
            &source_costs,
            policy,
        )
    }

    fn new_with_inputs(
        sources: Vec<DynamicRrfSource<'a>>,
        source_batch_size: usize,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        source_costs: &[f64],
        policy: DynamicRrfPolicy,
    ) -> OperationResult<Self> {
        if source_batch_size == 0 {
            return Err(OperationError::validation_error(
                "Dynamic RRF source batch size must be positive",
            ));
        }
        if policy.warmup_check_interval == Some(0) {
            return Err(OperationError::validation_error(
                "Dynamic RRF warmup check interval must be positive",
            ));
        }
        if policy.exhaustive_after_pulls_per_source == Some(0) {
            return Err(OperationError::validation_error(
                "Dynamic RRF exhaustive pull budget must be positive",
            ));
        }
        let source_count = sources.len();
        if policy.fairness_after_actions_per_source == Some(0) {
            return Err(OperationError::validation_error(
                "Dynamic RRF fairness interval must be positive",
            ));
        }
        if source_costs.len() != source_count {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF source costs have length {}; expected {source_count}",
                source_costs.len()
            )));
        }
        if source_costs
            .iter()
            .any(|cost| !cost.is_finite() || *cost <= 0.0)
        {
            return Err(OperationError::validation_error(
                "Dynamic RRF source costs must be finite and positive",
            ));
        }
        Ok(Self {
            sources,
            source_batch_size,
            state: DynamicRrfState::new(source_count, top_k, rrf_k, weights)?,
            source_pulls: vec![0; source_count],
            source_history: vec![Vec::new(); source_count],
            source_costs: source_costs.to_vec(),
            source_last_advanced: vec![0; source_count],
            source_schedule: Vec::new(),
            competitive_missing: vec![top_k.max(1); source_count],
            policy,
            warmup_pulls: top_k.saturating_mul(source_count).saturating_mul(2),
            total_pulls: 0,
            advance_actions: 0,
            pulls_since_check: 0,
            certification_checks: 0,
            force_check: true,
            exhausted_order: None,
        })
    }

    pub fn source_pulls(&self) -> &[usize] {
        &self.source_pulls
    }

    pub fn source_history(&self, source: usize) -> Option<&[ExtendedPointId]> {
        self.source_history.get(source).map(Vec::as_slice)
    }

    pub fn source_schedule(&self) -> &[usize] {
        &self.source_schedule
    }

    pub fn source_is_exhausted(&self, source: usize) -> Option<bool> {
        self.state.exhausted.get(source).copied()
    }

    /// Extends the exact prefix requested from this recoverable session.
    /// Already certified and observed ranks remain valid; decreasing the target
    /// would make a nested consumer's progress ambiguous and is rejected.
    pub fn extend_top_k(&mut self, top_k: usize) -> OperationResult<()> {
        if top_k < self.state.top_k {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF prefix target cannot decrease from {} to {top_k}",
                self.state.top_k,
            )));
        }
        self.state.top_k = top_k;
        self.warmup_pulls = self
            .warmup_pulls
            .max(top_k.saturating_mul(self.sources.len()).saturating_mul(2));
        self.force_check = true;
        Ok(())
    }

    pub fn run_to_prefix(&mut self, top_k: usize) -> OperationResult<DynamicRrfExecution> {
        self.extend_top_k(top_k)?;
        if let Some(order) = &self.exhausted_order {
            return Ok(DynamicRrfExecution {
                point_ids: order.iter().take(top_k).copied().collect(),
                source_pulls: self.source_pulls.clone(),
                source_exhausted: self.state.exhausted.clone(),
                certification_checks: self.certification_checks,
                stop_reason: DynamicRrfStopReason::AllSourcesExhausted,
            });
        }
        let execution = self.run_to_completion()?;
        if execution.stop_reason == DynamicRrfStopReason::AllSourcesExhausted {
            let order = self.state.complete_order();
            let point_ids = order.iter().take(top_k).copied().collect();
            self.exhausted_order = Some(order);
            return Ok(DynamicRrfExecution {
                point_ids,
                ..execution
            });
        }
        Ok(execution)
    }

    pub fn replace_source(
        &mut self,
        source: usize,
        mut replacement: ExactRrfStream<'a>,
    ) -> OperationResult<()> {
        let Some(history) = self.source_history.get(source) else {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF replacement source {source} is out of range"
            )));
        };
        if self.state.exhausted[source] {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF source {source} is already exhausted and cannot be replaced"
            )));
        }
        for (rank, expected) in history.iter().enumerate() {
            let actual = replacement.next().transpose()?.ok_or_else(|| {
                OperationError::validation_error(format!(
                    "Dynamic RRF replacement source {source} ended before observed rank {rank}"
                ))
            })?;
            if actual != *expected {
                return Err(OperationError::validation_error(format!(
                    "Dynamic RRF replacement source {source} disagrees at rank {rank}: expected {expected}, got {actual}"
                )));
            }
        }
        self.sources[source] = DynamicRrfSource::Point(replacement);
        Ok(())
    }

    /// Advances until Top-K is fixed or every requested source reaches its
    /// target pull count. `None` means that source does not gate the pause.
    pub fn advance_until_each(
        &mut self,
        minimum_pulls: &[Option<usize>],
    ) -> OperationResult<DynamicRrfAdvance> {
        if minimum_pulls.len() != self.sources.len() {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF pause targets have length {}; expected {}",
                minimum_pulls.len(),
                self.sources.len()
            )));
        }
        let pause_enabled = minimum_pulls.iter().any(Option::is_some);
        loop {
            let drain_to_exhaustion =
                self.policy
                    .exhaustive_after_pulls_per_source
                    .is_some_and(|budget| {
                        self.total_pulls >= budget.saturating_mul(self.sources.len())
                    });
            let check_interval = if self.total_pulls <= self.warmup_pulls {
                self.policy
                    .warmup_check_interval
                    .unwrap_or(self.sources.len())
            } else {
                (self.total_pulls / 8).clamp(MIN_CERTIFICATION_INTERVAL, MAX_CERTIFICATION_INTERVAL)
            };
            if self.state.all_sources_exhausted()
                || (!drain_to_exhaustion
                    && (self.force_check || self.pulls_since_check >= check_interval))
            {
                self.certification_checks += 1;
                if let Some(point_ids) = self.state.fixed_top_k() {
                    let stop_reason = if self.state.all_sources_exhausted() {
                        DynamicRrfStopReason::AllSourcesExhausted
                    } else {
                        DynamicRrfStopReason::TopKFixed
                    };
                    return Ok(DynamicRrfAdvance::Fixed(DynamicRrfExecution {
                        point_ids,
                        source_pulls: self.source_pulls.clone(),
                        source_exhausted: self.state.exhausted.clone(),
                        certification_checks: self.certification_checks,
                        stop_reason,
                    }));
                }
                // The competitor snapshot is advisory and costs O(observed identities).
                // Refresh on a doubling schedule so an exact certification check does not
                // acquire a second full-state scan every time.
                if self.policy.scheduler == DynamicRrfScheduler::CompetitorCostAware
                    && (self.certification_checks <= 4
                        || self.certification_checks.is_power_of_two())
                {
                    self.competitive_missing = self.state.competitive_missing_counts();
                }
                self.pulls_since_check = 0;
                self.force_check = false;
            }

            let pause_reached = pause_enabled
                && minimum_pulls.iter().enumerate().all(|(source, target)| {
                    target.is_none_or(|target| {
                        self.source_pulls[source] >= target || self.state.exhausted[source]
                    })
                });
            if pause_reached {
                // A pause boundary is also an explicit certification boundary.
                // This keeps probe depth independent from the amortized check schedule.
                self.certification_checks += 1;
                if let Some(point_ids) = self.state.fixed_top_k() {
                    let stop_reason = if self.state.all_sources_exhausted() {
                        DynamicRrfStopReason::AllSourcesExhausted
                    } else {
                        DynamicRrfStopReason::TopKFixed
                    };
                    return Ok(DynamicRrfAdvance::Fixed(DynamicRrfExecution {
                        point_ids,
                        source_pulls: self.source_pulls.clone(),
                        source_exhausted: self.state.exhausted.clone(),
                        certification_checks: self.certification_checks,
                        stop_reason,
                    }));
                }
                if self.policy.scheduler == DynamicRrfScheduler::CompetitorCostAware
                    && (self.certification_checks <= 4
                        || self.certification_checks.is_power_of_two())
                {
                    self.competitive_missing = self.state.competitive_missing_counts();
                }
                return Ok(DynamicRrfAdvance::Paused);
            }

            let source = self.select_source_to_advance(minimum_pulls);
            self.source_schedule.push(source);
            self.advance_actions += 1;
            self.source_last_advanced[source] = self.advance_actions;

            let batch = match &mut self.sources[source] {
                DynamicRrfSource::Point(stream) => match stream.next().transpose()? {
                    Some(id) => ChannelRankBatch {
                        start_rank: self.source_pulls[source],
                        point_ids: vec![id],
                        exhausted: false,
                    },
                    None => ChannelRankBatch {
                        start_rank: self.source_pulls[source],
                        point_ids: Vec::new(),
                        exhausted: true,
                    },
                },
                DynamicRrfSource::Batch(pull) => pull(self.source_batch_size)?,
            };
            if batch.point_ids.len() > self.source_batch_size {
                return Err(OperationError::validation_error(format!(
                    "Dynamic RRF source {source} returned {} identities for a maximum batch size of {}",
                    batch.point_ids.len(),
                    self.source_batch_size,
                )));
            }
            self.state.observe_batch(source, &batch)?;
            self.source_pulls[source] += batch.point_ids.len();
            self.source_history[source].extend(batch.point_ids.iter().copied());
            self.total_pulls += batch.point_ids.len();
            self.pulls_since_check += batch.point_ids.len();
            if batch.exhausted {
                self.force_check = true;
            }
        }
    }

    pub fn run_to_completion(&mut self) -> OperationResult<DynamicRrfExecution> {
        match self.advance_until_each(&vec![None; self.sources.len()])? {
            DynamicRrfAdvance::Fixed(execution) => Ok(execution),
            DynamicRrfAdvance::Paused => {
                unreachable!("a session without pause targets cannot pause")
            }
        }
    }

    fn select_source_to_advance(&self, minimum_pulls: &[Option<usize>]) -> usize {
        let below_pause_target = |source: usize| {
            minimum_pulls[source].is_none_or(|target| self.source_pulls[source] < target)
        };
        if let Some(fairness) = self.policy.fairness_after_actions_per_source {
            let fairness_window = fairness.saturating_mul(self.sources.len());
            if let Some(source) = (0..self.sources.len())
                .filter(|source| {
                    self.state.next_possible_contribution(*source).is_some()
                        && below_pause_target(*source)
                        && self
                            .advance_actions
                            .saturating_sub(self.source_last_advanced[*source])
                            >= fairness_window
                })
                .max_by(|left, right| {
                    let left_wait = self
                        .advance_actions
                        .saturating_sub(self.source_last_advanced[*left]);
                    let right_wait = self
                        .advance_actions
                        .saturating_sub(self.source_last_advanced[*right]);
                    left_wait.cmp(&right_wait).then_with(|| right.cmp(left))
                })
            {
                return source;
            }
        }

        (0..self.sources.len())
            .filter_map(|source| {
                if !below_pause_target(source) {
                    return None;
                }
                let bound = self.state.next_possible_contribution(source)?;
                let priority = match self.policy.scheduler {
                    DynamicRrfScheduler::MaxNextContribution => f64::from(bound),
                    DynamicRrfScheduler::CompetitorCostAware => {
                        let next_bound = position_score(
                            self.state.next_positions[source].saturating_add(1),
                            self.state.rrf_k,
                            self.state.weights[source],
                        );
                        let bound_reduction = f64::from((bound - next_bound).max(0.0));
                        let live_competitors = self.competitive_missing[source].max(1) as f64;
                        bound_reduction * live_competitors / self.source_costs[source]
                    }
                };
                Some((source, OrderedFloat(priority), OrderedFloat(bound)))
            })
            .max_by(
                |(left_source, left_priority, left_bound),
                 (right_source, right_priority, right_bound)| {
                    left_priority
                        .cmp(right_priority)
                        .then_with(|| left_bound.cmp(right_bound))
                        .then_with(|| right_source.cmp(left_source))
                },
            )
            .map(|(source, _, _)| source)
            .expect("unfinished Dynamic RRF must have a source to advance")
    }
}
