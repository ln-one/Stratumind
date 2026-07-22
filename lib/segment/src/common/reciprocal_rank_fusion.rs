//! Reciprocal Rank Fusion (RRF) is a method for combining rankings from multiple sources.
//! See <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>
// Stratumind extension: deterministic exact ranked-stream composition.

use std::collections::hash_map::Entry;

use ahash::AHashMap;
use itertools::Either;
use ordered_float::OrderedFloat;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::types::{ExtendedPointId, ScoredPoint};

/// Mitigates the impact of high rankings by outlier systems
pub const DEFAULT_RRF_K: usize = 2;
const MIN_CERTIFICATION_INTERVAL: usize = 64;
const MAX_CERTIFICATION_INTERVAL: usize = 1_024;

/// Compute the RRF score for a given position with optional weight.
///
/// The formula is: `1.0 / ((position + 1) as f32 / weight + k as f32 - 1.0)`
///
/// With weight=1.0 (default), this becomes the standard RRF formula: `1.0 / (position + k)`
///
/// Higher or lower weight means the positions are "compressed" or "stretched":
///
/// weight=3.0 is equivalent to dividing the position by 3,
///            so element at pos 2 contributes like pos 0 would with weight=1.0.
///
/// `(position + 1)` accounts for 0-based indexing,
///  so weight affects the score of the top-ranked item (pos=0) as well.
///
/// This means a 3:1 weight ratio is equivalent to "for each 3 results of first prefetch,
/// have one result of second".
fn position_score(position: usize, k: usize, weight: f32) -> f32 {
    // Avoid division by zero - if weight is 0, treat as negligible contribution
    if weight <= 0.0 {
        return 0.0;
    }

    1.0 / ((position + 1) as f32 / weight + k as f32 - 1.0)
}

#[derive(Debug)]
struct PartialRrfPoint {
    contributions: Vec<Option<f32>>,
}

/// Incremental RRF state over exact, rank-ordered input streams.
///
/// The state fixes a result only when no observed or still-unseen point can
/// change its position under the configured identity tie-break. Input scores
/// are intentionally absent: RRF depends only on each source rank.
#[derive(Debug)]
pub struct DynamicRrfState {
    rrf_k: usize,
    top_k: usize,
    weights: Vec<f32>,
    next_positions: Vec<usize>,
    exhausted: Vec<bool>,
    points: AHashMap<ExtendedPointId, PartialRrfPoint>,
}

/// Recoverable exact rank stream.
///
/// `None` is the only successful end-of-stream signal. Producer failures stay
/// distinct from exhaustion so a partial prefix can never be certified after
/// an I/O, snapshot, or cancellation error.
pub type ExactRrfStream<'a> = Box<dyn Iterator<Item = OperationResult<ExtendedPointId>> + 'a>;

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
    /// remains the responsibility of [`DynamicRrfState::fixed_top_k`].
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

/// Recoverable exact fusion session. Streams may be paused, replaced by an
/// equivalent implementation after prefix replay, resumed, or cancelled by
/// dropping the session. Replacement never trusts a new stream until its whole
/// observed prefix has matched the original identity order.
pub struct DynamicRrfSession<'a> {
    sources: Vec<ExactRrfStream<'a>>,
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
        self.sources[source] = replacement;
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

            match self.sources[source].next().transpose()? {
                Some(id) => {
                    self.state.observe(source, id)?;
                    self.source_pulls[source] += 1;
                    self.source_history[source].push(id);
                    self.total_pulls += 1;
                    self.pulls_since_check += 1;
                }
                None => {
                    self.state.finish_source(source)?;
                    self.force_check = true;
                }
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

impl DynamicRrfState {
    pub fn new(
        source_count: usize,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
    ) -> OperationResult<Self> {
        if source_count == 0 {
            return Err(OperationError::validation_error(
                "Dynamic RRF requires at least one source",
            ));
        }
        if rrf_k == 0 {
            return Err(OperationError::validation_error(
                "Dynamic RRF requires k to be greater than zero",
            ));
        }

        let weights = match weights {
            Some(weights) if weights.len() != source_count => {
                return Err(OperationError::validation_error(format!(
                    "Number of weights in Dynamic RRF should match number of sources: got {}, expected {source_count}",
                    weights.len(),
                )));
            }
            Some(weights) => weights.to_vec(),
            None => vec![1.0; source_count],
        };

        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(OperationError::validation_error(
                "Dynamic RRF weights must be finite and non-negative",
            ));
        }

        Ok(Self {
            rrf_k,
            top_k,
            weights,
            next_positions: vec![0; source_count],
            exhausted: vec![false; source_count],
            points: AHashMap::new(),
        })
    }

    pub fn observe(&mut self, source: usize, id: ExtendedPointId) -> OperationResult<()> {
        if source >= self.weights.len() {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF source {source} is out of range",
            )));
        }
        if self.exhausted[source] {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF source {source} was already exhausted",
            )));
        }

        let contribution = position_score(
            self.next_positions[source],
            self.rrf_k,
            self.weights[source],
        );
        let point = self.points.entry(id).or_insert_with(|| PartialRrfPoint {
            contributions: vec![None; self.weights.len()],
        });
        if point.contributions[source].is_some() {
            return Err(OperationError::validation_error(format!(
                "Point {id} occurs more than once in Dynamic RRF source {source}",
            )));
        }
        point.contributions[source] = Some(contribution);
        self.next_positions[source] += 1;
        Ok(())
    }

    pub fn finish_source(&mut self, source: usize) -> OperationResult<()> {
        let Some(exhausted) = self.exhausted.get_mut(source) else {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF source {source} is out of range",
            )));
        };
        *exhausted = true;
        Ok(())
    }

    pub fn next_possible_contribution(&self, source: usize) -> Option<f32> {
        if source >= self.weights.len() || self.exhausted[source] {
            return None;
        }
        Some(position_score(
            self.next_positions[source],
            self.rrf_k,
            self.weights[source],
        ))
    }

    pub fn all_sources_exhausted(&self) -> bool {
        self.exhausted.iter().all(|exhausted| *exhausted)
    }

    /// Approximate how many still-competitive identities are missing each
    /// source contribution. The scheduler may use this snapshot, but the exact
    /// stop decision never does.
    fn competitive_missing_counts(&self) -> Vec<usize> {
        let source_count = self.weights.len();
        let mut counts = vec![0; source_count];
        if self.all_sources_exhausted() {
            return counts;
        }
        if self.points.len() < self.top_k {
            let demand = self.top_k.saturating_sub(self.points.len()).max(1);
            for (source, count) in counts.iter_mut().enumerate() {
                if !self.exhausted[source] {
                    *count = demand;
                }
            }
            return counts;
        }

        let next_contributions: Vec<_> = (0..source_count)
            .map(|source| self.next_possible_contribution(source).unwrap_or(0.0))
            .collect();
        let mut lower_bounds: Vec<_> = self
            .points
            .values()
            .map(|point| {
                point
                    .contributions
                    .iter()
                    .map(|contribution| contribution.unwrap_or(0.0))
                    .sum::<f32>()
            })
            .collect();
        lower_bounds.sort_unstable_by(|left, right| OrderedFloat(*right).cmp(&OrderedFloat(*left)));
        let threshold = lower_bounds[self.top_k - 1];

        for point in self.points.values() {
            let upper_bound = point
                .contributions
                .iter()
                .zip(&next_contributions)
                .map(|(contribution, next)| contribution.unwrap_or(*next))
                .sum::<f32>();
            if upper_bound < threshold {
                continue;
            }
            for (source, contribution) in point.contributions.iter().enumerate() {
                if contribution.is_none() && !self.exhausted[source] {
                    counts[source] += 1;
                }
            }
        }

        let unseen_upper_bound: f32 = next_contributions.iter().sum();
        if unseen_upper_bound >= threshold {
            for (source, count) in counts.iter_mut().enumerate() {
                if !self.exhausted[source] {
                    *count += 1;
                }
            }
        }
        counts
    }

    /// Returns the exact fused identity order once it is fixed, regardless of
    /// whether every final RRF score has itself been fully materialized.
    pub fn fixed_top_k(&self) -> Option<Vec<ExtendedPointId>> {
        if self.top_k == 0 {
            return Some(Vec::new());
        }

        let all_exhausted = self.exhausted.iter().all(|exhausted| *exhausted);
        if !all_exhausted && self.points.len() < self.top_k {
            return None;
        }
        if all_exhausted {
            return Some(self.complete_order().into_iter().take(self.top_k).collect());
        }

        let next_contributions: Vec<_> = (0..self.weights.len())
            .map(|source| self.next_possible_contribution(source).unwrap_or(0.0))
            .collect();
        let unseen_upper_bound: f32 = next_contributions.iter().sum();
        let unseen_points_possible = true;

        let mut remaining: Vec<_> = self
            .points
            .iter()
            .map(|(id, point)| {
                let lower_bound = point
                    .contributions
                    .iter()
                    .map(|contribution| contribution.unwrap_or(0.0))
                    .sum::<f32>();
                let upper_bound = point
                    .contributions
                    .iter()
                    .zip(&next_contributions)
                    .map(|(contribution, next)| contribution.unwrap_or(*next))
                    .sum::<f32>();
                (*id, lower_bound, upper_bound)
            })
            .collect();
        let target_count = self.top_k;
        let mut fixed = Vec::with_capacity(target_count);

        for _ in 0..target_count {
            let winner_index = remaining
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| rrf_order(a.0, a.1, b.0, b.1))
                .map(|(index, _)| index)?;
            let winner = remaining[winner_index];

            if unseen_points_possible && winner.1 <= unseen_upper_bound {
                return None;
            }
            if remaining.iter().enumerate().any(|(index, challenger)| {
                index != winner_index
                    && !guaranteed_before(winner.0, winner.1, challenger.0, challenger.2)
            }) {
                return None;
            }

            fixed.push(winner.0);
            remaining.swap_remove(winner_index);
        }

        Some(fixed)
    }

    fn complete_order(&self) -> Vec<ExtendedPointId> {
        debug_assert!(self.all_sources_exhausted());
        let mut points: Vec<_> = self
            .points
            .iter()
            .filter_map(|(id, point)| {
                let score = point
                    .contributions
                    .iter()
                    .map(|contribution| contribution.unwrap_or(0.0))
                    .sum::<f32>();
                (score > 0.0).then_some((*id, score))
            })
            .collect();
        points.sort_unstable_by(|left, right| rrf_order(left.0, left.1, right.0, right.1));
        points.into_iter().map(|(id, _)| id).collect()
    }
}

/// Pull exact channel streams only until their fused Top-K identity order is
/// certified. The scheduler advances the source whose next unseen rank has the
/// largest possible RRF contribution; source index resolves equal bounds.
pub fn execute_dynamic_rrf(
    sources: Vec<ExactRrfStream<'_>>,
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
) -> OperationResult<DynamicRrfExecution> {
    execute_dynamic_rrf_with_policy(sources, top_k, rrf_k, weights, DynamicRrfPolicy::default())
}

pub fn execute_dynamic_rrf_with_policy(
    sources: Vec<ExactRrfStream<'_>>,
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
    policy: DynamicRrfPolicy,
) -> OperationResult<DynamicRrfExecution> {
    DynamicRrfSession::new(sources, top_k, rrf_k, weights, policy)?.run_to_completion()
}

fn rrf_order(
    left_id: ExtendedPointId,
    left_score: f32,
    right_id: ExtendedPointId,
    right_score: f32,
) -> std::cmp::Ordering {
    OrderedFloat(right_score)
        .cmp(&OrderedFloat(left_score))
        .then_with(|| left_id.cmp(&right_id))
}

fn guaranteed_before(
    winner_id: ExtendedPointId,
    winner_lower_bound: f32,
    challenger_id: ExtendedPointId,
    challenger_upper_bound: f32,
) -> bool {
    winner_lower_bound > challenger_upper_bound
        || (winner_lower_bound == challenger_upper_bound && winner_id < challenger_id)
}

/// Compute RRF scores for multiple results from different sources.
/// Each response can have a different length.
/// The input scores are irrelevant, only the order matters.
///
/// # Arguments
/// * `responses` - Iterator of response vectors from different sources
/// * `k` - The RRF K parameter (default is 2)
/// * `weights` - Optional weights for each source. If provided, must match the number of sources.
///   Higher weight = more influence on final ranking.
///   If None, all sources are weighted equally (weight = 1.0).
///
/// The output is a single sorted list of ScoredPoint.
/// Does not break ties.
pub fn rrf_scoring(
    responses: Vec<Vec<ScoredPoint>>,
    k: usize,
    weights: Option<&[f32]>,
) -> OperationResult<Vec<ScoredPoint>> {
    // track scored points by id
    let mut points_by_id: AHashMap<ExtendedPointId, ScoredPoint> = AHashMap::new();

    let weights = if let Some(weights) = weights {
        if weights.len() != responses.len() {
            return Err(OperationError::validation_error(format!(
                "Number of weights in RRF should match number of pre-fetches: got {}, expected {}",
                weights.len(),
                responses.len()
            )));
        }
        Either::Left(weights.iter().copied())
    } else {
        Either::Right(std::iter::repeat(1.0f32))
    };

    for (response, weight) in responses.into_iter().zip(weights) {
        for (pos, mut point) in response.into_iter().enumerate() {
            let rrf_score = position_score(pos, k, weight);
            match points_by_id.entry(point.id) {
                Entry::Occupied(mut entry) => {
                    // accumulate score
                    entry.get_mut().score += rrf_score;
                }
                Entry::Vacant(entry) => {
                    point.score = rrf_score;
                    // init score
                    entry.insert(point);
                }
            }
        }
    }

    let mut scores: Vec<_> = points_by_id.into_values().collect();
    scores.sort_unstable_by(|a, b| {
        // sort by score descending
        OrderedFloat(b.score).cmp(&OrderedFloat(a.score))
    });

    Ok(scores)
}

/// Compute the deterministic positive-score order used by Stratumind exact
/// composition without changing Qdrant's existing RRF response semantics.
///
/// Qdrant historically retains zero-score identities and does not specify an
/// identity tie-break. Exact ranked streams require both choices to be fixed,
/// so this wrapper deliberately narrows the order only at the Stratumind
/// boundary.
pub fn exact_rrf_scoring(
    responses: Vec<Vec<ScoredPoint>>,
    k: usize,
    weights: Option<&[f32]>,
) -> OperationResult<Vec<ScoredPoint>> {
    let mut scores = rrf_scoring(responses, k, weights)?;
    scores.retain(|point| point.score > 0.0);
    scores.sort_unstable_by(|left, right| rrf_order(left.id, left.score, right.id, right.score));
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use sparse::common::sparse_vector::RemappedSparseVector;
    use sparse::index::block_max::{BlockMaxIndex, SparseDocument};

    use super::*;
    use crate::types::ScoredPoint;

    fn make_scored_point(id: u64, score: f32) -> ScoredPoint {
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

    fn exhaustive_top_k(
        sources: &[Vec<ExtendedPointId>],
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
    ) -> Vec<ExtendedPointId> {
        let responses = sources
            .iter()
            .map(|source| {
                source
                    .iter()
                    .map(|id| make_scored_point(id.as_u64(), 0.0))
                    .collect()
            })
            .collect();
        exact_rrf_scoring(responses, rrf_k, weights)
            .unwrap()
            .into_iter()
            .take(top_k)
            .map(|point| point.id)
            .collect()
    }

    fn run_dynamic(
        sources: &[Vec<ExtendedPointId>],
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
    ) -> (Vec<ExtendedPointId>, Vec<usize>) {
        let mut state = DynamicRrfState::new(sources.len(), top_k, rrf_k, weights).unwrap();
        let mut observed = vec![0; sources.len()];

        loop {
            for (source, points) in sources.iter().enumerate() {
                if observed[source] == points.len() && !state.exhausted[source] {
                    state.finish_source(source).unwrap();
                }
            }

            if let Some(fixed) = state.fixed_top_k() {
                return (fixed, observed);
            }

            let source = (0..sources.len())
                .filter(|source| !state.exhausted[*source])
                .max_by(|left, right| {
                    let left_bound = state.next_possible_contribution(*left).unwrap();
                    let right_bound = state.next_possible_contribution(*right).unwrap();
                    OrderedFloat(left_bound)
                        .cmp(&OrderedFloat(right_bound))
                        .then_with(|| right.cmp(left))
                })
                .expect("unfinished Dynamic RRF must have a source to advance");
            let id = sources[source][observed[source]];
            state.observe(source, id).unwrap();
            observed[source] += 1;
        }
    }

    fn random_rankings(
        allow_missing_points: bool,
    ) -> impl Strategy<Value = (Vec<Vec<ExtendedPointId>>, Vec<f32>, usize, usize)> {
        (1usize..48, 1usize..5)
            .prop_flat_map(|(point_count, source_count)| {
                (
                    Just(point_count),
                    Just(source_count),
                    prop::collection::vec(any::<u16>(), point_count * source_count),
                    prop::collection::vec(any::<bool>(), point_count * source_count),
                    prop::collection::vec(0u8..5, source_count),
                    1usize..point_count + 1,
                    1usize..101,
                )
            })
            .prop_map(
                move |(point_count, source_count, keys, presence, weight_classes, top_k, rrf_k)| {
                    let sources = (0..source_count)
                        .map(|source| {
                            let mut ids: Vec<_> = (0..point_count as u64)
                                .filter(|id| {
                                    !allow_missing_points
                                        || presence[source * point_count + *id as usize]
                                })
                                .collect();
                            ids.sort_unstable_by_key(|id| {
                                (keys[source * point_count + *id as usize], *id)
                            });
                            ids.into_iter().map(ExtendedPointId::from).collect()
                        })
                        .collect();
                    let weights = weight_classes
                        .into_iter()
                        .map(|weight| match weight {
                            0 => 0.0,
                            1 => 0.5,
                            2 => 1.0,
                            3 => 2.0,
                            _ => 4.0,
                        })
                        .collect();
                    (sources, weights, top_k, rrf_k)
                },
            )
    }

    #[test]
    fn test_rrf_scoring_empty() {
        let responses = vec![];
        let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();
        assert_eq!(scored_points.len(), 0);
    }

    #[test]
    fn test_rrf_scoring_one() {
        let responses = vec![vec![make_scored_point(1, 0.9)]];
        let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();
        assert_eq!(scored_points.len(), 1);
        assert_eq!(scored_points[0].id, 1.into());
        assert_eq!(scored_points[0].score, 0.5); // 1 / (0 + 2)
    }

    #[test]
    fn test_rrf_scoring() {
        let responses = vec![
            vec![make_scored_point(2, 0.9), make_scored_point(1, 0.8)],
            vec![
                make_scored_point(1, 0.7),
                make_scored_point(2, 0.6),
                make_scored_point(3, 0.5),
            ],
            vec![
                make_scored_point(5, 0.9),
                make_scored_point(3, 0.5),
                make_scored_point(1, 0.4),
            ],
        ];

        // top 10
        let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();
        assert_eq!(scored_points.len(), 4);
        // assert that the list is sorted
        assert!(
            scored_points
                .array_windows()
                .all(|[a, b]| a.score >= b.score),
        );

        assert_eq!(scored_points.len(), 4);
        assert_eq!(scored_points[0].id, 1.into());
        assert_eq!(scored_points[0].score, 1.0833334);

        assert_eq!(scored_points[1].id, 2.into());
        assert_eq!(scored_points[1].score, 0.8333334);

        assert_eq!(scored_points[2].id, 3.into());
        assert_eq!(scored_points[2].score, 0.5833334);

        assert_eq!(scored_points[3].id, 5.into());
        assert_eq!(scored_points[3].score, 0.5);
    }

    #[test]
    fn test_rrf_scoring_weighted() {
        // Two sources: first with weight 3, second with weight 1
        // This should give 3x more influence to the first source
        let responses = vec![
            vec![make_scored_point(1, 0.9), make_scored_point(2, 0.8)],
            vec![make_scored_point(2, 0.9), make_scored_point(1, 0.8)],
        ];

        // Without weights - both equal
        let scored_points = rrf_scoring(responses.clone(), DEFAULT_RRF_K, None).unwrap();
        assert_eq!(scored_points[0].score, scored_points[1].score);

        // With weights [3.0, 1.0] - first source has 3x weight
        // Higher weight means positions are "compressed" - position N with weight W
        // contributes like position N/W would with weight 1.
        let weights = [3.0, 1.0];
        let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, Some(&weights)).unwrap();

        // Point 2 scores higher because:
        // - Being at pos 1 in high-weight source (w=3) costs less (effective pos = 1/3)
        // - Being at pos 0 in low-weight source still gives full 1/k score
        // So the weighted RRF favors items that rank well across sources,
        // with higher-weight sources having their position penalties reduced.
        assert!(scored_points[0].id == 2.into());
        assert!(scored_points[0].score > scored_points[1].score);
    }

    #[test]
    fn test_rrf_scoring_weighted_ratio() {
        // Test that weight ratio of 3:1 means position 3 in source 1 equals position 1 in source 2
        let k = 60; // Use higher k for clearer demonstration

        // Source 1: item A at position 0, item B at position 3
        // Source 2: item B at position 0, item A at position 1
        let responses = vec![
            vec![
                make_scored_point(11, 0.0),
                make_scored_point(12, 0.0),
                make_scored_point(13, 0.0),
                make_scored_point(14, 0.0),
                make_scored_point(15, 0.0),
                make_scored_point(16, 0.0),
                make_scored_point(17, 0.0),
                make_scored_point(18, 0.0),
            ],
            vec![
                make_scored_point(21, 0.0),
                make_scored_point(22, 0.0),
                make_scored_point(23, 0.0),
                make_scored_point(24, 0.0),
                make_scored_point(25, 0.0),
                make_scored_point(26, 0.0),
                make_scored_point(27, 0.0),
                make_scored_point(28, 0.0),
            ],
        ];

        let weights = [3.0, 1.0];
        let scored_points = rrf_scoring(responses, k, Some(&weights)).unwrap();

        // Check that points from the first group appear 3 times more frequently in the top ranks than points from the second group
        let top_10 = &scored_points[..10];
        let count_source_1 = top_10
            .iter()
            .filter(|p| p.id.as_u64() >= 10 && p.id.as_u64() < 20)
            .count();
        let count_source_2 = top_10
            .iter()
            .filter(|p| p.id.as_u64() >= 20 && p.id.as_u64() < 30)
            .count();

        // With a 3:1 weight ratio, we expect the count of source 1 items in the top 10 to be roughly 3 times that of source 2
        assert!(count_source_1 >= 2 * count_source_2); // Allow some variance due to tie-breaking and small sample size
    }

    #[test]
    fn test_rrf_scoring_weights_length_mismatch() {
        let responses = vec![
            vec![make_scored_point(1, 0.9)],
            vec![make_scored_point(2, 0.9)],
        ];

        // 3 weights for 2 responses should fail
        let weights = [1.0, 2.0, 3.0];
        let result = rrf_scoring(responses.clone(), DEFAULT_RRF_K, Some(&weights));
        assert!(result.is_err());

        // 1 weight for 2 responses should fail
        let weights = [1.0];
        let result = rrf_scoring(responses, DEFAULT_RRF_K, Some(&weights));
        assert!(result.is_err());
    }

    #[test]
    fn test_rrf_scoring_zero_weight() {
        // Test that zero weight source contributes nothing
        let responses = vec![
            vec![make_scored_point(1, 0.9)],
            vec![make_scored_point(2, 0.9)],
        ];

        let weights = [1.0, 0.0];
        let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, Some(&weights)).unwrap();

        // Only point 1 should have a score, point 2 should have 0
        let p1 = scored_points.iter().find(|p| p.id == 1.into()).unwrap();
        let p2 = scored_points.iter().find(|p| p.id == 2.into()).unwrap();

        assert_eq!(p1.score, 0.5); // 1/(0+2)
        assert_eq!(p2.score, 0.0); // zero weight
    }

    #[test]
    fn exact_rrf_identity_breaks_score_ties() {
        let responses = vec![
            vec![make_scored_point(2, 0.0)],
            vec![make_scored_point(1, 0.0)],
        ];

        let scored_points = exact_rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();

        assert_eq!(scored_points[0].id, 1.into());
        assert_eq!(scored_points[1].id, 2.into());
    }

    #[test]
    fn exact_rrf_excludes_zero_score_identities() {
        let responses = vec![
            vec![make_scored_point(1, 0.0)],
            vec![make_scored_point(2, 0.0)],
        ];

        let scored_points = exact_rrf_scoring(responses, DEFAULT_RRF_K, Some(&[1.0, 0.0])).unwrap();

        assert_eq!(scored_points.len(), 1);
        assert_eq!(scored_points[0].id, 1.into());
    }

    #[test]
    fn test_dynamic_rrf_stops_early_for_identical_sources() {
        let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
        let sources = vec![source.clone(), source];

        let expected = exhaustive_top_k(&sources, 3, DEFAULT_RRF_K, None);
        let (actual, observed) = run_dynamic(&sources, 3, DEFAULT_RRF_K, None);

        assert_eq!(actual, expected);
        assert_eq!(observed, vec![3, 3]);
    }

    #[test]
    fn executor_stops_early_for_identical_sources() {
        let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
        let sources: Vec<ExactRrfStream<'_>> = vec![
            infallible_exact_rrf_stream(source.clone().into_iter()),
            infallible_exact_rrf_stream(source.clone().into_iter()),
        ];

        let execution = execute_dynamic_rrf(sources, 3, DEFAULT_RRF_K, None).unwrap();

        assert_eq!(execution.point_ids, source[..3]);
        assert_eq!(execution.source_pulls, vec![3, 3]);
        assert_eq!(execution.source_exhausted, vec![false, false]);
        assert_eq!(execution.stop_reason, DynamicRrfStopReason::TopKFixed);
    }

    #[test]
    fn pause_target_does_not_turn_an_incomplete_prefix_into_exact_eof() {
        let sources: Vec<ExactRrfStream<'_>> = vec![
            infallible_exact_rrf_stream([ExtendedPointId::from(1)].into_iter()),
            infallible_exact_rrf_stream(
                [
                    ExtendedPointId::from(2),
                    ExtendedPointId::from(3),
                    ExtendedPointId::from(4),
                ]
                .into_iter(),
            ),
        ];
        let mut session = DynamicRrfSession::new(
            sources,
            10,
            DEFAULT_RRF_K,
            None,
            DynamicRrfPolicy::default(),
        )
        .unwrap();

        let advance = session.advance_until_each(&[Some(1), Some(3)]).unwrap();

        assert!(matches!(advance, DynamicRrfAdvance::Paused));
        assert_eq!(session.source_pulls(), &[1, 3]);
        assert_eq!(session.source_is_exhausted(0), Some(false));
        assert_eq!(session.source_is_exhausted(1), Some(false));
    }

    #[test]
    fn recoverable_session_extends_an_exact_prefix_without_restarting_sources() {
        let first: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
        let second: Vec<_> = (0..100).rev().map(ExtendedPointId::from).collect();
        let expected = exhaustive_top_k(&[first.clone(), second.clone()], 12, DEFAULT_RRF_K, None);
        let sources: Vec<ExactRrfStream<'_>> = vec![
            infallible_exact_rrf_stream(first.into_iter()),
            infallible_exact_rrf_stream(second.into_iter()),
        ];
        let mut session =
            DynamicRrfSession::new(sources, 1, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
                .unwrap();
        let mut previous_pulls = vec![0; 2];

        for prefix in 1..=12 {
            let execution = session.run_to_prefix(prefix).unwrap();
            assert_eq!(execution.point_ids, expected[..prefix]);
            assert!(
                execution
                    .source_pulls
                    .iter()
                    .zip(&previous_pulls)
                    .all(|(current, previous)| current >= previous),
            );
            previous_pulls = execution.source_pulls;
        }
        assert!(session.run_to_prefix(11).is_err());
    }

    #[test]
    fn exhausted_session_reuses_one_complete_order_for_later_prefixes() {
        let first = vec![ExtendedPointId::from(1)];
        let second = vec![ExtendedPointId::from(1)];
        let expected = exhaustive_top_k(&[first.clone(), second.clone()], 3, DEFAULT_RRF_K, None);
        let sources: Vec<ExactRrfStream<'_>> = vec![
            infallible_exact_rrf_stream(first.into_iter()),
            infallible_exact_rrf_stream(second.into_iter()),
        ];
        let mut session =
            DynamicRrfSession::new(sources, 1, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
                .unwrap();

        let first = session.run_to_prefix(2).unwrap();
        assert_eq!(first.stop_reason, DynamicRrfStopReason::AllSourcesExhausted);
        let extended = session.run_to_prefix(3).unwrap();

        assert_eq!(extended.point_ids, expected);
        assert_eq!(extended.source_pulls, first.source_pulls);
        assert_eq!(extended.certification_checks, first.certification_checks);
    }

    #[test]
    fn resumable_session_replays_prefix_before_replacing_streams() {
        let first: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
        let second: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
        let expected = exhaustive_top_k(&[first.clone(), second.clone()], 5, DEFAULT_RRF_K, None);
        let sources: Vec<ExactRrfStream<'_>> = vec![
            infallible_exact_rrf_stream(first.clone().into_iter()),
            infallible_exact_rrf_stream(second.clone().into_iter()),
        ];
        let mut session =
            DynamicRrfSession::new(sources, 5, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
                .unwrap();

        let progress = session.advance_until_each(&[Some(2), Some(2)]).unwrap();

        assert_eq!(progress, DynamicRrfAdvance::Paused);
        assert_eq!(session.source_pulls(), &[2, 2]);
        session
            .replace_source(0, infallible_exact_rrf_stream(first.into_iter()))
            .unwrap();
        session
            .replace_source(1, infallible_exact_rrf_stream(second.into_iter()))
            .unwrap();
        assert_eq!(session.run_to_completion().unwrap().point_ids, expected);
    }

    #[test]
    fn resumable_session_rejects_replacement_prefix_mismatch() {
        let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
        let sources: Vec<ExactRrfStream<'_>> =
            vec![infallible_exact_rrf_stream(source.into_iter())];
        let mut session =
            DynamicRrfSession::new(sources, 5, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
                .unwrap();
        assert_eq!(
            session.advance_until_each(&[Some(2)]).unwrap(),
            DynamicRrfAdvance::Paused
        );
        let invalid: Vec<_> = [0_u64, 999, 2, 3, 4]
            .into_iter()
            .map(ExtendedPointId::from)
            .collect();

        let error = session
            .replace_source(0, infallible_exact_rrf_stream(invalid.into_iter()))
            .unwrap_err();

        assert!(error.to_string().contains("disagrees at rank 1"));
    }

    #[test]
    fn producer_failure_is_not_treated_as_exact_exhaustion() {
        let source: ExactRrfStream<'_> = Box::new(
            vec![
                Ok(ExtendedPointId::from(1_u64)),
                Err(OperationError::cancelled("test producer stopped")),
            ]
            .into_iter(),
        );
        let mut session = DynamicRrfSession::new(
            vec![source],
            2,
            DEFAULT_RRF_K,
            None,
            DynamicRrfPolicy::default(),
        )
        .unwrap();

        let error = session.run_to_completion().unwrap_err();

        assert!(matches!(error, OperationError::Cancelled { .. }));
        assert_eq!(session.source_pulls(), &[1]);
        assert_eq!(session.source_is_exhausted(0), Some(false));
    }

    #[test]
    fn competitor_scheduler_prefers_equal_progress_at_lower_cost() {
        let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
        let streams: Vec<ExactRrfStream<'_>> = vec![
            infallible_exact_rrf_stream(source.clone().into_iter()),
            infallible_exact_rrf_stream(source.clone().into_iter()),
        ];
        let mut session = DynamicRrfSession::new_with_source_costs(
            streams,
            3,
            DEFAULT_RRF_K,
            None,
            &[100.0, 1.0],
            DynamicRrfPolicy::default(),
        )
        .unwrap();

        let execution = session.run_to_completion().unwrap();

        assert_eq!(execution.point_ids, source[..3]);
        assert_eq!(session.source_schedule().first(), Some(&1));
    }

    #[test]
    fn source_costs_are_validated_at_the_exact_boundary() {
        let stream = || infallible_exact_rrf_stream(std::iter::once(ExtendedPointId::from(1_u64)));

        assert!(
            DynamicRrfSession::new_with_source_costs(
                vec![stream()],
                1,
                DEFAULT_RRF_K,
                None,
                &[],
                DynamicRrfPolicy::default(),
            )
            .is_err()
        );
        assert!(
            DynamicRrfSession::new_with_source_costs(
                vec![stream()],
                1,
                DEFAULT_RRF_K,
                None,
                &[0.0],
                DynamicRrfPolicy::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn executor_fuses_block_max_streams_exactly() {
        let documents = (0..64)
            .map(|id| SparseDocument {
                id,
                vector: RemappedSparseVector {
                    indices: vec![0, 1, 2],
                    values: vec![1.0, (id % 7) as f32, (63 - id) as f32],
                },
            })
            .collect();
        let index = BlockMaxIndex::build(documents, 8).unwrap();
        let first_query = RemappedSparseVector {
            indices: vec![0, 1],
            values: vec![1.0, 2.0],
        };
        let second_query = RemappedSparseVector {
            indices: vec![0, 2],
            values: vec![1.0, 0.5],
        };
        let ranked_ids = |query: RemappedSparseVector| {
            index
                .stream(query)
                .unwrap()
                .map(|point| ExtendedPointId::from(u64::from(point.idx)))
                .collect::<Vec<_>>()
        };
        let exhaustive_sources = vec![
            ranked_ids(first_query.clone()),
            ranked_ids(second_query.clone()),
        ];
        let expected = exhaustive_top_k(&exhaustive_sources, 7, DEFAULT_RRF_K, None);
        let sources: Vec<ExactRrfStream<'_>> = vec![
            Box::new(
                index
                    .stream(first_query)
                    .unwrap()
                    .map(|point| Ok(ExtendedPointId::from(u64::from(point.idx)))),
            ),
            Box::new(
                index
                    .stream(second_query)
                    .unwrap()
                    .map(|point| Ok(ExtendedPointId::from(u64::from(point.idx)))),
            ),
        ];

        let actual = execute_dynamic_rrf(sources, 7, DEFAULT_RRF_K, None).unwrap();

        assert_eq!(actual.point_ids, expected);
        assert!(
            actual.source_pulls.iter().sum::<usize>()
                < exhaustive_sources.iter().map(Vec::len).sum::<usize>()
        );
    }

    #[test]
    fn test_dynamic_rrf_handles_opposite_sources() {
        let ascending: Vec<_> = (0..20).map(ExtendedPointId::from).collect();
        let descending: Vec<_> = (0..20).rev().map(ExtendedPointId::from).collect();
        let sources = vec![ascending, descending];

        let expected = exhaustive_top_k(&sources, 5, DEFAULT_RRF_K, None);
        let (actual, _) = run_dynamic(&sources, 5, DEFAULT_RRF_K, None);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_dynamic_rrf_zero_weights_produce_no_positive_fusion_result() {
        let sources = vec![
            vec![3.into(), 2.into(), 1.into()],
            vec![2.into(), 3.into(), 1.into()],
        ];
        let weights = [0.0, 0.0];

        let expected = exhaustive_top_k(&sources, 3, DEFAULT_RRF_K, Some(&weights));
        let (actual, observed) = run_dynamic(&sources, 3, DEFAULT_RRF_K, Some(&weights));

        assert_eq!(actual, expected);
        assert!(actual.is_empty());
        assert_eq!(observed, vec![3, 3]);
    }

    #[test]
    fn test_dynamic_rrf_rejects_duplicate_point_in_source() {
        let mut state = DynamicRrfState::new(1, 1, DEFAULT_RRF_K, None).unwrap();
        state.observe(0, 1.into()).unwrap();

        let duplicate = state.observe(0, 1.into());

        assert!(duplicate.is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1_000))]

        #[test]
        fn dynamic_rrf_matches_exhaustive_rrf(
            (sources, weights, top_k, rrf_k) in random_rankings(false)
        ) {
            let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
            let (actual, _) = run_dynamic(&sources, top_k, rrf_k, Some(&weights));

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn dynamic_rrf_matches_exhaustive_rrf_with_partial_sources(
            (sources, weights, top_k, rrf_k) in random_rankings(true)
        ) {
            let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
            let (actual, _) = run_dynamic(&sources, top_k, rrf_k, Some(&weights));

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn batched_dynamic_rrf_executor_matches_exhaustive_rrf(
            (sources, weights, top_k, rrf_k) in random_rankings(true)
        ) {
            let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
            let streams: Vec<ExactRrfStream<'static>> = sources
                .into_iter()
                .map(|source| infallible_exact_rrf_stream(source.into_iter()))
                .collect();
            let actual = execute_dynamic_rrf(streams, top_k, rrf_k, Some(&weights)).unwrap();

            prop_assert_eq!(actual.point_ids, expected);
        }

        #[test]
        fn cost_aware_dynamic_rrf_matches_exhaustive_for_varied_costs(
            (sources, weights, top_k, rrf_k) in random_rankings(true),
        ) {
            let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
            let costs: Vec<_> = (0..sources.len())
                .map(|source| 1.0 + ((source * 7_919 + top_k * 97 + rrf_k * 53) % 999) as f64)
                .collect();
            let streams: Vec<ExactRrfStream<'static>> = sources
                .into_iter()
                .map(|source| infallible_exact_rrf_stream(source.into_iter()))
                .collect();
            let actual = DynamicRrfSession::new_with_source_costs(
                streams,
                top_k,
                rrf_k,
                Some(&weights),
                &costs,
                DynamicRrfPolicy::default(),
            )
            .unwrap()
            .run_to_completion()
            .unwrap();

            prop_assert_eq!(actual.point_ids, expected);
        }
    }
}
