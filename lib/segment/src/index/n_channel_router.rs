// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Correctness-neutral cost routing from short exact channel prefixes.

use std::collections::HashSet;

use crate::common::reciprocal_rank_fusion::DynamicRrfScheduler;
use crate::types::ExtendedPointId;

/// Every variant names an exact-equivalent physical implementation. The
/// router cannot select an approximate plan, so a bad estimate can only add
/// work or latency.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExactSparseDecodePlan {
    #[default]
    IndependentNative,
    SharedNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SafeExactPlan {
    pub scheduler: DynamicRrfScheduler,
    pub sparse_decode: ExactSparseDecodePlan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafeExactRouter {
    cost_ratio_for_competitor_scheduler: f64,
    shared_sparse_max_unique_to_independent_milli: u16,
}

impl Default for SafeExactRouter {
    fn default() -> Self {
        Self {
            cost_ratio_for_competitor_scheduler: 1.25,
            shared_sparse_max_unique_to_independent_milli: 950,
        }
    }
}

impl SafeExactRouter {
    pub fn new(
        cost_ratio_for_competitor_scheduler: f64,
        shared_sparse_max_unique_to_independent_milli: u16,
    ) -> Self {
        assert!(
            cost_ratio_for_competitor_scheduler.is_finite()
                && cost_ratio_for_competitor_scheduler >= 1.0,
            "cost ratio threshold must be finite and at least one"
        );
        assert!(
            shared_sparse_max_unique_to_independent_milli <= 1_000,
            "Sparse sharing threshold must be in [0, 1000]"
        );
        Self {
            cost_ratio_for_competitor_scheduler,
            shared_sparse_max_unique_to_independent_milli,
        }
    }

    pub fn decide(
        &self,
        source_costs: &[f64],
        scheduler_costs_are_incremental: bool,
        sparse_unique_posting_elements: usize,
        sparse_independent_posting_elements: usize,
    ) -> SafeExactPlan {
        let (minimum_cost, maximum_cost) = source_costs
            .iter()
            .copied()
            .filter(|cost| cost.is_finite() && *cost > 0.0)
            .fold((f64::INFINITY, 0.0f64), |(minimum, maximum), cost| {
                (minimum.min(cost), maximum.max(cost))
            });
        let heterogeneous_costs = scheduler_costs_are_incremental
            && source_costs.len() > 1
            && minimum_cost.is_finite()
            && maximum_cost / minimum_cost >= self.cost_ratio_for_competitor_scheduler;
        let share_sparse = sparse_independent_posting_elements > 0
            && sparse_unique_posting_elements.saturating_mul(1_000)
                <= sparse_independent_posting_elements
                    .saturating_mul(self.shared_sparse_max_unique_to_independent_milli as usize);

        SafeExactPlan {
            scheduler: if heterogeneous_costs {
                DynamicRrfScheduler::CompetitorCostAware
            } else {
                DynamicRrfScheduler::MaxNextContribution
            },
            sparse_decode: if share_sparse {
                ExactSparseDecodePlan::SharedNative
            } else {
                ExactSparseDecodePlan::IndependentNative
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixExecutionChoice {
    Dynamic,
    Exhaustive,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefixAgreementDecision {
    pub choice: PrefixExecutionChoice,
    pub average_pairwise_overlap: f64,
    pub minimum_pairwise_overlap: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefixAgreementRouter {
    minimum_overlap: f64,
}

impl PrefixAgreementRouter {
    pub fn new(minimum_overlap: f64) -> Self {
        assert!(
            minimum_overlap.is_finite() && (0.0..=1.0).contains(&minimum_overlap),
            "prefix overlap threshold must be finite and in [0, 1]"
        );
        Self { minimum_overlap }
    }

    pub fn decide(&self, prefixes: &[Vec<ExtendedPointId>]) -> PrefixAgreementDecision {
        let average_overlap = average_pairwise_overlap(prefixes);
        let minimum_overlap = minimum_pairwise_overlap(prefixes);
        PrefixAgreementDecision {
            choice: if minimum_overlap >= self.minimum_overlap {
                PrefixExecutionChoice::Dynamic
            } else {
                PrefixExecutionChoice::Exhaustive
            },
            average_pairwise_overlap: average_overlap,
            minimum_pairwise_overlap: minimum_overlap,
        }
    }
}

pub fn minimum_pairwise_overlap(prefixes: &[Vec<ExtendedPointId>]) -> f64 {
    pairwise_overlaps(prefixes)
        .into_iter()
        .reduce(f64::min)
        .unwrap_or(1.0)
}

pub fn average_pairwise_overlap(prefixes: &[Vec<ExtendedPointId>]) -> f64 {
    if prefixes.len() <= 1 {
        return 1.0;
    }
    let overlaps = pairwise_overlaps(prefixes);
    overlaps.iter().sum::<f64>() / overlaps.len() as f64
}

fn pairwise_overlaps(prefixes: &[Vec<ExtendedPointId>]) -> Vec<f64> {
    if prefixes.len() <= 1 {
        return vec![1.0];
    }
    let sets: Vec<HashSet<_>> = prefixes
        .iter()
        .map(|prefix| prefix.iter().copied().collect())
        .collect();
    let mut overlaps = Vec::new();
    for left in 0..sets.len() {
        for right in left + 1..sets.len() {
            let denominator = sets[left].len().min(sets[right].len());
            let overlap = if denominator == 0 {
                0.0
            } else {
                sets[left].intersection(&sets[right]).count() as f64 / denominator as f64
            };
            overlaps.push(overlap);
        }
    }
    overlaps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(values: &[u64]) -> Vec<ExtendedPointId> {
        values.iter().copied().map(ExtendedPointId::from).collect()
    }

    #[test]
    fn routes_identical_prefixes_to_dynamic() {
        let router = PrefixAgreementRouter::new(0.25);
        let decision = router.decide(&[ids(&[1, 2, 3]), ids(&[1, 2, 3])]);
        assert_eq!(decision.choice, PrefixExecutionChoice::Dynamic);
        assert_eq!(decision.average_pairwise_overlap, 1.0);
        assert_eq!(decision.minimum_pairwise_overlap, 1.0);
    }

    #[test]
    fn routes_disjoint_prefixes_to_exhaustive() {
        let router = PrefixAgreementRouter::new(0.25);
        let decision = router.decide(&[ids(&[1, 2, 3]), ids(&[4, 5, 6])]);
        assert_eq!(decision.choice, PrefixExecutionChoice::Exhaustive);
        assert_eq!(decision.average_pairwise_overlap, 0.0);
        assert_eq!(decision.minimum_pairwise_overlap, 0.0);
    }

    #[test]
    fn averages_all_channel_pairs() {
        let overlap =
            average_pairwise_overlap(&[ids(&[1, 2, 3, 4]), ids(&[1, 2, 5, 6]), ids(&[1, 7, 8, 9])]);
        assert!((overlap - 1.0 / 3.0).abs() < f64::EPSILON);
        assert_eq!(
            minimum_pairwise_overlap(
                &[ids(&[1, 2, 3, 4]), ids(&[1, 2, 5, 6]), ids(&[1, 7, 8, 9]),]
            ),
            0.25
        );
    }

    #[test]
    fn safe_router_uses_cost_aware_schedule_for_heterogeneous_channels() {
        let plan = SafeExactRouter::default().decide(&[768.0, 3.0], true, 100, 100);

        assert_eq!(plan.scheduler, DynamicRrfScheduler::CompetitorCostAware);
        assert_eq!(plan.sparse_decode, ExactSparseDecodePlan::IndependentNative);
    }

    #[test]
    fn safe_router_does_not_treat_eager_preparation_as_marginal_pull_cost() {
        let plan = SafeExactRouter::default().decide(&[768.0, 3.0], false, 100, 100);

        assert_eq!(plan.scheduler, DynamicRrfScheduler::MaxNextContribution);
    }

    #[test]
    fn safe_router_selects_shared_native_decode_only_for_material_savings() {
        let router = SafeExactRouter::default();

        assert_eq!(
            router.decide(&[1.0, 1.0], true, 90, 100).sparse_decode,
            ExactSparseDecodePlan::SharedNative
        );
        assert_eq!(
            router.decide(&[1.0, 1.0], true, 99, 100).sparse_decode,
            ExactSparseDecodePlan::IndependentNative
        );
    }
}
