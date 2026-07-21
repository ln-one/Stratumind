// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Correctness-neutral cost routing from short exact channel prefixes.

use std::collections::HashSet;

use crate::types::ExtendedPointId;

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
}
