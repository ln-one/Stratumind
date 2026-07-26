// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use ahash::{AHashMap, AHashSet};
use ordered_float::OrderedFloat;

use super::types::ChannelRankBatch;
use super::{guaranteed_before, position_score, rrf_order};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::types::ExtendedPointId;

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
    pub(super) rrf_k: usize,
    pub(super) top_k: usize,
    pub(super) weights: Vec<f32>,
    pub(super) next_positions: Vec<usize>,
    pub(super) exhausted: Vec<bool>,
    points: AHashMap<ExtendedPointId, PartialRrfPoint>,
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

    /// Atomically applies a contiguous exact-rank batch.
    pub fn observe_batch(
        &mut self,
        source: usize,
        batch: &ChannelRankBatch,
    ) -> OperationResult<()> {
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
        if batch.start_rank != self.next_positions[source] {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF source {source} batch starts at rank {}; expected {}",
                batch.start_rank, self.next_positions[source],
            )));
        }
        if batch.point_ids.is_empty() && !batch.exhausted {
            return Err(OperationError::validation_error(format!(
                "Dynamic RRF source {source} returned an empty non-exhausted batch",
            )));
        }

        let mut unique = AHashSet::with_capacity(batch.point_ids.len());
        for id in &batch.point_ids {
            if !unique.insert(*id) {
                return Err(OperationError::validation_error(format!(
                    "Point {id} occurs more than once in one Dynamic RRF source {source} batch",
                )));
            }
            if self
                .points
                .get(id)
                .is_some_and(|point| point.contributions[source].is_some())
            {
                return Err(OperationError::validation_error(format!(
                    "Point {id} occurs more than once in Dynamic RRF source {source}",
                )));
            }
        }

        for (offset, id) in batch.point_ids.iter().copied().enumerate() {
            let contribution =
                position_score(batch.start_rank + offset, self.rrf_k, self.weights[source]);
            let point = self.points.entry(id).or_insert_with(|| PartialRrfPoint {
                contributions: vec![None; self.weights.len()],
            });
            point.contributions[source] = Some(contribution);
        }
        self.next_positions[source] += batch.point_ids.len();
        if batch.exhausted {
            self.exhausted[source] = true;
        }
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
    pub(super) fn competitive_missing_counts(&self) -> Vec<usize> {
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

    pub(super) fn complete_order(&self) -> Vec<ExtendedPointId> {
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
