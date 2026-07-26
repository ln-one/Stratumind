// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::collections::hash_map::Entry;

use ahash::AHashMap;
use itertools::Either;
use ordered_float::OrderedFloat;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::types::{ExtendedPointId, ScoredPoint};

pub(super) fn position_score(position: usize, k: usize, weight: f32) -> f32 {
    // Avoid division by zero - if weight is 0, treat as negligible contribution
    if weight <= 0.0 {
        return 0.0;
    }

    1.0 / ((position + 1) as f32 / weight + k as f32 - 1.0)
}

pub(super) fn rrf_order(
    left_id: ExtendedPointId,
    left_score: f32,
    right_id: ExtendedPointId,
    right_score: f32,
) -> std::cmp::Ordering {
    OrderedFloat(right_score)
        .cmp(&OrderedFloat(left_score))
        .then_with(|| left_id.cmp(&right_id))
}

pub(super) fn guaranteed_before(
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
