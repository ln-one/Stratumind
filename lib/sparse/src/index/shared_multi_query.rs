// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact full Sparse rankings for several queries with one posting decode per unique term.
//!
//! This is a physical fallback for overlapping query rewrites. It preserves one
//! independent logical ranking per channel while sharing compressed posting reads.

use std::collections::BTreeMap;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use ordered_float::OrderedFloat;

use crate::SearchScratchArena;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimOffset, DimWeight};
use crate::index::inverted_index::InvertedIndex;
use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use crate::index::posting_list_common::PostingListIter;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedSparseTelemetry {
    pub channels: usize,
    pub unique_terms: usize,
    pub logical_term_accesses: usize,
    pub physical_posting_elements_decoded: usize,
    pub logical_posting_elements: usize,
    pub nonzero_channel_scores: usize,
}

pub struct SharedSparseRankings {
    pub rankings: Vec<Vec<ScoredPointOffset>>,
    pub telemetry: SharedSparseTelemetry,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedSparseCostEstimate {
    pub unique_terms: usize,
    pub logical_term_accesses: usize,
    pub unique_posting_elements: usize,
    pub logical_posting_elements: usize,
}

pub fn estimate_shared_sparse_cost(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    queries: &[RemappedSparseVector],
    hardware_counter: &HardwareCounterCell,
) -> Result<SharedSparseCostEstimate, String> {
    let term_multiplicity = sparse_term_multiplicity(index, queries)?;
    estimate_from_term_multiplicity(index, term_multiplicity, hardware_counter)
}

pub fn estimate_shared_sparse_cost_pair(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    logical_queries: &[RemappedSparseVector],
    deduplicated_queries: &[RemappedSparseVector],
    hardware_counter: &HardwareCounterCell,
) -> Result<(SharedSparseCostEstimate, SharedSparseCostEstimate), String> {
    let logical = sparse_term_multiplicity(index, logical_queries)?;
    let deduplicated = sparse_term_multiplicity(index, deduplicated_queries)?;
    let mut logical_estimate = SharedSparseCostEstimate {
        unique_terms: logical.len(),
        logical_term_accesses: logical.values().sum(),
        ..Default::default()
    };
    let mut deduplicated_estimate = SharedSparseCostEstimate {
        unique_terms: deduplicated.len(),
        logical_term_accesses: deduplicated.values().sum(),
        ..Default::default()
    };
    let mut posting_lengths = BTreeMap::<DimOffset, usize>::new();
    for term in logical.keys().chain(deduplicated.keys()) {
        if posting_lengths.contains_key(term) {
            continue;
        }
        let posting_len = index
            .posting_list_len(*term, hardware_counter)
            .map_err(|error| error.to_string())?;
        posting_lengths.insert(*term, posting_len);
    }
    for (term, multiplicity) in logical {
        let posting_len = posting_lengths[&term];
        logical_estimate.unique_posting_elements += posting_len;
        logical_estimate.logical_posting_elements += posting_len * multiplicity;
    }
    for (term, multiplicity) in deduplicated {
        let posting_len = posting_lengths[&term];
        deduplicated_estimate.unique_posting_elements += posting_len;
        deduplicated_estimate.logical_posting_elements += posting_len * multiplicity;
    }
    Ok((logical_estimate, deduplicated_estimate))
}

fn sparse_term_multiplicity(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    queries: &[RemappedSparseVector],
) -> Result<BTreeMap<DimOffset, usize>, String> {
    let mut term_multiplicity = BTreeMap::<DimOffset, usize>::new();
    for (channel, query) in queries.iter().enumerate() {
        if !query.is_sorted()
            || query
                .values
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(format!(
                "Sparse channel {channel} is not sorted finite non-negative"
            ));
        }
        for (&term, &weight) in query.indices.iter().zip(&query.values) {
            if weight != 0.0 && (term as usize) < index.len() {
                *term_multiplicity.entry(term).or_default() += 1;
            }
        }
    }
    Ok(term_multiplicity)
}

fn estimate_from_term_multiplicity(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    term_multiplicity: BTreeMap<DimOffset, usize>,
    hardware_counter: &HardwareCounterCell,
) -> Result<SharedSparseCostEstimate, String> {
    let mut estimate = SharedSparseCostEstimate {
        unique_terms: term_multiplicity.len(),
        logical_term_accesses: term_multiplicity.values().sum(),
        ..Default::default()
    };
    for (term, multiplicity) in term_multiplicity {
        let posting_len = index
            .posting_list_len(term, hardware_counter)
            .map_err(|error| error.to_string())?;
        estimate.unique_posting_elements += posting_len;
        estimate.logical_posting_elements += posting_len * multiplicity;
    }
    Ok(estimate)
}

pub fn evaluate_shared_sparse(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    queries: &[RemappedSparseVector],
    point_capacity: usize,
    arena: &SearchScratchArena,
    hardware_counter: &HardwareCounterCell,
) -> Result<SharedSparseRankings, String> {
    let mut term_channels = BTreeMap::<DimOffset, Vec<(usize, DimWeight)>>::new();
    for (channel, query) in queries.iter().enumerate() {
        if !query.is_sorted()
            || query
                .values
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(format!(
                "Sparse channel {channel} is not sorted finite non-negative"
            ));
        }
        for (&term, &weight) in query.indices.iter().zip(&query.values) {
            if weight != 0.0 && (term as usize) < index.len() {
                term_channels
                    .entry(term)
                    .or_default()
                    .push((channel, weight));
            }
        }
    }

    let mut scores = vec![vec![0.0_f32; point_capacity]; queries.len()];
    let mut touched = vec![Vec::<PointOffsetType>::new(); queries.len()];
    let mut telemetry = SharedSparseTelemetry {
        channels: queries.len(),
        unique_terms: term_channels.len(),
        logical_term_accesses: term_channels.values().map(Vec::len).sum(),
        ..Default::default()
    };

    for (term, channels) in term_channels {
        let posting_len = index
            .posting_list_len(term, hardware_counter)
            .map_err(|error| error.to_string())?;
        telemetry.logical_posting_elements += posting_len * channels.len();
        let posting = index
            .get(term, arena, hardware_counter)
            .map_err(|error| error.to_string())?;
        for element in posting.into_std_iter() {
            telemetry.physical_posting_elements_decoded += 1;
            let ordinal = element.record_id as usize;
            if ordinal >= point_capacity {
                return Err(format!(
                    "Sparse point {} exceeds accumulator capacity {point_capacity}",
                    element.record_id
                ));
            }
            for &(channel, query_weight) in &channels {
                let contribution = element.weight * query_weight;
                if contribution == 0.0 {
                    continue;
                }
                let slot = &mut scores[channel][ordinal];
                if *slot == 0.0 {
                    touched[channel].push(element.record_id);
                }
                *slot += contribution;
            }
        }
    }

    let rankings = touched
        .into_iter()
        .enumerate()
        .map(|(channel, ids)| {
            let mut ranking: Vec<_> = ids
                .into_iter()
                .filter_map(|idx| {
                    let score = scores[channel][idx as usize];
                    (score != 0.0).then_some(ScoredPointOffset { idx, score })
                })
                .collect();
            ranking.sort_unstable_by(|left, right| {
                OrderedFloat(right.score)
                    .cmp(&OrderedFloat(left.score))
                    .then_with(|| left.idx.cmp(&right.idx))
            });
            ranking
        })
        .collect::<Vec<_>>();
    telemetry.nonzero_channel_scores = rankings.iter().map(Vec::len).sum();
    Ok(SharedSparseRankings {
        rankings,
        telemetry,
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::index::block_max::{BlockMaxIndex, SparseDocument};
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    #[test]
    fn shared_decode_matches_independent_exact_rankings() {
        let documents: Vec<_> = (0..193)
            .map(|id| SparseDocument {
                id,
                vector: RemappedSparseVector {
                    indices: vec![0, 2, 5],
                    values: vec![1.0, (id % 11) as f32, ((id * 7) % 17) as f32],
                },
            })
            .collect();
        let ram = InvertedIndexBuilder::build_from_iterator(
            documents
                .iter()
                .map(|document| (document.id, document.vector.clone())),
        );
        let postings =
            InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(Cow::Owned(ram), ".")
                .unwrap();
        let block = BlockMaxIndex::build(documents, 17).unwrap();
        let queries = vec![
            RemappedSparseVector {
                indices: vec![0, 2],
                values: vec![1.0, 2.0],
            },
            RemappedSparseVector {
                indices: vec![0, 5],
                values: vec![0.5, 3.0],
            },
            RemappedSparseVector {
                indices: vec![0, 2, 5],
                values: vec![1.0, 1.0, 1.0],
            },
        ];
        let arena = SearchScratchArena::new_slow();
        let hardware_counter = HardwareCounterCell::disposable();

        let actual =
            evaluate_shared_sparse(&postings, &queries, 193, &arena, &hardware_counter).unwrap();
        let estimate = estimate_shared_sparse_cost(&postings, &queries, &hardware_counter).unwrap();
        let (paired_estimate, paired_deduplicated_estimate) =
            estimate_shared_sparse_cost_pair(&postings, &queries, &queries, &hardware_counter)
                .unwrap();
        let expected: Vec<Vec<_>> = queries
            .into_iter()
            .map(|query| block.stream(query).unwrap().collect())
            .collect();

        assert_eq!(actual.rankings, expected);
        assert!(
            actual.telemetry.physical_posting_elements_decoded
                < actual.telemetry.logical_posting_elements
        );
        assert_eq!(
            estimate.unique_posting_elements,
            actual.telemetry.physical_posting_elements_decoded
        );
        assert_eq!(
            estimate.logical_posting_elements,
            actual.telemetry.logical_posting_elements
        );
        assert_eq!(paired_estimate, estimate);
        assert_eq!(paired_deduplicated_estimate, estimate);
    }
}
