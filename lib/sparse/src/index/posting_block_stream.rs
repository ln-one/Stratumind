// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Resumable exact score-ordered stream over compressed Sparse postings.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use common::universal_io::Result;
use ordered_float::OrderedFloat;
use serde::Serialize;

use super::inverted_index::InvertedIndex;
use super::posting_list_common::PostingListIter;
use crate::SearchScratchArena;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimId, DimWeight};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct PostingBlockStreamTelemetry {
    pub posting_lists: usize,
    pub batches: usize,
    pub bound_evaluations: usize,
    pub zero_bound_batches: usize,
    pub batches_expanded: usize,
    pub posting_elements_visited: usize,
    pub nonzero_documents_scored: usize,
    pub points_emitted: usize,
    pub max_pending_batches: usize,
    pub max_pending_points: usize,
    pub max_buffered_points: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingBatch {
    start: PointOffsetType,
    end: PointOffsetType,
    upper_bound: f32,
}

impl Eq for PendingBatch {}

impl Ord for PendingBatch {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.upper_bound)
            .cmp(&OrderedFloat(other.upper_bound))
            .then_with(|| other.start.cmp(&self.start))
    }
}

impl PartialOrd for PendingBatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingPoint {
    point: ScoredPointOffset,
    batch_index: usize,
}

impl Eq for PendingPoint {}

impl Ord for PendingPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.point.score)
            .cmp(&OrderedFloat(other.point.score))
            .then_with(|| other.point.idx.cmp(&self.point.idx))
    }
}

impl PartialOrd for PendingPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingBlockPoint(ScoredPointOffset);

impl Eq for PendingBlockPoint {}

impl Ord for PendingBlockPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.idx.cmp(&self.0.idx))
    }
}

impl PartialOrd for PendingBlockPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
struct WeightedPosting<T> {
    iterator: T,
    query_weight: DimWeight,
}

pub struct PostingBlockStream<'a, I: InvertedIndex> {
    postings: Vec<WeightedPosting<I::Iter<'a>>>,
    pending_batches: BinaryHeap<PendingBatch>,
    pending_points: BinaryHeap<PendingPoint>,
    active_batches: Vec<BinaryHeap<PendingBlockPoint>>,
    buffered_points: usize,
    telemetry: PostingBlockStreamTelemetry,
}

impl<'a, I: InvertedIndex> PostingBlockStream<'a, I> {
    pub fn new(
        index: &'a I,
        query: RemappedSparseVector,
        batch_size: usize,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
    ) -> Result<Self> {
        assert!(batch_size > 0, "posting stream batch size must be positive");
        assert!(
            query.indices.len() == query.values.len()
                && query.is_sorted()
                && query
                    .values
                    .iter()
                    .all(|weight| weight.is_finite() && *weight >= 0.0),
            "posting stream requires sorted finite non-negative query weights"
        );
        let batch_width: PointOffsetType = batch_size
            .try_into()
            .expect("posting stream batch size exceeds PointOffsetType");
        let postings = open_postings(index, &query, arena, hardware_counter)?;
        let mut bound_postings = postings.clone();
        let mut telemetry = PostingBlockStreamTelemetry {
            posting_lists: postings.len(),
            ..Default::default()
        };
        let min_id = bound_postings
            .iter_mut()
            .filter_map(|posting| posting.iterator.peek().map(|element| element.record_id))
            .min();
        let max_id = bound_postings
            .iter()
            .filter_map(|posting| posting.iterator.last_id())
            .max();
        let mut batches = Vec::new();
        if let (Some(mut start), Some(max_id)) = (min_id, max_id) {
            loop {
                let end = start
                    .saturating_add(batch_width.saturating_sub(1))
                    .min(max_id);
                let mut upper_bound = 0.0;
                for posting in &mut bound_postings {
                    if let Some(max_weight) = posting.iterator.max_weight_till_id(end) {
                        upper_bound += max_weight * posting.query_weight;
                    }
                }
                telemetry.batches += 1;
                telemetry.bound_evaluations += bound_postings.len();
                if upper_bound > 0.0 {
                    batches.push(PendingBatch {
                        start,
                        end,
                        upper_bound,
                    });
                } else {
                    telemetry.zero_bound_batches += 1;
                }
                for posting in &mut bound_postings {
                    posting.iterator.skip_till_id(end);
                }
                if end == max_id {
                    break;
                }
                start = end + 1;
            }
        }
        telemetry.max_pending_batches = batches.len();
        Ok(Self {
            postings,
            pending_batches: BinaryHeap::from(batches),
            pending_points: BinaryHeap::new(),
            active_batches: Vec::new(),
            buffered_points: 0,
            telemetry,
        })
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        self.telemetry
    }

    fn next_point_is_fixed(&self) -> bool {
        let Some(point) = self.pending_points.peek() else {
            return false;
        };
        let Some(batch) = self.pending_batches.peek() else {
            return true;
        };
        point.point.score > batch.upper_bound
            || (point.point.score == batch.upper_bound && point.point.idx < batch.start)
    }

    fn expand_next_batch(&mut self) {
        let Some(batch) = self.pending_batches.pop() else {
            return;
        };
        self.telemetry.batches_expanded += 1;
        let mut postings = self.postings.clone();
        let mut scores = vec![0.0; (batch.end - batch.start + 1) as usize];
        for posting in &mut postings {
            posting.iterator.skip_to(batch.start);
            let elements_before = posting.iterator.len_to_end();
            let query_weight = posting.query_weight;
            posting.iterator.for_each_till_id(
                batch.end,
                scores.as_mut_slice(),
                |scores, id, weight| {
                    scores[(id - batch.start) as usize] += weight * query_weight;
                },
            );
            self.telemetry.posting_elements_visited +=
                elements_before - posting.iterator.len_to_end();
        }
        let mut points = Vec::new();
        for (offset, score) in scores.into_iter().enumerate() {
            if score == 0.0 {
                continue;
            }
            let id = batch.start + offset as PointOffsetType;
            assert!(
                score <= batch.upper_bound,
                "posting block certificate violated for point {id}: score {score}, bound {}",
                batch.upper_bound
            );
            self.telemetry.nonzero_documents_scored += 1;
            points.push(ScoredPointOffset { idx: id, score });
        }
        let mut points = BinaryHeap::from(
            points
                .into_iter()
                .map(PendingBlockPoint)
                .collect::<Vec<_>>(),
        );
        if let Some(point) = points.pop().map(|point| point.0) {
            let batch_index = self.active_batches.len();
            self.buffered_points += points.len() + 1;
            self.active_batches.push(points);
            self.pending_points
                .push(PendingPoint { point, batch_index });
        }
        self.telemetry.max_pending_points = self
            .telemetry
            .max_pending_points
            .max(self.pending_points.len());
        self.telemetry.max_buffered_points =
            self.telemetry.max_buffered_points.max(self.buffered_points);
    }

    fn pop_next_point(&mut self) -> Option<ScoredPointOffset> {
        let pending = self.pending_points.pop()?;
        self.buffered_points -= 1;
        if let Some(point) = self.active_batches[pending.batch_index]
            .pop()
            .map(|point| point.0)
        {
            self.pending_points.push(PendingPoint {
                point,
                batch_index: pending.batch_index,
            });
        }
        self.telemetry.points_emitted += 1;
        Some(pending.point)
    }
}

impl<I: InvertedIndex> Iterator for PostingBlockStream<'_, I> {
    type Item = ScoredPointOffset;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_point_is_fixed() {
                return self.pop_next_point();
            }
            if self.pending_batches.is_empty() {
                return self.pop_next_point();
            }
            self.expand_next_batch();
        }
    }
}

fn open_postings<'a, I: InvertedIndex>(
    index: &'a I,
    query: &RemappedSparseVector,
    arena: &'a SearchScratchArena,
    hardware_counter: &'a HardwareCounterCell,
) -> Result<Vec<WeightedPosting<I::Iter<'a>>>> {
    let mut postings = Vec::with_capacity(query.indices.len());
    for (&id, &query_weight) in query.indices.iter().zip(&query.values) {
        let iterator = index.get(id as DimId, arena, hardware_counter)?;
        if iterator.last_id().is_some() {
            postings.push(WeightedPosting {
                iterator,
                query_weight,
            });
        }
    }
    Ok(postings)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    #[test]
    fn compressed_posting_stream_matches_exhaustive_order() {
        let mut builder = InvertedIndexBuilder::new();
        let mut documents = Vec::new();
        for id in 0..30_000u32 {
            let vector = RemappedSparseVector {
                indices: vec![0, 1, 2],
                values: vec![1.0, 100.0 - id as f32 / 400.0, (id % 17) as f32 / 3.0],
            };
            builder.add(id, vector.clone());
            documents.push((id, vector));
        }
        let ram = builder.build();
        let temp = tempfile::tempdir().unwrap();
        let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
            Cow::Owned(ram),
            temp.path(),
        )
        .unwrap();
        let query = RemappedSparseVector {
            indices: vec![0, 1, 2],
            values: vec![1.0, 2.0, 0.5],
        };
        let mut expected: Vec<_> = documents
            .iter()
            .filter_map(|(id, vector)| {
                vector
                    .score(&query)
                    .filter(|score| *score != 0.0)
                    .map(|score| ScoredPointOffset { idx: *id, score })
            })
            .collect();
        expected.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.idx.cmp(&right.idx))
        });
        let hardware_counter = HardwareCounterCell::disposable();
        let arena = SearchScratchArena::new_slow();
        let mut stream =
            PostingBlockStream::new(&index, query.clone(), 4096, &arena, &hardware_counter)
                .unwrap();

        let actual: Vec<_> = stream.by_ref().collect();

        assert_eq!(actual, expected);

        let mut top_stream =
            PostingBlockStream::new(&index, query, 4096, &arena, &hardware_counter).unwrap();
        let actual_top: Vec<_> = top_stream.by_ref().take(20).collect();
        assert_eq!(actual_top, expected[..20]);
        assert!(top_stream.telemetry().batches_expanded < top_stream.telemetry().batches);
    }
}
