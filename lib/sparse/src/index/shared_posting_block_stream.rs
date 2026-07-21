// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact lazy Sparse streams that share one materialization per document batch.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::rc::Rc;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use common::universal_io::Result;
use ordered_float::OrderedFloat;

use crate::SearchScratchArena;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimId, DimWeight};
use crate::index::inverted_index::InvertedIndex;
use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use crate::index::posting_list_common::PostingListIter;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedPostingBlockTelemetry {
    pub channels: usize,
    pub unique_terms: usize,
    pub logical_term_accesses: usize,
    pub batches: usize,
    pub unique_batch_materializations: usize,
    pub reused_batch_materializations: usize,
    pub physical_posting_elements_decoded: usize,
    pub logical_posting_elements_decoded: usize,
    pub physical_score_multiplications: usize,
    pub logical_score_multiplications: usize,
    pub physical_bound_evaluations: usize,
    pub logical_bound_evaluations: usize,
    pub nonzero_channel_scores: usize,
    pub cached_channel_points: usize,
}

struct TermConsumers {
    channels: Vec<(usize, DimWeight)>,
    weight_groups: Vec<(DimWeight, Vec<usize>)>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedPostingChannelTelemetry {
    pub bound_evaluations: usize,
    pub zero_bound_batches: usize,
    pub batches_requested: usize,
    pub points_emitted: usize,
    pub max_buffered_points: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingBatch {
    index: usize,
    start: PointOffsetType,
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
    batch_slot: usize,
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
struct BatchPoint(ScoredPointOffset);

impl Eq for BatchPoint {}

impl Ord for BatchPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.idx.cmp(&self.0.idx))
    }
}

impl PartialOrd for BatchPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Coordinator<'a> {
    index: &'a InvertedIndexCompressedImmutableRam<f32>,
    term_channels: BTreeMap<DimId, TermConsumers>,
    channel_count: usize,
    batch_size: usize,
    max_id: PointOffsetType,
    cache: Vec<Option<Rc<Vec<Vec<ScoredPointOffset>>>>>,
    arena: &'a SearchScratchArena,
    hardware_counter: &'a HardwareCounterCell,
    telemetry: SharedPostingBlockTelemetry,
}

impl Coordinator<'_> {
    fn materialize(&mut self, batch_index: usize) -> Result<Rc<Vec<Vec<ScoredPointOffset>>>> {
        if let Some(cached) = self.cache[batch_index].as_ref() {
            self.telemetry.reused_batch_materializations += 1;
            return Ok(Rc::clone(cached));
        }
        self.telemetry.unique_batch_materializations += 1;
        let start = batch_index * self.batch_size;
        let start: PointOffsetType = start.try_into().expect("Sparse batch start exceeds u32");
        let end = start
            .saturating_add(self.batch_size.saturating_sub(1) as PointOffsetType)
            .min(self.max_id);
        let width = (end - start + 1) as usize;
        // A posting updates several channels for the same document. Keep those
        // channel slots adjacent so the hot write loop does not jump between
        // channel-sized arrays.
        let mut scores = vec![0.0_f32; width * self.channel_count];
        for (&term, consumers) in &self.term_channels {
            let mut posting = self.index.get(term, self.arena, self.hardware_counter)?;
            if start > 0 {
                posting.skip_till_id(start - 1);
            }
            let before = posting.len_to_end();
            posting.for_each_till_id(end, scores.as_mut_slice(), |scores, id, weight| {
                let offset = (id - start) as usize;
                for (query_weight, channels) in &consumers.weight_groups {
                    let contribution = weight * *query_weight;
                    for &channel in channels {
                        scores[offset * self.channel_count + channel] += contribution;
                    }
                }
            });
            let decoded = before - posting.len_to_end();
            self.telemetry.physical_posting_elements_decoded += decoded;
            self.telemetry.logical_posting_elements_decoded += decoded * consumers.channels.len();
            self.telemetry.physical_score_multiplications +=
                decoded * consumers.weight_groups.len();
            self.telemetry.logical_score_multiplications += decoded * consumers.channels.len();
        }
        let channel_points = (0..self.channel_count)
            .map(|channel| {
                (0..width)
                    .filter_map(|offset| {
                        let score = scores[offset * self.channel_count + channel];
                        (score != 0.0).then_some(ScoredPointOffset {
                            idx: start + offset as PointOffsetType,
                            score,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let point_count = channel_points.iter().map(Vec::len).sum::<usize>();
        self.telemetry.nonzero_channel_scores += point_count;
        self.telemetry.cached_channel_points += point_count;
        let cached = Rc::new(channel_points);
        self.cache[batch_index] = Some(Rc::clone(&cached));
        Ok(cached)
    }
}

#[derive(Clone)]
pub struct SharedPostingBlockHandle<'a>(Rc<RefCell<Coordinator<'a>>>);

impl SharedPostingBlockHandle<'_> {
    pub fn telemetry(&self) -> SharedPostingBlockTelemetry {
        self.0.borrow().telemetry
    }
}

pub struct SharedPostingBlockStream<'a> {
    coordinator: Rc<RefCell<Coordinator<'a>>>,
    channel: usize,
    pending_batches: BinaryHeap<PendingBatch>,
    pending_points: BinaryHeap<PendingPoint>,
    active_batches: Vec<BinaryHeap<BatchPoint>>,
    buffered_points: usize,
    telemetry: SharedPostingChannelTelemetry,
}

impl SharedPostingBlockStream<'_> {
    pub fn telemetry(&self) -> SharedPostingChannelTelemetry {
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

    fn expand_next_batch(&mut self) -> Result<()> {
        let Some(batch) = self.pending_batches.pop() else {
            return Ok(());
        };
        self.telemetry.batches_requested += 1;
        let cached = self.coordinator.borrow_mut().materialize(batch.index)?;
        let mut points = BinaryHeap::from(
            cached[self.channel]
                .iter()
                .copied()
                .map(|point| {
                    assert!(
                        point.score <= batch.upper_bound,
                        "shared posting certificate violated for point {}: score {}, bound {}",
                        point.idx,
                        point.score,
                        batch.upper_bound
                    );
                    BatchPoint(point)
                })
                .collect::<Vec<_>>(),
        );
        if let Some(point) = points.pop().map(|point| point.0) {
            let batch_slot = self.active_batches.len();
            self.buffered_points += points.len() + 1;
            self.active_batches.push(points);
            self.pending_points.push(PendingPoint { point, batch_slot });
            self.telemetry.max_buffered_points =
                self.telemetry.max_buffered_points.max(self.buffered_points);
        }
        Ok(())
    }

    fn pop_next_point(&mut self) -> Option<ScoredPointOffset> {
        let pending = self.pending_points.pop()?;
        self.buffered_points -= 1;
        if let Some(point) = self.active_batches[pending.batch_slot]
            .pop()
            .map(|point| point.0)
        {
            self.pending_points.push(PendingPoint {
                point,
                batch_slot: pending.batch_slot,
            });
        }
        self.telemetry.points_emitted += 1;
        Some(pending.point)
    }
}

impl Iterator for SharedPostingBlockStream<'_> {
    type Item = ScoredPointOffset;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_point_is_fixed() {
                return self.pop_next_point();
            }
            if self.pending_batches.is_empty() {
                return self.pop_next_point();
            }
            self.expand_next_batch()
                .expect("shared posting stream index read failed");
        }
    }
}

pub fn build_shared_posting_block_streams<'a>(
    index: &'a InvertedIndexCompressedImmutableRam<f32>,
    queries: &[RemappedSparseVector],
    batch_size: usize,
    arena: &'a SearchScratchArena,
    hardware_counter: &'a HardwareCounterCell,
) -> Result<(
    Vec<SharedPostingBlockStream<'a>>,
    SharedPostingBlockHandle<'a>,
)> {
    assert!(batch_size > 0, "shared posting batch size must be positive");
    let mut raw_term_channels = BTreeMap::<DimId, Vec<(usize, DimWeight)>>::new();
    for (channel, query) in queries.iter().enumerate() {
        assert!(
            query.indices.len() == query.values.len()
                && query.is_sorted()
                && query
                    .values
                    .iter()
                    .all(|weight| weight.is_finite() && *weight >= 0.0),
            "shared posting stream requires sorted finite non-negative query weights"
        );
        for (&term, &weight) in query.indices.iter().zip(&query.values) {
            if weight != 0.0 && (term as usize) < index.len() {
                raw_term_channels
                    .entry(term)
                    .or_default()
                    .push((channel, weight));
            }
        }
    }
    let term_channels = raw_term_channels
        .into_iter()
        .map(|(term, channels)| {
            let mut grouped = BTreeMap::<OrderedFloat<DimWeight>, Vec<usize>>::new();
            for &(channel, weight) in &channels {
                grouped
                    .entry(OrderedFloat(weight))
                    .or_default()
                    .push(channel);
            }
            let weight_groups = grouped
                .into_iter()
                .map(|(weight, channels)| (weight.into_inner(), channels))
                .collect();
            (
                term,
                TermConsumers {
                    channels,
                    weight_groups,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut max_id = None;
    for &term in term_channels.keys() {
        let posting = index.get(term, arena, hardware_counter)?;
        max_id = max_id.max(posting.last_id());
    }
    let max_id = max_id.unwrap_or(0);
    let batch_count = max_id as usize / batch_size + 1;
    let mut term_batch_maxima = BTreeMap::<DimId, Vec<DimWeight>>::new();
    let mut physical_bound_evaluations = 0usize;
    for &term in term_channels.keys() {
        let mut posting = index.get(term, arena, hardware_counter)?;
        let mut maxima = Vec::with_capacity(batch_count);
        for batch_index in 0..batch_count {
            let start = batch_index * batch_size;
            let start: PointOffsetType = start.try_into().expect("Sparse batch start exceeds u32");
            let end = start
                .saturating_add(batch_size.saturating_sub(1) as PointOffsetType)
                .min(max_id);
            maxima.push(posting.max_weight_till_id(end).unwrap_or(0.0));
            physical_bound_evaluations += 1;
            posting.skip_till_id(end);
        }
        term_batch_maxima.insert(term, maxima);
    }
    let mut channel_batches = Vec::with_capacity(queries.len());
    let mut channel_telemetry = Vec::with_capacity(queries.len());
    for query in queries {
        let mut terms = Vec::new();
        for (&term, &query_weight) in query.indices.iter().zip(&query.values) {
            if query_weight == 0.0 || (term as usize) >= index.len() {
                continue;
            }
            if let Some(maxima) = term_batch_maxima.get(&term) {
                terms.push((maxima, query_weight));
            }
        }
        let mut batches = Vec::new();
        let mut telemetry = SharedPostingChannelTelemetry::default();
        for batch_index in 0..batch_count {
            let start = batch_index * batch_size;
            let start: PointOffsetType = start.try_into().expect("Sparse batch start exceeds u32");
            let mut upper_bound = 0.0;
            for (maxima, query_weight) in &terms {
                upper_bound += maxima[batch_index] * *query_weight;
                telemetry.bound_evaluations += 1;
            }
            if upper_bound > 0.0 {
                batches.push(PendingBatch {
                    index: batch_index,
                    start,
                    upper_bound,
                });
            } else {
                telemetry.zero_bound_batches += 1;
            }
        }
        channel_batches.push(BinaryHeap::from(batches));
        channel_telemetry.push(telemetry);
    }
    let coordinator = Rc::new(RefCell::new(Coordinator {
        index,
        term_channels,
        channel_count: queries.len(),
        batch_size,
        max_id,
        cache: vec![None; batch_count],
        arena,
        hardware_counter,
        telemetry: SharedPostingBlockTelemetry {
            channels: queries.len(),
            unique_terms: 0,
            logical_term_accesses: 0,
            batches: batch_count,
            physical_bound_evaluations,
            logical_bound_evaluations: channel_telemetry
                .iter()
                .map(|telemetry| telemetry.bound_evaluations)
                .sum(),
            ..Default::default()
        },
    }));
    {
        let mut state = coordinator.borrow_mut();
        state.telemetry.unique_terms = state.term_channels.len();
        state.telemetry.logical_term_accesses = state
            .term_channels
            .values()
            .map(|consumers| consumers.channels.len())
            .sum();
    }
    let streams = channel_batches
        .into_iter()
        .zip(channel_telemetry)
        .enumerate()
        .map(
            |(channel, (pending_batches, telemetry))| SharedPostingBlockStream {
                coordinator: Rc::clone(&coordinator),
                channel,
                pending_batches,
                pending_points: BinaryHeap::new(),
                active_batches: Vec::new(),
                buffered_points: 0,
                telemetry,
            },
        )
        .collect();
    Ok((streams, SharedPostingBlockHandle(coordinator)))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
    use crate::index::posting_block_stream::PostingBlockStream;

    #[test]
    fn shared_lazy_streams_match_independent_order_and_reuse_batches() {
        let mut builder = InvertedIndexBuilder::new();
        for id in 0..30_000u32 {
            builder.add(
                id,
                RemappedSparseVector {
                    indices: vec![0, 1, 2],
                    values: vec![1.0, 100.0 - id as f32 / 400.0, (id % 17) as f32 / 3.0],
                },
            );
        }
        let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
            Cow::Owned(builder.build()),
            ".",
        )
        .unwrap();
        let queries = vec![
            RemappedSparseVector {
                indices: vec![0, 1],
                values: vec![1.0, 2.0],
            },
            RemappedSparseVector {
                indices: vec![0, 1, 2],
                values: vec![0.5, 1.0, 3.0],
            },
        ];
        let hardware_counter = HardwareCounterCell::disposable();
        let arena = SearchScratchArena::new_slow();
        let (streams, handle) =
            build_shared_posting_block_streams(&index, &queries, 4_096, &arena, &hardware_counter)
                .unwrap();
        let actual: Vec<Vec<_>> = streams.into_iter().map(|stream| stream.collect()).collect();
        let expected: Vec<Vec<_>> = queries
            .into_iter()
            .map(|query| {
                PostingBlockStream::new(&index, query, 4_096, &arena, &hardware_counter)
                    .unwrap()
                    .collect()
            })
            .collect();

        assert_eq!(actual, expected);
        assert!(handle.telemetry().reused_batch_materializations > 0);
        assert!(
            handle.telemetry().physical_posting_elements_decoded
                < handle.telemetry().logical_posting_elements_decoded
        );
    }
}
