use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use common::universal_io::Result;
use ordered_float::OrderedFloat;

use super::telemetry::PostingBlockMaxTelemetry;
use super::{ExactSparseStreamError, PendingPoint};
use crate::SearchScratchArena;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimId, DimWeight};
use crate::index::inverted_index::InvertedIndex;
use crate::index::posting_batch::score_posting_batch;
use crate::index::posting_list_common::PostingListIter;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PendingBatch {
    pub(super) start: PointOffsetType,
    pub(super) end: PointOffsetType,
    pub(super) upper_bound: ScoreType,
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

enum ActiveBatch {
    Heap(BinaryHeap<PendingBlockPoint>),
}

impl ActiveBatch {
    fn len(&self) -> usize {
        match self {
            Self::Heap(points) => points.len(),
        }
    }

    fn pop(&mut self) -> Option<ScoredPointOffset> {
        match self {
            Self::Heap(points) => points.pop().map(|point| point.0),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Heap(points) => points.capacity(),
        }
    }
}

#[derive(Clone)]
struct WeightedPosting<T> {
    iterator: T,
    query_weight: DimWeight,
}

pub(super) struct PostingBlockMaxKernel {
    query: RemappedSparseVector,
    active_batches: Vec<ActiveBatch>,
    scores: Vec<ScoreType>,
}

impl PostingBlockMaxKernel {
    pub(super) fn new<'a, I: InvertedIndex>(
        index: &'a I,
        query: &RemappedSparseVector,
        batch_size: usize,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
        telemetry: &mut PostingBlockMaxTelemetry,
    ) -> Result<(Self, BinaryHeap<PendingBatch>)> {
        let postings = open_postings(index, query, arena, hardware_counter)?;
        telemetry.query_posting_elements = postings
            .iter()
            .map(|posting| posting.iterator.len_to_end())
            .sum();
        telemetry.posting_lists = postings.len();

        let batches = plan_batches_native(&postings, batch_size, telemetry)
            .unwrap_or_else(|| plan_batches(&postings, batch_size, telemetry));
        let pending_batches = BinaryHeap::from(batches);
        telemetry.max_pending_batches = pending_batches.len();
        telemetry.pending_batch_capacity = pending_batches.capacity();

        Ok((
            Self {
                query: query.clone(),
                active_batches: Vec::new(),
                scores: Vec::new(),
            },
            pending_batches,
        ))
    }

    pub(super) fn expand<'a, I: InvertedIndex>(
        &mut self,
        index: &'a I,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
        batch: PendingBatch,
        stopped: &AtomicBool,
        telemetry: &mut PostingBlockMaxTelemetry,
    ) -> std::result::Result<Option<(PendingPoint, usize)>, ExactSparseStreamError> {
        if stopped.load(Relaxed) {
            return Err(ExactSparseStreamError::Cancelled);
        }
        telemetry.batches_expanded += 1;
        let batch_len = (batch.end - batch.start + 1) as usize;
        let postings = open_postings(index, &self.query, arena, hardware_counter)
            .map_err(|error| ExactSparseStreamError::ReaderFailure(error.to_string()))?;

        self.score_range(&postings, batch, batch_len, telemetry);
        let points = collect_scored_points(
            self.scores.iter().copied().enumerate(),
            batch.start,
            batch.upper_bound,
            stopped,
            telemetry,
        )?;
        let mut points = ActiveBatch::Heap(BinaryHeap::from(
            points
                .into_iter()
                .map(PendingBlockPoint)
                .collect::<Vec<_>>(),
        ));

        telemetry.score_buffer_capacity = self.scores.capacity();
        let Some(point) = points.pop() else {
            return Ok(None);
        };
        let buffered = points.len() + 1;
        let batch_index = self.active_batches.len();
        self.active_batches.push(points);
        telemetry.active_batch_capacity = self.active_batches.capacity();
        telemetry.result_buffer_capacity = telemetry
            .result_buffer_capacity
            .max(self.active_batches[batch_index].capacity());
        Ok(Some((PendingPoint { point, batch_index }, buffered)))
    }

    fn score_range<T: PostingListIter + Clone>(
        &mut self,
        postings: &[WeightedPosting<T>],
        batch: PendingBatch,
        batch_len: usize,
        telemetry: &mut PostingBlockMaxTelemetry,
    ) {
        let mut postings = postings.to_vec();
        telemetry.posting_elements_visited += score_posting_batch(
            postings
                .iter_mut()
                .map(|posting| (&mut posting.iterator, posting.query_weight)),
            batch.start,
            batch.end,
            &mut self.scores,
        );
        telemetry.score_buffer_slots += batch_len;
        telemetry.max_score_buffer_slots = telemetry.max_score_buffer_slots.max(batch_len);
    }

    pub(super) fn pop_active(&mut self, batch_index: usize) -> Option<ScoredPointOffset> {
        self.active_batches[batch_index].pop()
    }

    pub(super) fn clear(&mut self) {
        self.active_batches.clear();
        self.scores.clear();
    }
}

fn plan_batches<T: PostingListIter + Clone>(
    postings: &[WeightedPosting<T>],
    batch_size: usize,
    telemetry: &mut PostingBlockMaxTelemetry,
) -> Vec<PendingBatch> {
    let batch_width: PointOffsetType = batch_size
        .try_into()
        .expect("posting stream batch size exceeds PointOffsetType");
    let mut bound_postings = postings.to_vec();
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
    batches
}

fn plan_batches_native<T: PostingListIter + Clone>(
    postings: &[WeightedPosting<T>],
    batch_size: usize,
    telemetry: &mut PostingBlockMaxTelemetry,
) -> Option<Vec<PendingBatch>> {
    let batch_width: PointOffsetType = batch_size.try_into().ok()?;
    let min_id = postings
        .iter()
        .filter_map(|posting| posting.iterator.clone().peek())
        .map(|element| element.record_id)
        .min();
    let max_id = postings
        .iter()
        .filter_map(|posting| posting.iterator.last_id())
        .max();
    let (min_id, max_id) = match (min_id, max_id) {
        (Some(min_id), Some(max_id)) => (min_id, max_id),
        _ => return Some(Vec::new()),
    };
    let range_count = ((max_id - min_id) / batch_width + 1) as usize;
    let mut bounds = vec![0.0_f32; range_count];
    let mut term_maxima = vec![0.0_f32; range_count];
    for posting in postings {
        if !posting
            .iterator
            .fill_block_max_ranges(min_id, batch_width, &mut term_maxima)
        {
            return None;
        }
        for (bound, maximum) in bounds.iter_mut().zip(&term_maxima) {
            *bound += maximum * posting.query_weight;
        }
    }
    telemetry.batches += range_count;
    telemetry.bound_evaluations += range_count * postings.len();
    let mut batches = Vec::with_capacity(range_count);
    for (range, upper_bound) in bounds.into_iter().enumerate() {
        if upper_bound == 0.0 {
            telemetry.zero_bound_batches += 1;
            continue;
        }
        let start = min_id + range as PointOffsetType * batch_width;
        let end = start
            .saturating_add(batch_width.saturating_sub(1))
            .min(max_id);
        batches.push(PendingBatch {
            start,
            end,
            upper_bound,
        });
    }
    Some(batches)
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

fn collect_scored_points(
    scores: impl Iterator<Item = (usize, ScoreType)>,
    batch_start: PointOffsetType,
    batch_upper_bound: ScoreType,
    stopped: &AtomicBool,
    telemetry: &mut PostingBlockMaxTelemetry,
) -> std::result::Result<Vec<ScoredPointOffset>, ExactSparseStreamError> {
    let mut points = Vec::new();
    for (offset, score) in scores {
        if stopped.load(Relaxed) {
            return Err(ExactSparseStreamError::Cancelled);
        }
        if score == 0.0 {
            continue;
        }
        let id = batch_start + offset as PointOffsetType;
        if score > batch_upper_bound {
            return Err(ExactSparseStreamError::CertificateViolation {
                point: id,
                score,
                upper_bound: batch_upper_bound,
            });
        }
        telemetry.nonzero_documents_scored += 1;
        points.push(ScoredPointOffset { idx: id, score });
    }
    Ok(points)
}
