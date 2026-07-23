// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Native resumable exact score-ordered cursor over compressed Sparse postings.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use common::universal_io::Result;
use ordered_float::OrderedFloat;
use serde::Serialize;

use super::inverted_index::InvertedIndex;
use super::posting_batch::score_posting_batch;
use super::posting_list_common::PostingListIter;
use crate::SearchScratchArena;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimId, DimWeight};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub enum NativeSparsePhysicalPlan {
    #[default]
    EagerPostingBlock,
    /// Qdrant's original document-at-a-time kernel retained as a resumable
    /// suffix-certified stream.
    NativeSearchContext,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeSparsePlan {
    Auto,
    EagerPostingBlock,
    NativeSearchContext,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct PostingBlockStreamTelemetry {
    pub plan: NativeSparsePhysicalPlan,
    pub cursor_started: bool,
    pub query_terms: usize,
    pub query_posting_elements: usize,
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

/// Object-safe contract shared by all exact resumable Sparse physical plans.
pub trait ExactSparseCursor {
    fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<Option<ScoredPointOffset>, NativeSparseCursorError>;

    fn telemetry(&self) -> PostingBlockStreamTelemetry;
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

pub struct NativeCertifiedSparseCursor<'a, I: InvertedIndex> {
    postings: Vec<WeightedPosting<I::Iter<'a>>>,
    pending_batches: BinaryHeap<PendingBatch>,
    pending_points: BinaryHeap<PendingPoint>,
    active_batches: Vec<BinaryHeap<PendingBlockPoint>>,
    scores: Vec<ScoreType>,
    buffered_points: usize,
    telemetry: PostingBlockStreamTelemetry,
    terminal_error: Option<NativeSparseCursorError>,
}

/// Backward-compatible name used by existing Sparse benchmarks.
pub type PostingBlockStream<'a, I> = NativeCertifiedSparseCursor<'a, I>;

#[derive(Clone, Debug, PartialEq)]
pub enum NativeSparseCursorError {
    Cancelled,
    CertificateViolation {
        point: PointOffsetType,
        score: f32,
        upper_bound: f32,
    },
}

impl Display for NativeSparseCursorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("native Sparse cursor was cancelled"),
            Self::CertificateViolation {
                point,
                score,
                upper_bound,
            } => write!(
                formatter,
                "native Sparse certificate violated for point {point}: score {score}, bound {upper_bound}"
            ),
        }
    }
}

impl Error for NativeSparseCursorError {}

impl<'a, I: InvertedIndex> NativeCertifiedSparseCursor<'a, I> {
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
            plan: NativeSparsePhysicalPlan::EagerPostingBlock,
            cursor_started: true,
            query_terms: query.indices.len(),
            posting_lists: postings.len(),
            query_posting_elements: postings
                .iter()
                .map(|posting| posting.iterator.len_to_end())
                .sum(),
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
            scores: Vec::new(),
            buffered_points: 0,
            telemetry,
            terminal_error: None,
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

    fn expand_next_batch(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<(), NativeSparseCursorError> {
        if stopped.load(Relaxed) {
            return Err(NativeSparseCursorError::Cancelled);
        }
        let Some(batch) = self.pending_batches.pop() else {
            return Ok(());
        };
        self.telemetry.batches_expanded += 1;
        let mut postings = self.postings.clone();
        self.telemetry.posting_elements_visited += score_posting_batch(
            postings
                .iter_mut()
                .map(|posting| (&mut posting.iterator, posting.query_weight)),
            batch.start,
            batch.end,
            &mut self.scores,
        );
        let mut points = Vec::new();
        for (offset, &score) in self.scores.iter().enumerate() {
            if stopped.load(Relaxed) {
                return Err(NativeSparseCursorError::Cancelled);
            }
            if score == 0.0 {
                continue;
            }
            let id = batch.start + offset as PointOffsetType;
            if score > batch.upper_bound {
                return Err(NativeSparseCursorError::CertificateViolation {
                    point: id,
                    score,
                    upper_bound: batch.upper_bound,
                });
            }
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
        Ok(())
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

    /// Produces the next exact score-ranked point.
    ///
    /// `Ok(None)` is the only exhaustion signal. Cancellation and certificate
    /// failures are sticky, so a failed partial prefix cannot be mistaken for
    /// a complete ranking.
    pub fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<Option<ScoredPointOffset>, NativeSparseCursorError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if stopped.load(Relaxed) {
            let error = NativeSparseCursorError::Cancelled;
            self.terminal_error = Some(error.clone());
            return Err(error);
        }
        loop {
            if self.next_point_is_fixed() {
                return Ok(self.pop_next_point());
            }
            if self.pending_batches.is_empty() {
                return Ok(self.pop_next_point());
            }
            if let Err(error) = self.expand_next_batch(stopped) {
                self.pending_batches.clear();
                self.pending_points.clear();
                self.active_batches.clear();
                self.buffered_points = 0;
                self.terminal_error = Some(error.clone());
                return Err(error);
            }
        }
    }
}

impl<I: InvertedIndex> Iterator for NativeCertifiedSparseCursor<'_, I> {
    type Item = ScoredPointOffset;

    fn next(&mut self) -> Option<Self::Item> {
        let stopped = AtomicBool::new(false);
        self.next_result(&stopped)
            .expect("infallible PostingBlockStream adapter failed")
    }
}

impl<I: InvertedIndex> ExactSparseCursor for NativeCertifiedSparseCursor<'_, I> {
    fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<Option<ScoredPointOffset>, NativeSparseCursorError> {
        NativeCertifiedSparseCursor::next_result(self, stopped)
    }

    fn telemetry(&self) -> PostingBlockStreamTelemetry {
        NativeCertifiedSparseCursor::telemetry(self)
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
    use std::sync::atomic::Ordering::Relaxed;

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

    #[test]
    fn native_cursor_distinguishes_cancellation_from_exact_eof() {
        let mut builder = InvertedIndexBuilder::new();
        for id in 0..128u32 {
            builder.add(
                id,
                RemappedSparseVector {
                    indices: vec![0],
                    values: vec![1.0 + id as f32],
                },
            );
        }
        let temp = tempfile::tempdir().unwrap();
        let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
            Cow::Owned(builder.build()),
            temp.path(),
        )
        .unwrap();
        let arena = SearchScratchArena::new_slow();
        let hardware_counter = HardwareCounterCell::disposable();
        let stopped = AtomicBool::new(false);
        let mut cursor: NativeCertifiedSparseCursor<'_, _> = PostingBlockStream::new(
            &index,
            RemappedSparseVector {
                indices: vec![0],
                values: vec![1.0],
            },
            16,
            &arena,
            &hardware_counter,
        )
        .unwrap();

        assert!(cursor.next_result(&stopped).unwrap().is_some());
        stopped.store(true, Relaxed);
        assert_eq!(
            cursor.next_result(&stopped),
            Err(NativeSparseCursorError::Cancelled)
        );
        assert_eq!(
            cursor.next_result(&stopped),
            Err(NativeSparseCursorError::Cancelled)
        );

        let active = AtomicBool::new(false);
        let mut empty = PostingBlockStream::new(
            &index,
            RemappedSparseVector::default(),
            16,
            &arena,
            &hardware_counter,
        )
        .unwrap();
        assert_eq!(empty.next_result(&active).unwrap(), None);
    }
}
