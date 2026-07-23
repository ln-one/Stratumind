use std::cmp::{Ordering, max, min};
use std::collections::BinaryHeap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use common::universal_io::Result;

use super::inverted_index::InvertedIndex;
use super::posting_block_stream::{
    ExactSparseCursor, NativeSparseCursorError, NativeSparsePhysicalPlan,
    PostingBlockStreamTelemetry,
};
use super::posting_list_common::PostingListIter;
use crate::SearchScratchArena;
use crate::common::sparse_vector::RemappedSparseVector;

struct NativePostingList<T: PostingListIter> {
    iterator: T,
    query_weight: f32,
}

/// Native, resumable adapter over Qdrant's document-at-a-time Sparse kernel.
///
/// Posting iterators advance exactly once. A point is published only after
/// suffix metadata proves it strictly outranks every unscored document.
pub struct NativeSearchContextRankStream<'a, T: PostingListIter> {
    postings: Vec<NativePostingList<T>>,
    min_record_id: Option<PointOffsetType>,
    max_record_id: PointOffsetType,
    scores: Vec<ScoreType>,
    pending_points: BinaryHeap<PendingRankPoint>,
    certified_prefix: Vec<ScoredPointOffset>,
    delivered_prefix: usize,
    remaining_upper_bound: ScoreType,
    batch_size: PointOffsetType,
    hardware_counter: &'a HardwareCounterCell,
    telemetry: PostingBlockStreamTelemetry,
    terminal_error: Option<NativeSparseCursorError>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseCertifiedRankInterval {
    pub point: PointOffsetType,
    pub min_rank: usize,
    pub max_rank: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseIncrementalRankCertificate {
    pub exact_prefix: Vec<PointOffsetType>,
    pub candidates: Vec<SparseCertifiedRankInterval>,
    pub unnamed_rank_floor: usize,
    pub exhausted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingRankPoint(ScoredPointOffset);

impl Eq for PendingRankPoint {}

impl Ord for PendingRankPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .score
            .total_cmp(&other.0.score)
            .then_with(|| other.0.idx.cmp(&self.0.idx))
    }
}

impl PartialOrd for PendingRankPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T: PostingListIter> NativeSearchContextRankStream<'a, T> {
    pub fn new(
        query: RemappedSparseVector,
        batch_size: usize,
        inverted_index: &'a impl InvertedIndex<Iter<'a> = T>,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
    ) -> Result<Self> {
        assert!(
            batch_size > 0,
            "native Sparse rank-stream batch size must be positive"
        );
        assert!(
            query.indices.len() == query.values.len()
                && query.is_sorted()
                && query
                    .values
                    .iter()
                    .all(|weight| weight.is_finite() && *weight >= 0.0),
            "native Sparse rank stream requires sorted finite non-negative query weights"
        );

        let mut postings = Vec::with_capacity(query.indices.len());
        let mut min_record_id = None;
        let mut max_record_id = 0;
        let mut posting_elements = 0;
        for (&query_index, &query_weight) in query.indices.iter().zip(&query.values) {
            let mut iterator = inverted_index.get(query_index, arena, hardware_counter)?;
            let Some(first) = iterator.peek() else {
                continue;
            };
            min_record_id = Some(
                min_record_id.map_or(first.record_id, |current| min(current, first.record_id)),
            );
            max_record_id = max(max_record_id, iterator.last_id().unwrap_or(first.record_id));
            posting_elements += iterator.len_to_end();
            postings.push(NativePostingList {
                iterator,
                query_weight,
            });
        }

        let mut stream = Self {
            postings,
            min_record_id,
            max_record_id,
            scores: Vec::new(),
            pending_points: BinaryHeap::new(),
            certified_prefix: Vec::new(),
            delivered_prefix: 0,
            remaining_upper_bound: ScoreType::INFINITY,
            batch_size: batch_size
                .try_into()
                .expect("native Sparse rank-stream batch size exceeds PointOffsetType"),
            hardware_counter,
            telemetry: PostingBlockStreamTelemetry {
                plan: NativeSparsePhysicalPlan::NativeSearchContext,
                cursor_started: true,
                query_terms: query.indices.len(),
                query_posting_elements: posting_elements,
                posting_lists: query.indices.len(),
                ..Default::default()
            },
            terminal_error: None,
        };
        stream.remaining_upper_bound = stream.compute_remaining_upper_bound();
        Ok(stream)
    }

    /// Outward-rounded upper bound for every document not scored yet.
    fn compute_remaining_upper_bound(&mut self) -> ScoreType {
        if self.postings.is_empty() {
            return ScoreType::NEG_INFINITY;
        }
        if !T::reliable_max_next_weight() {
            return ScoreType::INFINITY;
        }

        let mut upper_bound = 0.0_f32;
        for posting in &mut self.postings {
            self.telemetry.bound_evaluations += 1;
            let Some(element) = posting.iterator.peek() else {
                continue;
            };
            let max_weight = element.weight.max(element.max_next_weight);
            let contribution = max_weight * posting.query_weight;
            if contribution.is_nan() || contribution == ScoreType::INFINITY {
                return ScoreType::INFINITY;
            }
            upper_bound += contribution.max(0.0);
            if !upper_bound.is_finite() {
                return ScoreType::INFINITY;
            }
        }
        next_up_nonnegative(upper_bound)
    }

    fn next_point_is_certified(&self) -> bool {
        self.pending_points
            .peek()
            .is_some_and(|point| point.0.score > self.remaining_upper_bound)
    }

    fn promote_certified_prefix(&mut self) {
        while (self.next_point_is_certified() || self.min_record_id.is_none())
            && !self.pending_points.is_empty()
        {
            self.certified_prefix
                .push(self.pending_points.pop().expect("pending point exists").0);
        }
    }

    fn advance_one_batch(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<(), NativeSparseCursorError> {
        if stopped.load(Relaxed) {
            return Err(NativeSparseCursorError::Cancelled);
        }
        let Some(batch_start_id) = self.min_record_id else {
            return Ok(());
        };
        let batch_last_id = batch_start_id
            .saturating_add(self.batch_size.saturating_sub(1))
            .min(self.max_record_id);
        let batch_len = batch_last_id - batch_start_id + 1;
        let certificate = self.remaining_upper_bound;

        self.scores.clear();
        self.scores.resize(batch_len as usize, 0.0);
        self.telemetry.batches += 1;
        self.telemetry.batches_expanded += 1;
        for posting in &mut self.postings {
            let elements_before = posting.iterator.len_to_end();
            posting.iterator.for_each_till_id(
                batch_last_id,
                self.scores.as_mut_slice(),
                #[inline(always)]
                |scores, id, weight| {
                    let local_id = (id - batch_start_id) as usize;
                    *unsafe { scores.get_unchecked_mut(local_id) } += weight * posting.query_weight;
                },
            );
            let visited = elements_before - posting.iterator.len_to_end();
            self.telemetry.posting_elements_visited += visited;
            self.hardware_counter
                .cpu_counter()
                .incr_delta(visited * posting.iterator.element_size());
        }

        for (offset, &score) in self.scores.iter().enumerate() {
            if stopped.load(Relaxed) {
                return Err(NativeSparseCursorError::Cancelled);
            }
            if score == 0.0 {
                continue;
            }
            let idx = batch_start_id + offset as PointOffsetType;
            if !score.is_finite() || score > certificate {
                return Err(NativeSparseCursorError::CertificateViolation {
                    point: idx,
                    score,
                    upper_bound: certificate,
                });
            }
            self.pending_points
                .push(PendingRankPoint(ScoredPointOffset { idx, score }));
            self.telemetry.nonzero_documents_scored += 1;
        }
        self.telemetry.max_pending_points = self
            .telemetry
            .max_pending_points
            .max(self.pending_points.len());
        self.telemetry.max_buffered_points = self
            .telemetry
            .max_buffered_points
            .max(self.pending_points.len());

        self.postings
            .retain(|posting| posting.iterator.len_to_end() != 0);
        self.min_record_id = next_min_id(&mut self.postings);
        self.remaining_upper_bound = self.compute_remaining_upper_bound();
        self.promote_certified_prefix();
        Ok(())
    }

    /// Advance one physical Qdrant DAAT batch and publish a tighter certificate.
    pub fn advance_certificate(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<bool, NativeSparseCursorError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if stopped.load(Relaxed) {
            let error = NativeSparseCursorError::Cancelled;
            self.terminal_error = Some(error.clone());
            return Err(error);
        }
        self.promote_certified_prefix();
        if self.min_record_id.is_none() {
            return Ok(false);
        }
        if let Err(error) = self.advance_one_batch(stopped) {
            self.pending_points.clear();
            self.terminal_error = Some(error.clone());
            return Err(error);
        }
        Ok(true)
    }

    pub fn rank_certificate(&self) -> SparseIncrementalRankCertificate {
        let exact_prefix: Vec<_> = self
            .certified_prefix
            .iter()
            .map(|point| point.idx)
            .collect();
        let mut pending: Vec<_> = self.pending_points.iter().map(|point| point.0).collect();
        pending.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.idx.cmp(&right.idx))
        });
        let remaining_identity_upper_bound: usize = self
            .postings
            .iter()
            .map(|posting| posting.iterator.len_to_end())
            .sum();
        let candidates = pending
            .into_iter()
            .enumerate()
            .map(|(offset, point)| {
                let min_rank = exact_prefix.len().saturating_add(offset);
                SparseCertifiedRankInterval {
                    point: point.idx,
                    min_rank,
                    max_rank: min_rank.saturating_add(remaining_identity_upper_bound),
                }
            })
            .collect();
        let exhausted = self.min_record_id.is_none();
        SparseIncrementalRankCertificate {
            unnamed_rank_floor: if exhausted {
                usize::MAX
            } else {
                exact_prefix.len()
            },
            exact_prefix,
            candidates,
            exhausted,
        }
    }

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
            self.promote_certified_prefix();
            if let Some(point) = self.certified_prefix.get(self.delivered_prefix).copied() {
                self.delivered_prefix += 1;
                self.telemetry.points_emitted += 1;
                return Ok(Some(point));
            }
            if self.min_record_id.is_none() {
                return Ok(None);
            }
            if let Err(error) = self.advance_one_batch(stopped) {
                self.pending_points.clear();
                self.terminal_error = Some(error.clone());
                return Err(error);
            }
        }
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        self.telemetry
    }
}

impl<T: PostingListIter> ExactSparseCursor for NativeSearchContextRankStream<'_, T> {
    fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<Option<ScoredPointOffset>, NativeSparseCursorError> {
        NativeSearchContextRankStream::next_result(self, stopped)
    }

    fn telemetry(&self) -> PostingBlockStreamTelemetry {
        NativeSearchContextRankStream::telemetry(self)
    }
}

fn next_min_id<T: PostingListIter>(
    postings: &mut [NativePostingList<T>],
) -> Option<PointOffsetType> {
    postings
        .iter_mut()
        .filter_map(|posting| posting.iterator.peek().map(|element| element.record_id))
        .min()
}

fn next_up_nonnegative(value: f32) -> f32 {
    debug_assert!(value >= 0.0 && value.is_finite());
    f32::from_bits(value.to_bits() + 1)
}
