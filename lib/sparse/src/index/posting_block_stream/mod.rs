// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Native resumable exact score-ordered cursor over compressed Sparse postings.
//!
//! The module is split by responsibility so the cursor remains a small state
//! machine while physical scoring and telemetry can evolve independently.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
#[cfg(feature = "stratumind-research")]
use std::time::Instant;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use common::universal_io::Result;
use ordered_float::OrderedFloat;

use super::inverted_index::InvertedIndex;
use crate::SearchScratchArena;
use crate::common::sparse_vector::RemappedSparseVector;

mod kernel;
mod telemetry;

use kernel::{PendingBatch, PostingBlockMaxKernel};
#[cfg(feature = "stratumind-research")]
pub use telemetry::PostingBlockPhaseTelemetry;
pub use telemetry::{
    PostingBlockMaxVariant, PostingBlockStreamTelemetry, SparseExecutionPlan, SparsePhysicalPlan,
};

/// Object-safe contract shared by all exact resumable Sparse physical plans.
pub trait ExactSparseCursor {
    fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<Option<ScoredPointOffset>, NativeSparseCursorError>;

    /// Return the next certified contiguous rank prefix, up to `max_results`.
    ///
    /// The default adapter preserves compatibility for point-oriented plans.
    /// Native kernels may override it to drain an already-certified prefix
    /// without one virtual call per point.
    fn next_batch(
        &mut self,
        max_results: usize,
        stopped: &AtomicBool,
    ) -> std::result::Result<Vec<ScoredPointOffset>, NativeSparseCursorError> {
        assert!(
            max_results > 0,
            "exact Sparse result batch must be positive"
        );
        let mut points = Vec::with_capacity(max_results);
        while points.len() < max_results {
            let Some(point) = self.next_result(stopped)? else {
                break;
            };
            points.push(point);
        }
        Ok(points)
    }

    fn telemetry(&self) -> PostingBlockStreamTelemetry;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PendingPoint {
    pub(super) point: ScoredPointOffset,
    pub(super) batch_index: usize,
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

pub struct PostingBlockMaxCursor<'a, I: InvertedIndex> {
    index: &'a I,
    arena: &'a SearchScratchArena,
    hardware_counter: &'a HardwareCounterCell,
    state: PostingBlockMaxState,
}

pub struct PostingBlockMaxState {
    kernel: PostingBlockMaxKernel,
    pending_batches: BinaryHeap<PendingBatch>,
    pending_points: BinaryHeap<PendingPoint>,
    buffered_points: usize,
    telemetry: PostingBlockStreamTelemetry,
    terminal_error: Option<NativeSparseCursorError>,
    #[cfg(feature = "stratumind-research")]
    phase_telemetry: bool,
}

impl<I: InvertedIndex> std::ops::Deref for PostingBlockMaxCursor<'_, I> {
    type Target = PostingBlockMaxState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<I: InvertedIndex> std::ops::DerefMut for PostingBlockMaxCursor<'_, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

/// Legacy research name retained for source compatibility.
pub type NativeCertifiedSparseCursor<'a, I> = PostingBlockMaxCursor<'a, I>;
/// Legacy research name retained for source compatibility.
pub type PostingBlockStream<'a, I> = PostingBlockMaxCursor<'a, I>;

#[derive(Clone, Debug, PartialEq)]
pub enum NativeSparseCursorError {
    Cancelled,
    ReaderFailure(String),
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
            Self::ReaderFailure(message) => {
                write!(formatter, "native Sparse reader failed: {message}")
            }
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

impl<'a, I: InvertedIndex> PostingBlockMaxCursor<'a, I> {
    pub fn new(
        index: &'a I,
        query: RemappedSparseVector,
        batch_size: usize,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
    ) -> Result<Self> {
        Self::new_inner(
            index,
            query,
            batch_size,
            PostingBlockMaxVariant::V1,
            arena,
            hardware_counter,
        )
    }

    pub fn new_with_variant(
        index: &'a I,
        query: RemappedSparseVector,
        batch_size: usize,
        variant: PostingBlockMaxVariant,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
    ) -> Result<Self> {
        Self::new_inner(index, query, batch_size, variant, arena, hardware_counter)
    }

    fn new_inner(
        index: &'a I,
        query: RemappedSparseVector,
        batch_size: usize,
        variant: PostingBlockMaxVariant,
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
        let mut telemetry = PostingBlockStreamTelemetry {
            plan: variant.physical_plan(),
            cursor_started: true,
            query_terms: query.indices.len(),
            ..Default::default()
        };
        #[cfg(feature = "stratumind-research")]
        let phase_telemetry = phase_telemetry_enabled();
        #[cfg(not(feature = "stratumind-research"))]
        let phase_telemetry = false;
        let (kernel, pending_batches) = PostingBlockMaxKernel::new(
            index,
            &query,
            batch_size,
            variant,
            arena,
            hardware_counter,
            &mut telemetry,
            phase_telemetry,
        )?;
        Ok(Self {
            index,
            arena,
            hardware_counter,
            state: PostingBlockMaxState {
                kernel,
                pending_batches,
                pending_points: BinaryHeap::new(),
                buffered_points: 0,
                telemetry,
                terminal_error: None,
                #[cfg(feature = "stratumind-research")]
                phase_telemetry,
            },
        })
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        self.telemetry
    }

    pub fn into_rank_state(self) -> PostingBlockMaxState {
        self.state
    }

    fn next_point_is_fixed(&mut self) -> bool {
        #[cfg(feature = "stratumind-research")]
        let started = self.phase_telemetry.then(Instant::now);
        let Some(point) = self.pending_points.peek() else {
            return false;
        };
        let Some(batch) = self.pending_batches.peek() else {
            return true;
        };
        let fixed = point.point.score > batch.upper_bound
            || (point.point.score == batch.upper_bound && point.point.idx < batch.start);
        #[cfg(feature = "stratumind-research")]
        if let Some(started) = started {
            self.telemetry.phase.proof_ns = self
                .telemetry
                .phase
                .proof_ns
                .saturating_add(saturating_elapsed_ns(started.elapsed().as_nanos()));
        }
        fixed
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
        let index = self.index;
        let arena = self.arena;
        let hardware_counter = self.hardware_counter;
        let state = &mut self.state;
        if let Some((point, buffered)) = state.kernel.expand(
            index,
            arena,
            hardware_counter,
            batch,
            stopped,
            &mut state.telemetry,
        )? {
            state.buffered_points += buffered;
            state.pending_points.push(point);
        }
        state.telemetry.max_pending_points = state
            .telemetry
            .max_pending_points
            .max(state.pending_points.len());
        state.telemetry.max_buffered_points = state
            .telemetry
            .max_buffered_points
            .max(state.buffered_points);
        Ok(())
    }

    fn pop_next_point(&mut self) -> Option<ScoredPointOffset> {
        #[cfg(feature = "stratumind-research")]
        let started = self.phase_telemetry.then(Instant::now);
        let pending = self.pending_points.pop()?;
        self.buffered_points -= 1;
        if let Some(point) = self.kernel.pop_active(pending.batch_index) {
            self.pending_points.push(PendingPoint {
                point,
                batch_index: pending.batch_index,
            });
        }
        self.telemetry.points_emitted += 1;
        #[cfg(feature = "stratumind-research")]
        if let Some(started) = started {
            self.telemetry.phase.delivery_ns = self
                .telemetry
                .phase
                .delivery_ns
                .saturating_add(saturating_elapsed_ns(started.elapsed().as_nanos()));
        }
        Some(pending.point)
    }

    fn finish_batch(
        &mut self,
        points: Vec<ScoredPointOffset>,
    ) -> std::result::Result<Vec<ScoredPointOffset>, NativeSparseCursorError> {
        if !points.is_empty()
            && (!self.pending_batches.is_empty() || !self.pending_points.is_empty())
        {
            self.telemetry.pause_count += 1;
        }
        Ok(points)
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
            if self.next_point_is_fixed() || self.pending_batches.is_empty() {
                return Ok(self.pop_next_point());
            }
            if let Err(error) = self.expand_next_batch(stopped) {
                self.pending_batches.clear();
                self.pending_points.clear();
                self.kernel.clear();
                self.buffered_points = 0;
                self.terminal_error = Some(error.clone());
                return Err(error);
            }
        }
    }

    /// Advance only until useful proof exists, then drain the currently fixed
    /// contiguous prefix without expanding another posting batch merely to
    /// fill the transport batch.
    pub fn next_batch(
        &mut self,
        max_results: usize,
        stopped: &AtomicBool,
    ) -> std::result::Result<Vec<ScoredPointOffset>, NativeSparseCursorError> {
        assert!(
            max_results > 0,
            "native Sparse result batch must be positive"
        );
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if stopped.load(Relaxed) {
            let error = NativeSparseCursorError::Cancelled;
            self.terminal_error = Some(error.clone());
            return Err(error);
        }
        let mut points = Vec::with_capacity(max_results);
        loop {
            if self.next_point_is_fixed() {
                points.push(self.pop_next_point().expect("fixed point exists"));
                if points.len() == max_results {
                    return self.finish_batch(points);
                }
                continue;
            }
            if self.pending_batches.is_empty() {
                while points.len() < max_results {
                    let Some(point) = self.pop_next_point() else {
                        break;
                    };
                    points.push(point);
                }
                return self.finish_batch(points);
            }
            if !points.is_empty() {
                return self.finish_batch(points);
            }
            if let Err(error) = self.expand_next_batch(stopped) {
                self.pending_batches.clear();
                self.pending_points.clear();
                self.kernel.clear();
                self.buffered_points = 0;
                self.terminal_error = Some(error.clone());
                return Err(error);
            }
        }
    }
}

impl PostingBlockMaxState {
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

    fn pop_next_point(&mut self) -> Option<ScoredPointOffset> {
        let pending = self.pending_points.pop()?;
        self.buffered_points -= 1;
        if let Some(point) = self.kernel.pop_active(pending.batch_index) {
            self.pending_points.push(PendingPoint {
                point,
                batch_index: pending.batch_index,
            });
        }
        self.telemetry.points_emitted += 1;
        Some(pending.point)
    }

    fn finish_batch(&mut self, points: Vec<ScoredPointOffset>) -> Vec<ScoredPointOffset> {
        if !points.is_empty()
            && (!self.pending_batches.is_empty() || !self.pending_points.is_empty())
        {
            self.telemetry.pause_count += 1;
        }
        points
    }

    /// Advance reader-independent Sparse state with an index borrowed only for
    /// this batch.
    pub fn next_batch_with<'a, I: InvertedIndex>(
        &mut self,
        index: &'a I,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
        max_results: usize,
        stopped: &AtomicBool,
    ) -> std::result::Result<Vec<ScoredPointOffset>, NativeSparseCursorError> {
        assert!(
            max_results > 0,
            "native Sparse result batch must be positive"
        );
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if stopped.load(Relaxed) {
            let error = NativeSparseCursorError::Cancelled;
            self.terminal_error = Some(error.clone());
            return Err(error);
        }
        let mut points = Vec::with_capacity(max_results);
        loop {
            if self.next_point_is_fixed() {
                points.push(self.pop_next_point().expect("fixed point exists"));
                if points.len() == max_results {
                    return Ok(self.finish_batch(points));
                }
                continue;
            }
            if self.pending_batches.is_empty() {
                while points.len() < max_results {
                    let Some(point) = self.pop_next_point() else {
                        break;
                    };
                    points.push(point);
                }
                return Ok(self.finish_batch(points));
            }
            if !points.is_empty() {
                return Ok(self.finish_batch(points));
            }
            let batch = self.pending_batches.pop().expect("pending batch exists");
            let expanded = self.kernel.expand(
                index,
                arena,
                hardware_counter,
                batch,
                stopped,
                &mut self.telemetry,
            );
            match expanded {
                Ok(Some((point, buffered))) => {
                    self.buffered_points += buffered;
                    self.pending_points.push(point);
                    self.telemetry.max_pending_points = self
                        .telemetry
                        .max_pending_points
                        .max(self.pending_points.len());
                    self.telemetry.max_buffered_points =
                        self.telemetry.max_buffered_points.max(self.buffered_points);
                }
                Ok(None) => {}
                Err(error) => {
                    self.pending_batches.clear();
                    self.pending_points.clear();
                    self.kernel.clear();
                    self.buffered_points = 0;
                    self.terminal_error = Some(error.clone());
                    return Err(error);
                }
            }
        }
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        self.telemetry
    }
}

#[cfg(feature = "stratumind-research")]
fn phase_telemetry_enabled() -> bool {
    std::env::var_os("SPECTRA_PBM_PHASE_TELEMETRY").is_some_and(|value| value == "1")
}

#[cfg(feature = "stratumind-research")]
fn saturating_elapsed_ns(nanos: u128) -> u64 {
    nanos.min(u128::from(u64::MAX)) as u64
}

impl<I: InvertedIndex> Iterator for PostingBlockMaxCursor<'_, I> {
    type Item = ScoredPointOffset;

    fn next(&mut self) -> Option<Self::Item> {
        let stopped = AtomicBool::new(false);
        self.next_result(&stopped)
            .expect("infallible PostingBlockStream adapter failed")
    }
}

impl<I: InvertedIndex> ExactSparseCursor for PostingBlockMaxCursor<'_, I> {
    fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> std::result::Result<Option<ScoredPointOffset>, NativeSparseCursorError> {
        PostingBlockMaxCursor::next_result(self, stopped)
    }

    fn next_batch(
        &mut self,
        max_results: usize,
        stopped: &AtomicBool,
    ) -> std::result::Result<Vec<ScoredPointOffset>, NativeSparseCursorError> {
        PostingBlockMaxCursor::next_batch(self, max_results, stopped)
    }

    fn telemetry(&self) -> PostingBlockStreamTelemetry {
        PostingBlockMaxCursor::telemetry(self)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    #[cfg(feature = "stratumind-research")]
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering::Relaxed;

    #[cfg(feature = "stratumind-research")]
    use proptest::prelude::*;

    use super::*;
    use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    #[test]
    fn reader_independent_state_resumes_on_different_workers() {
        let mut builder = InvertedIndexBuilder::new();
        for id in 0..257u32 {
            builder.add(
                id,
                RemappedSparseVector {
                    indices: vec![0],
                    values: vec![257.0 - id as f32],
                },
            );
        }
        let temp = tempfile::tempdir().unwrap();
        let index = std::sync::Arc::new(
            InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
                Cow::Owned(builder.build()),
                temp.path(),
            )
            .unwrap(),
        );
        let query = RemappedSparseVector {
            indices: vec![0],
            values: vec![1.0],
        };
        let arena = SearchScratchArena::new_slow();
        let hardware_counter = HardwareCounterCell::disposable();
        let state = PostingBlockMaxCursor::new_with_variant(
            index.as_ref(),
            query,
            32,
            PostingBlockMaxVariant::CompressedMetadata,
            &arena,
            &hardware_counter,
        )
        .unwrap()
        .into_rank_state();

        let index_for_first = index.clone();
        let first = std::thread::spawn(move || {
            let mut state = state;
            let arena = SearchScratchArena::new_slow();
            let hardware_counter = HardwareCounterCell::disposable();
            let points = state
                .next_batch_with(
                    index_for_first.as_ref(),
                    &arena,
                    &hardware_counter,
                    7,
                    &AtomicBool::new(false),
                )
                .unwrap();
            (state, points)
        })
        .join()
        .unwrap();
        let second = std::thread::spawn(move || {
            let (mut state, mut points) = first;
            let arena = SearchScratchArena::new_slow();
            let hardware_counter = HardwareCounterCell::disposable();
            while points.len() < 257 {
                let batch = state
                    .next_batch_with(
                        index.as_ref(),
                        &arena,
                        &hardware_counter,
                        64,
                        &AtomicBool::new(false),
                    )
                    .unwrap();
                if batch.is_empty() {
                    break;
                }
                points.extend(batch);
            }
            points
        })
        .join()
        .unwrap();

        assert_eq!(second.len(), 257);
        assert!(second.windows(2).all(|pair| pair[0].score > pair[1].score));
    }

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
            PostingBlockStream::new(&index, query.clone(), 4096, &arena, &hardware_counter)
                .unwrap();
        let actual_top: Vec<_> = top_stream.by_ref().take(20).collect();
        assert_eq!(actual_top, expected[..20]);
        assert!(top_stream.telemetry().batches_expanded < top_stream.telemetry().batches);

        let stopped = AtomicBool::new(false);
        let mut batch_stream =
            PostingBlockStream::new(&index, query, 4096, &arena, &hardware_counter).unwrap();
        let mut batch_top = Vec::new();
        while batch_top.len() < 20 {
            let points = batch_stream
                .next_batch((20 - batch_top.len()).min(8), &stopped)
                .unwrap();
            assert!(!points.is_empty());
            batch_top.extend(points);
        }
        assert_eq!(batch_top, expected[..20]);
    }

    #[cfg(feature = "stratumind-research")]
    #[test]
    fn posting_block_max_variants_match_full_order_and_batch_boundaries() {
        let mut builder = InvertedIndexBuilder::new();
        for id in 0..12_345_u32 {
            builder.add(
                id,
                RemappedSparseVector {
                    indices: vec![0, 1, 2],
                    values: vec![
                        (id % 29) as f32,
                        (id % 17) as f32 / 3.0,
                        (12_345 - id) as f32 / 97.0,
                    ],
                },
            );
        }
        let temp = tempfile::tempdir().unwrap();
        let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
            Cow::Owned(builder.build()),
            temp.path(),
        )
        .unwrap();
        let query = RemappedSparseVector {
            indices: vec![0, 1, 2],
            values: vec![1.0, 0.25, 0.75],
        };
        let arena = SearchScratchArena::new_slow();
        let hardware_counter = HardwareCounterCell::disposable();
        let stopped = AtomicBool::new(false);
        let expected: Vec<_> =
            PostingBlockMaxCursor::new(&index, query.clone(), 257, &arena, &hardware_counter)
                .unwrap()
                .collect();

        for variant in [
            PostingBlockMaxVariant::CompressedMetadata,
            PostingBlockMaxVariant::RangeDirectDense,
            PostingBlockMaxVariant::RangeDirectTouched,
            PostingBlockMaxVariant::RangeDirectSorted,
        ] {
            let mut cursor = PostingBlockMaxCursor::new_with_variant(
                &index,
                query.clone(),
                257,
                variant,
                &arena,
                &hardware_counter,
            )
            .unwrap();
            let mut actual = Vec::new();
            for requested in [1, 7, 32].into_iter().cycle() {
                let batch = cursor.next_batch(requested, &stopped).unwrap();
                if batch.is_empty() {
                    break;
                }
                assert!(batch.len() <= requested);
                actual.extend(batch);
            }
            assert_eq!(actual, expected, "{variant:?}");
        }
    }

    #[cfg(feature = "stratumind-research")]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn posting_block_max_variants_match_exhaustive_randomized(
            raw in prop::collection::vec((0_u8..80, 0_u8..8, 0_u8..32), 1..400),
            query_weights in prop::collection::vec(0_u8..8, 8),
            scan_span in 1_usize..96,
            result_batch in 1_usize..33,
        ) {
            let mut by_document: BTreeMap<u32, BTreeMap<u32, f32>> = BTreeMap::new();
            for (document, term, weight) in raw {
                if weight != 0 {
                    by_document
                        .entry(u32::from(document) * 17)
                        .or_default()
                        .insert(u32::from(term), f32::from(weight) / 5.0);
                }
            }
            let query_values: Vec<_> = query_weights
                .into_iter()
                .map(|weight| f32::from(weight) / 3.0)
                .collect();
            let mut builder = InvertedIndexBuilder::new();
            let mut documents = Vec::new();
            for (id, terms) in by_document {
                let vector = RemappedSparseVector {
                    indices: terms.keys().copied().collect(),
                    values: terms.values().copied().collect(),
                };
                builder.add(id, vector.clone());
                documents.push((id, vector));
            }
            let term_count = documents
                .iter()
                .flat_map(|(_, vector)| vector.indices.iter().copied())
                .max()
                .map_or(0, |term| term as usize + 1);
            let query = RemappedSparseVector {
                indices: (0..term_count as u32).collect(),
                values: query_values[..term_count].to_vec(),
            };
            let temp = tempfile::tempdir().unwrap();
            let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
                Cow::Owned(builder.build()),
                temp.path(),
            )
            .unwrap();
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
            let arena = SearchScratchArena::new_slow();
            let hardware_counter = HardwareCounterCell::disposable();
            let stopped = AtomicBool::new(false);

            for variant in [
                PostingBlockMaxVariant::V1,
                PostingBlockMaxVariant::CompressedMetadata,
                PostingBlockMaxVariant::RangeDirectDense,
                PostingBlockMaxVariant::RangeDirectTouched,
                PostingBlockMaxVariant::RangeDirectSorted,
            ] {
                let mut cursor = PostingBlockMaxCursor::new_with_variant(
                    &index,
                    query.clone(),
                    scan_span,
                    variant,
                    &arena,
                    &hardware_counter,
                )
                .unwrap();
                let mut actual = Vec::new();
                loop {
                    let batch = cursor.next_batch(result_batch, &stopped).unwrap();
                    if batch.is_empty() {
                        break;
                    }
                    actual.extend(batch);
                }
                prop_assert_eq!(&actual, &expected, "{:?}", variant);
            }
        }
    }

    #[test]
    fn exact_cursor_distinguishes_cancellation_from_exact_eof() {
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
