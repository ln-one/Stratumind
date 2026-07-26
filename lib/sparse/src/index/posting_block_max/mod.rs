// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Reader-independent exact Sparse ranking state over Qdrant compressed postings.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

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
pub use telemetry::{
    PostingBlockMaxTelemetry, PostingBlockMaxVariant, SparseExecutionPlan, SparsePhysicalPlan,
};

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

#[derive(Clone, Debug, PartialEq)]
pub enum ExactSparseStreamError {
    Cancelled,
    ReaderFailure(String),
    CertificateViolation {
        point: PointOffsetType,
        score: f32,
        upper_bound: f32,
    },
}

impl Display for ExactSparseStreamError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("exact Sparse stream was cancelled"),
            Self::ReaderFailure(message) => write!(formatter, "Sparse reader failed: {message}"),
            Self::CertificateViolation {
                point,
                score,
                upper_bound,
            } => write!(
                formatter,
                "Sparse certificate violated for point {point}: score {score}, bound {upper_bound}"
            ),
        }
    }
}

impl Error for ExactSparseStreamError {}

/// Complete owned state for one exact Sparse ranking session.
///
/// Posting readers and scratch memory are borrowed only by [`Self::next_batch_with`].
pub struct PostingBlockMaxState {
    kernel: PostingBlockMaxKernel,
    pending_batches: BinaryHeap<PendingBatch>,
    pending_points: BinaryHeap<PendingPoint>,
    buffered_points: usize,
    telemetry: PostingBlockMaxTelemetry,
    terminal_error: Option<ExactSparseStreamError>,
}

impl PostingBlockMaxState {
    pub fn new<I: InvertedIndex>(
        index: &I,
        query: RemappedSparseVector,
        batch_size: usize,
        arena: &SearchScratchArena,
        hardware_counter: &HardwareCounterCell,
    ) -> Result<Self> {
        Self::new_with_variant(
            index,
            query,
            batch_size,
            PostingBlockMaxVariant::CompressedMetadata,
            arena,
            hardware_counter,
        )
    }

    pub fn new_with_variant<I: InvertedIndex>(
        index: &I,
        query: RemappedSparseVector,
        batch_size: usize,
        variant: PostingBlockMaxVariant,
        arena: &SearchScratchArena,
        hardware_counter: &HardwareCounterCell,
    ) -> Result<Self> {
        assert!(batch_size > 0, "Sparse posting batch must be positive");
        assert!(
            query.indices.len() == query.values.len()
                && query.is_sorted()
                && query
                    .values
                    .iter()
                    .all(|weight| weight.is_finite() && *weight >= 0.0),
            "Sparse ranking requires sorted finite non-negative Query weights"
        );
        let mut telemetry = PostingBlockMaxTelemetry {
            plan: variant.physical_plan(),
            cursor_started: true,
            query_terms: query.indices.len(),
            ..Default::default()
        };
        let (kernel, pending_batches) = PostingBlockMaxKernel::new(
            index,
            &query,
            batch_size,
            variant,
            arena,
            hardware_counter,
            &mut telemetry,
            false,
        )?;
        Ok(Self {
            kernel,
            pending_batches,
            pending_points: BinaryHeap::new(),
            buffered_points: 0,
            telemetry,
            terminal_error: None,
        })
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

    pub fn next_batch_with<I: InvertedIndex>(
        &mut self,
        index: &I,
        arena: &SearchScratchArena,
        hardware_counter: &HardwareCounterCell,
        max_results: usize,
        stopped: &AtomicBool,
    ) -> std::result::Result<Vec<ScoredPointOffset>, ExactSparseStreamError> {
        assert!(max_results > 0, "Sparse result batch must be positive");
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if stopped.load(Relaxed) {
            let error = ExactSparseStreamError::Cancelled;
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
            match self.kernel.expand(
                index,
                arena,
                hardware_counter,
                batch,
                stopped,
                &mut self.telemetry,
            ) {
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

    pub fn telemetry(&self) -> PostingBlockMaxTelemetry {
        self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use super::*;
    use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    #[test]
    fn state_resumes_across_workers_and_matches_full_order() {
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
        let index = Arc::new(
            InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
                Cow::Owned(builder.build()),
                temp.path(),
            )
            .unwrap(),
        );
        let arena = SearchScratchArena::new_slow();
        let hardware_counter = HardwareCounterCell::new();
        let state = PostingBlockMaxState::new(
            index.as_ref(),
            RemappedSparseVector {
                indices: vec![0],
                values: vec![1.0],
            },
            32,
            &arena,
            &hardware_counter,
        )
        .unwrap();

        let first_index = index.clone();
        let first = std::thread::spawn(move || {
            let arena = SearchScratchArena::new_slow();
            let hardware_counter = HardwareCounterCell::new();
            let mut state = state;
            let batch = state
                .next_batch_with(
                    first_index.as_ref(),
                    &arena,
                    &hardware_counter,
                    7,
                    &AtomicBool::new(false),
                )
                .unwrap();
            (state, batch)
        })
        .join()
        .unwrap();

        let second = std::thread::spawn(move || {
            let arena = SearchScratchArena::new_slow();
            let hardware_counter = HardwareCounterCell::new();
            let (mut state, mut points) = first;
            loop {
                let batch = state
                    .next_batch_with(
                        index.as_ref(),
                        &arena,
                        &hardware_counter,
                        31,
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
        assert_eq!(
            second.iter().map(|point| point.idx).collect::<Vec<_>>(),
            (0..257).collect::<Vec<_>>(),
        );
    }
}
