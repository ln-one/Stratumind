// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Type-erased exact Sparse cursor over Qdrant's concrete Segment indexes.

use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;
use sparse::SearchScratchArena;
use sparse::common::sparse_vector::SparseVector;
use sparse::index::posting_block_stream::{
    ExactSparseCursor, NativeSparseCursorError, PostingBlockMaxState, PostingBlockStreamTelemetry,
    SparseExecutionPlan,
};

use super::VectorIndexEnum;
use crate::common::operation_error::{OperationError, OperationResult};

/// One object-safe exact stream contract for all Sparse storage encodings and
/// interchangeable physical plans.
pub struct ExactSparseIndexCursor<'a> {
    inner: Box<dyn ExactSparseCursor + 'a>,
}

pub struct ExactSparseIndexState {
    inner: PostingBlockMaxState,
}

impl ExactSparseIndexState {
    pub(crate) fn open(
        index: &VectorIndexEnum,
        query: &SparseVector,
        batch_size: usize,
        arena: &SearchScratchArena,
        hardware_counter: &HardwareCounterCell,
    ) -> OperationResult<Self> {
        let inner = match index {
            VectorIndexEnum::SparseRam(index) => {
                index.exact_rank_state(query, batch_size, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedImmutableRamF32(index) => {
                index.exact_rank_state(query, batch_size, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedImmutableRamF16(index) => {
                index.exact_rank_state(query, batch_size, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedImmutableRamU8(index) => {
                index.exact_rank_state(query, batch_size, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedMmapF32(index) => {
                index.exact_rank_state(query, batch_size, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedMmapF16(index) => {
                index.exact_rank_state(query, batch_size, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedMmapU8(index) => {
                index.exact_rank_state(query, batch_size, arena, hardware_counter)?
            }
            VectorIndexEnum::Plain(_) | VectorIndexEnum::Hnsw(_) => {
                return Err(OperationError::WrongSparse);
            }
        };
        Ok(Self { inner })
    }

    pub(crate) fn next_batch(
        &mut self,
        index: &VectorIndexEnum,
        max_results: usize,
        arena: &SearchScratchArena,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        match index {
            VectorIndexEnum::SparseRam(index) => index.advance_exact_rank_state(
                &mut self.inner,
                max_results,
                arena,
                hardware_counter,
                stopped,
            ),
            VectorIndexEnum::SparseCompressedImmutableRamF32(index) => index
                .advance_exact_rank_state(
                    &mut self.inner,
                    max_results,
                    arena,
                    hardware_counter,
                    stopped,
                ),
            VectorIndexEnum::SparseCompressedImmutableRamF16(index) => index
                .advance_exact_rank_state(
                    &mut self.inner,
                    max_results,
                    arena,
                    hardware_counter,
                    stopped,
                ),
            VectorIndexEnum::SparseCompressedImmutableRamU8(index) => index
                .advance_exact_rank_state(
                    &mut self.inner,
                    max_results,
                    arena,
                    hardware_counter,
                    stopped,
                ),
            VectorIndexEnum::SparseCompressedMmapF32(index) => index.advance_exact_rank_state(
                &mut self.inner,
                max_results,
                arena,
                hardware_counter,
                stopped,
            ),
            VectorIndexEnum::SparseCompressedMmapF16(index) => index.advance_exact_rank_state(
                &mut self.inner,
                max_results,
                arena,
                hardware_counter,
                stopped,
            ),
            VectorIndexEnum::SparseCompressedMmapU8(index) => index.advance_exact_rank_state(
                &mut self.inner,
                max_results,
                arena,
                hardware_counter,
                stopped,
            ),
            VectorIndexEnum::Plain(_) | VectorIndexEnum::Hnsw(_) => {
                Err(OperationError::WrongSparse)
            }
        }
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        self.inner.telemetry()
    }
}

impl<'a> ExactSparseIndexCursor<'a> {
    pub(crate) fn open(
        index: &'a VectorIndexEnum,
        query: &SparseVector,
        batch_size: usize,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
    ) -> OperationResult<Self> {
        #[cfg(feature = "stratumind-research")]
        let plan = match std::env::var("SPECTRA_SPARSE_NATIVE_PLAN").as_deref() {
            Ok("posting-block-max") => SparseExecutionPlan::PostingBlockMax,
            Ok("") | Err(_) => SparseExecutionPlan::Auto,
            Ok(value) => {
                return Err(OperationError::validation_error(format!(
                    "unknown research Sparse native plan: {value}"
                )));
            }
        };
        #[cfg(not(feature = "stratumind-research"))]
        let plan = SparseExecutionPlan::Auto;
        let inner = match index {
            VectorIndexEnum::SparseRam(index) => {
                index.exact_cursor_with_plan(query, batch_size, plan, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedImmutableRamF32(index) => {
                index.exact_cursor_with_plan(query, batch_size, plan, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedImmutableRamF16(index) => {
                index.exact_cursor_with_plan(query, batch_size, plan, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedImmutableRamU8(index) => {
                index.exact_cursor_with_plan(query, batch_size, plan, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedMmapF32(index) => {
                index.exact_cursor_with_plan(query, batch_size, plan, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedMmapF16(index) => {
                index.exact_cursor_with_plan(query, batch_size, plan, arena, hardware_counter)?
            }
            VectorIndexEnum::SparseCompressedMmapU8(index) => {
                index.exact_cursor_with_plan(query, batch_size, plan, arena, hardware_counter)?
            }
            VectorIndexEnum::Plain(_) | VectorIndexEnum::Hnsw(_) => {
                return Err(OperationError::WrongSparse);
            }
        };
        Ok(Self { inner })
    }

    pub fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> OperationResult<Option<ScoredPointOffset>> {
        self.inner.next_result(stopped).map_err(exact_cursor_error)
    }

    pub fn next_batch(
        &mut self,
        max_results: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        self.inner
            .next_batch(max_results, stopped)
            .map_err(exact_cursor_error)
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        self.inner.telemetry()
    }
}

fn exact_cursor_error(error: NativeSparseCursorError) -> OperationError {
    match error {
        NativeSparseCursorError::Cancelled => {
            OperationError::cancelled("native Sparse Segment cursor was cancelled")
        }
        NativeSparseCursorError::ReaderFailure(_)
        | NativeSparseCursorError::CertificateViolation { .. } => {
            OperationError::inconsistent_storage(error.to_string())
        }
    }
}
