// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Type-erased exact Sparse cursor over Qdrant's concrete Segment indexes.

use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;
use sparse::SearchScratchArena;
use sparse::common::sparse_vector::SparseVector;
use sparse::index::posting_block_stream::{
    ExactSparseCursor, NativeSparseCursorError, NativeSparsePlan, PostingBlockStreamTelemetry,
};

use super::VectorIndexEnum;
use crate::common::operation_error::{OperationError, OperationResult};

/// One object-safe exact stream contract for all Sparse storage encodings and
/// interchangeable physical plans.
pub struct NativeSparseIndexCursor<'a> {
    inner: Box<dyn ExactSparseCursor + 'a>,
}

impl<'a> NativeSparseIndexCursor<'a> {
    pub(crate) fn open(
        index: &'a VectorIndexEnum,
        query: &SparseVector,
        batch_size: usize,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
    ) -> OperationResult<Self> {
        let inner = match index {
            VectorIndexEnum::SparseRam(index) => index.native_exact_cursor_with_plan(
                query,
                batch_size,
                NativeSparsePlan::Auto,
                arena,
                hardware_counter,
            )?,
            VectorIndexEnum::SparseCompressedImmutableRamF32(index) => index
                .native_exact_cursor_with_plan(
                    query,
                    batch_size,
                    NativeSparsePlan::Auto,
                    arena,
                    hardware_counter,
                )?,
            VectorIndexEnum::SparseCompressedImmutableRamF16(index) => index
                .native_exact_cursor_with_plan(
                    query,
                    batch_size,
                    NativeSparsePlan::Auto,
                    arena,
                    hardware_counter,
                )?,
            VectorIndexEnum::SparseCompressedImmutableRamU8(index) => index
                .native_exact_cursor_with_plan(
                    query,
                    batch_size,
                    NativeSparsePlan::Auto,
                    arena,
                    hardware_counter,
                )?,
            VectorIndexEnum::SparseCompressedMmapF32(index) => index
                .native_exact_cursor_with_plan(
                    query,
                    batch_size,
                    NativeSparsePlan::Auto,
                    arena,
                    hardware_counter,
                )?,
            VectorIndexEnum::SparseCompressedMmapF16(index) => index
                .native_exact_cursor_with_plan(
                    query,
                    batch_size,
                    NativeSparsePlan::Auto,
                    arena,
                    hardware_counter,
                )?,
            VectorIndexEnum::SparseCompressedMmapU8(index) => index.native_exact_cursor_with_plan(
                query,
                batch_size,
                NativeSparsePlan::Auto,
                arena,
                hardware_counter,
            )?,
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
        self.inner.next_result(stopped).map_err(native_cursor_error)
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        self.inner.telemetry()
    }
}

fn native_cursor_error(error: NativeSparseCursorError) -> OperationError {
    match error {
        NativeSparseCursorError::Cancelled => {
            OperationError::cancelled("native Sparse Segment cursor was cancelled")
        }
        NativeSparseCursorError::CertificateViolation { .. } => {
            OperationError::inconsistent_storage(error.to_string())
        }
    }
}
