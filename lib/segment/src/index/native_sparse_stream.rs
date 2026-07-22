// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Type-erased exact Sparse cursor over Qdrant's concrete Segment indexes.

use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;
use common::universal_io::MmapFile;
use half::f16;
use sparse::SearchScratchArena;
use sparse::common::sparse_vector::SparseVector;
use sparse::common::types::QuantizedU8;
use sparse::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use sparse::index::inverted_index::inverted_index_compressed_mmap::InvertedIndexCompressedMmap;
use sparse::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use sparse::index::posting_block_stream::{
    NativeCertifiedSparseCursor, NativeSparseCursorError, PostingBlockStreamTelemetry,
};

use super::VectorIndexEnum;
use crate::common::operation_error::{OperationError, OperationResult};

/// One exact cursor type for every Sparse representation supported by a
/// persisted Qdrant Segment.
pub enum NativeSparseIndexCursor<'a> {
    Ram(NativeCertifiedSparseCursor<'a, InvertedIndexRam>),
    ImmutableRamF32(NativeCertifiedSparseCursor<'a, InvertedIndexCompressedImmutableRam<f32>>),
    ImmutableRamF16(NativeCertifiedSparseCursor<'a, InvertedIndexCompressedImmutableRam<f16>>),
    ImmutableRamU8(
        NativeCertifiedSparseCursor<'a, InvertedIndexCompressedImmutableRam<QuantizedU8>>,
    ),
    MmapF32(NativeCertifiedSparseCursor<'a, InvertedIndexCompressedMmap<f32, MmapFile>>),
    MmapF16(NativeCertifiedSparseCursor<'a, InvertedIndexCompressedMmap<f16, MmapFile>>),
    MmapU8(NativeCertifiedSparseCursor<'a, InvertedIndexCompressedMmap<QuantizedU8, MmapFile>>),
}

impl<'a> NativeSparseIndexCursor<'a> {
    pub(crate) fn open(
        index: &'a VectorIndexEnum,
        query: &SparseVector,
        batch_size: usize,
        arena: &'a SearchScratchArena,
        hardware_counter: &'a HardwareCounterCell,
    ) -> OperationResult<Self> {
        match index {
            VectorIndexEnum::SparseRam(index) => Ok(Self::Ram(index.native_exact_cursor(
                query,
                batch_size,
                arena,
                hardware_counter,
            )?)),
            VectorIndexEnum::SparseCompressedImmutableRamF32(index) => Ok(Self::ImmutableRamF32(
                index.native_exact_cursor(query, batch_size, arena, hardware_counter)?,
            )),
            VectorIndexEnum::SparseCompressedImmutableRamF16(index) => Ok(Self::ImmutableRamF16(
                index.native_exact_cursor(query, batch_size, arena, hardware_counter)?,
            )),
            VectorIndexEnum::SparseCompressedImmutableRamU8(index) => Ok(Self::ImmutableRamU8(
                index.native_exact_cursor(query, batch_size, arena, hardware_counter)?,
            )),
            VectorIndexEnum::SparseCompressedMmapF32(index) => Ok(Self::MmapF32(
                index.native_exact_cursor(query, batch_size, arena, hardware_counter)?,
            )),
            VectorIndexEnum::SparseCompressedMmapF16(index) => Ok(Self::MmapF16(
                index.native_exact_cursor(query, batch_size, arena, hardware_counter)?,
            )),
            VectorIndexEnum::SparseCompressedMmapU8(index) => Ok(Self::MmapU8(
                index.native_exact_cursor(query, batch_size, arena, hardware_counter)?,
            )),
            VectorIndexEnum::Plain(_) | VectorIndexEnum::Hnsw(_) => {
                Err(OperationError::WrongSparse)
            }
        }
    }

    pub fn next_result(
        &mut self,
        stopped: &AtomicBool,
    ) -> OperationResult<Option<ScoredPointOffset>> {
        let result = match self {
            Self::Ram(cursor) => cursor.next_result(stopped),
            Self::ImmutableRamF32(cursor) => cursor.next_result(stopped),
            Self::ImmutableRamF16(cursor) => cursor.next_result(stopped),
            Self::ImmutableRamU8(cursor) => cursor.next_result(stopped),
            Self::MmapF32(cursor) => cursor.next_result(stopped),
            Self::MmapF16(cursor) => cursor.next_result(stopped),
            Self::MmapU8(cursor) => cursor.next_result(stopped),
        };
        result.map_err(native_cursor_error)
    }

    pub fn telemetry(&self) -> PostingBlockStreamTelemetry {
        match self {
            Self::Ram(cursor) => cursor.telemetry(),
            Self::ImmutableRamF32(cursor) => cursor.telemetry(),
            Self::ImmutableRamF16(cursor) => cursor.telemetry(),
            Self::ImmutableRamU8(cursor) => cursor.telemetry(),
            Self::MmapF32(cursor) => cursor.telemetry(),
            Self::MmapF16(cursor) => cursor.telemetry(),
            Self::MmapU8(cursor) => cursor.telemetry(),
        }
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
