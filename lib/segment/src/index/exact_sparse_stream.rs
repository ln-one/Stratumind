// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Type-erased exact Sparse cursor over Qdrant's concrete Segment indexes.

use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;
use sparse::SearchScratchArena;
use sparse::common::sparse_vector::SparseVector;
use sparse::index::posting_block_max::{PostingBlockMaxState, PostingBlockMaxTelemetry};

use super::VectorIndexEnum;
use crate::common::operation_error::{OperationError, OperationResult};

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

    pub fn telemetry(&self) -> PostingBlockMaxTelemetry {
        self.inner.telemetry()
    }
}
