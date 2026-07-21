// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact resumable Sparse stream backed by geometrically growing native Top-K queries.

use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;

use super::inverted_index::InvertedIndex;
use super::search_context::SearchContext;
use crate::SearchScratchPool;
use crate::common::sparse_vector::RemappedSparseVector;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdaptiveTopKStreamTelemetry {
    pub refills: usize,
    pub final_requested_limit: usize,
    pub results_materialized: usize,
    pub posting_elements: usize,
    pub posting_elements_visited: usize,
    pub posting_elements_skipped: usize,
    pub batches: usize,
    pub points_emitted: usize,
}

pub struct AdaptiveTopKStream<'a, I: InvertedIndex> {
    index: &'a I,
    query: RemappedSparseVector,
    initial_limit: usize,
    next_limit: usize,
    growth_factor: usize,
    block_pruning: bool,
    block_batch_size: usize,
    endpoint_trend_router: bool,
    pool: &'a SearchScratchPool,
    stopped: &'a AtomicBool,
    hardware_counter: &'a HardwareCounterCell,
    buffer: VecDeque<ScoredPointOffset>,
    emitted: usize,
    exhausted: bool,
    telemetry: AdaptiveTopKStreamTelemetry,
}

impl<'a, I: InvertedIndex> AdaptiveTopKStream<'a, I> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        index: &'a I,
        query: RemappedSparseVector,
        initial_limit: usize,
        growth_factor: usize,
        block_pruning: bool,
        block_batch_size: usize,
        endpoint_trend_router: bool,
        pool: &'a SearchScratchPool,
        stopped: &'a AtomicBool,
        hardware_counter: &'a HardwareCounterCell,
    ) -> Self {
        assert!(initial_limit > 0, "initial Top-K limit must be positive");
        assert!(
            growth_factor >= 2,
            "Top-K growth factor must be at least two"
        );
        assert!(block_batch_size > 0, "block batch size must be positive");
        Self {
            index,
            query,
            initial_limit,
            next_limit: initial_limit,
            growth_factor,
            block_pruning,
            block_batch_size,
            endpoint_trend_router,
            pool,
            stopped,
            hardware_counter,
            buffer: VecDeque::new(),
            emitted: 0,
            exhausted: false,
            telemetry: AdaptiveTopKStreamTelemetry::default(),
        }
    }

    pub fn telemetry(&self) -> AdaptiveTopKStreamTelemetry {
        self.telemetry
    }

    fn refill(&mut self) {
        if self.exhausted {
            return;
        }
        let corpus_limit = self.index.vector_count().max(self.initial_limit);
        let requested_limit = self.next_limit.min(corpus_limit);
        let mut scratch = self.pool.get();
        let mut context = SearchContext::new(
            self.query.clone(),
            requested_limit,
            self.index,
            &mut scratch,
            self.stopped,
            self.hardware_counter,
        )
        .expect("adaptive Sparse stream index read failed");
        // The exact stream freezes the same posting-accumulation semantics at
        // every refill. Max-next pruning is a separate numerical certificate
        // and stays disabled until its equal-score boundary is outward-safe.
        context.set_posting_pruning(false);
        context.set_block_pruning(self.block_pruning);
        if self.endpoint_trend_router {
            context.apply_block_max_endpoint_trend_router();
        }
        if context.block_pruning_enabled() {
            context.set_batch_size(self.block_batch_size);
        }
        let points = context.search(&|_| true);
        let progress = context.telemetry();
        self.telemetry.refills += 1;
        self.telemetry.final_requested_limit = requested_limit;
        self.telemetry.results_materialized += points.len();
        self.telemetry.posting_elements += progress.posting_elements;
        self.telemetry.posting_elements_visited += progress.posting_elements_visited;
        self.telemetry.posting_elements_skipped += progress.posting_elements_skipped;
        self.telemetry.batches += progress.batch_count;

        let returned = points.len();
        if returned <= self.emitted {
            self.exhausted = true;
            return;
        }
        self.buffer.extend(points.into_iter().skip(self.emitted));
        self.exhausted = returned < requested_limit || requested_limit >= corpus_limit;
        if !self.exhausted {
            self.next_limit = requested_limit
                .saturating_mul(self.growth_factor)
                .min(corpus_limit);
        }
    }
}

impl<I: InvertedIndex> Iterator for AdaptiveTopKStream<'_, I> {
    type Item = ScoredPointOffset;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            self.refill();
        }
        let point = self.buffer.pop_front()?;
        self.emitted += 1;
        self.telemetry.points_emitted += 1;
        Some(point)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    #[test]
    fn adaptive_stream_matches_single_exhaustive_native_order() {
        let mut builder = InvertedIndexBuilder::new();
        for id in 0..10_000u32 {
            builder.add(
                id,
                RemappedSparseVector {
                    indices: vec![0, 1 + id % 13],
                    values: vec![1.0 + (id % 19) as f32, 0.25 + (id % 7) as f32],
                },
            );
        }
        let ram = builder.build();
        let temp = tempfile::tempdir().unwrap();
        let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
            Cow::Owned(ram),
            temp.path(),
        )
        .unwrap();
        let query = RemappedSparseVector {
            indices: vec![0, 4],
            values: vec![1.0, 2.0],
        };
        let pool = SearchScratchPool::new();
        let stopped = AtomicBool::new(false);
        let hardware_counter = HardwareCounterCell::disposable();
        let mut scratch = pool.get();
        let mut exhaustive = SearchContext::new(
            query.clone(),
            index.vector_count(),
            &index,
            &mut scratch,
            &stopped,
            &hardware_counter,
        )
        .unwrap();
        let expected = exhaustive.search(&|_| true);
        drop(scratch);

        let mut stream = AdaptiveTopKStream::new(
            &index,
            query,
            7,
            2,
            false,
            4096,
            false,
            &pool,
            &stopped,
            &hardware_counter,
        );
        let actual: Vec<_> = stream.by_ref().collect();

        assert_eq!(actual, expected);
        assert!(stream.telemetry().refills > 1);
    }
}
