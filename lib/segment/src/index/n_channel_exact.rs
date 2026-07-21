// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact N-channel fusion over shared Dense and Sparse indexes.

use std::borrow::Cow;
use std::cell::Cell;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use common::counter::hardware_counter::HardwareCounterCell;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::adaptive_top_k_stream::{AdaptiveTopKStream, AdaptiveTopKStreamTelemetry};
use sparse::index::block_max::{BlockMaxError, BlockMaxIndex, BlockMaxTelemetry, SparseDocument};
use sparse::index::inverted_index::InvertedIndex;
use sparse::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::posting_block_stream::{PostingBlockStream, PostingBlockStreamTelemetry};
use sparse::index::shared_multi_query::{
    SharedSparseTelemetry, estimate_shared_sparse_cost, estimate_shared_sparse_cost_pair,
    evaluate_shared_sparse,
};
use sparse::index::shared_posting_block_stream::{
    SharedPostingBlockHandle, SharedPostingBlockStream, SharedPostingBlockTelemetry,
    SharedPostingChannelTelemetry, build_shared_posting_block_streams,
};
use sparse::{SearchScratchArena, SearchScratchPool};

use super::dense_ball::DenseDocument;
use super::dense_quantized::{DenseQuantizedError, DenseQuantizedIndex, DenseQuantizedTelemetry};
use super::n_channel_rank_sharing::{
    dense_representatives, share_identical_rank_streams, unique_sparse_queries,
};
use super::n_channel_streams::{
    AdaptiveSparseIdentityStream, CancellableIdentityStream, DenseIdentityStream,
    PostingSparseIdentityStream, SharedPostingIdentityStream, SharedSparseIdentityStream,
    SparseIdentityStream,
};
use crate::common::operation_error::OperationError;
use crate::common::reciprocal_rank_fusion::{
    DynamicRrfAdvance, DynamicRrfExecution, DynamicRrfPolicy, DynamicRrfSession, ExactRrfStream,
};
use crate::types::ExtendedPointId;

#[derive(Debug)]
pub enum NChannelExactError {
    IdentityUniverseMismatch,
    EmptyChannels,
    Cancelled,
    Sparse(BlockMaxError),
    SparsePosting(String),
    Dense(DenseQuantizedError),
    Fusion(OperationError),
}

impl Display for NChannelExactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdentityUniverseMismatch => formatter.write_str(
                "N-channel Dense and Sparse indexes must share one point identity universe",
            ),
            Self::EmptyChannels => formatter.write_str("N-channel search requires a channel"),
            Self::Cancelled => formatter.write_str("N-channel exact search was cancelled"),
            Self::Sparse(error) => write!(formatter, "N-channel Sparse index failed: {error}"),
            Self::SparsePosting(error) => {
                write!(
                    formatter,
                    "N-channel compressed Sparse index failed: {error}"
                )
            }
            Self::Dense(error) => write!(formatter, "N-channel Dense index failed: {error}"),
            Self::Fusion(error) => write!(formatter, "N-channel fusion failed: {error}"),
        }
    }
}

impl Error for NChannelExactError {}

impl From<BlockMaxError> for NChannelExactError {
    fn from(error: BlockMaxError) -> Self {
        Self::Sparse(error)
    }
}

impl From<DenseQuantizedError> for NChannelExactError {
    fn from(error: DenseQuantizedError) -> Self {
        Self::Dense(error)
    }
}

impl From<OperationError> for NChannelExactError {
    fn from(error: OperationError) -> Self {
        Self::Fusion(error)
    }
}

#[derive(Clone)]
pub enum ExactChannelQuery<'a> {
    Dense(&'a [f32]),
    Sparse(RemappedSparseVector),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseStreamStrategy {
    DocumentBlock,
    PostingBlock {
        batch_size: usize,
    },
    AdaptiveNative {
        initial_limit: usize,
        growth_factor: usize,
        batch_size: usize,
        endpoint_trend_router: bool,
    },
    SharedFull,
    AutoShared {
        batch_size: usize,
        max_unique_to_logical_milli: u16,
    },
    ProbeThenShared {
        batch_size: usize,
        probe_depth: usize,
    },
    SharedLazy {
        batch_size: usize,
    },
    AutoSharedLazy {
        batch_size: usize,
    },
}

impl Default for SparseStreamStrategy {
    fn default() -> Self {
        Self::AdaptiveNative {
            initial_limit: 4096,
            growth_factor: 2,
            batch_size: 4096,
            endpoint_trend_router: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseStreamStrategy {
    Independent,
    SharedDocumentMajor,
    ReplayIdentical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactChannelTelemetry {
    Dense(DenseQuantizedTelemetry),
    Sparse(BlockMaxTelemetry),
    SparsePosting(PostingBlockStreamTelemetry),
    SparseAdaptive(AdaptiveTopKStreamTelemetry),
    SparseShared(SharedSparseChannelTelemetry),
    SparseSharedPosting(SharedPostingChannelTelemetry),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedSparseChannelTelemetry {
    pub points_emitted: usize,
}

#[derive(Debug)]
pub struct NChannelExactSearchResult {
    pub execution: DynamicRrfExecution,
    pub channels: Vec<ExactChannelTelemetry>,
    pub physical: NChannelPhysicalTelemetry,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NChannelPhysicalTelemetry {
    pub logical_rank_streams: usize,
    pub physical_rank_streams: usize,
    pub shared_rank_stream_groups: usize,
    pub logical_rank_pulls: usize,
    pub physical_rank_pulls: usize,
    pub rank_replay_buffered_identities: usize,
    pub dense_channels: usize,
    pub dense_physical_channels: usize,
    pub dense_document_major_passes: usize,
    pub dense_document_rows_traversed: usize,
    pub dense_logical_dot_products: usize,
    pub dense_physical_dot_products: usize,
    pub dense_preparation_ns: u128,
    pub sparse_shared: SharedSparseTelemetry,
    pub sparse_shared_preparation_ns: u128,
    pub sparse_auto_requested: bool,
    pub sparse_auto_selected_shared: bool,
    pub sparse_estimated_unique_posting_elements: usize,
    pub sparse_estimated_logical_posting_elements: usize,
    pub sparse_estimated_deduplicated_posting_elements: usize,
    pub sparse_probe_then_shared_requested: bool,
    pub sparse_probe_selected_shared: bool,
    pub sparse_probe_depth: usize,
    pub sparse_prefix_identities_replayed: usize,
    pub sparse_shared_lazy: SharedPostingBlockTelemetry,
}

#[derive(Clone, Debug)]
pub struct NChannelExactIndex {
    dense: DenseQuantizedIndex,
    sparse: BlockMaxIndex,
    sparse_postings: InvertedIndexCompressedImmutableRam<f32>,
    point_capacity: usize,
}

impl NChannelExactIndex {
    pub fn build(
        dense_documents: Vec<DenseDocument>,
        sparse_documents: Vec<SparseDocument>,
        sparse_block_size: usize,
    ) -> Result<Self, NChannelExactError> {
        let mut dense_ids: Vec<_> = dense_documents.iter().map(|document| document.id).collect();
        let mut sparse_ids: Vec<_> = sparse_documents
            .iter()
            .map(|document| document.id)
            .collect();
        dense_ids.sort_unstable();
        sparse_ids.sort_unstable();
        if dense_ids != sparse_ids {
            return Err(NChannelExactError::IdentityUniverseMismatch);
        }
        let point_capacity = dense_ids.last().map_or(0, |id| *id as usize + 1);
        let sparse_ram = InvertedIndexBuilder::build_from_iterator(
            sparse_documents
                .iter()
                .map(|document| (document.id, document.vector.clone())),
        );
        let sparse_postings =
            InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(Cow::Owned(sparse_ram), ".")
                .map_err(|error| NChannelExactError::SparsePosting(error.to_string()))?;
        Ok(Self {
            dense: DenseQuantizedIndex::build(dense_documents)?,
            sparse: BlockMaxIndex::build(sparse_documents, sparse_block_size)?,
            sparse_postings,
            point_capacity,
        })
    }

    pub fn document_count(&self) -> usize {
        self.dense.document_count()
    }

    pub fn dense_index(&self) -> &DenseQuantizedIndex {
        &self.dense
    }

    pub fn sparse_index(&self) -> &BlockMaxIndex {
        &self.sparse
    }

    pub fn sparse_posting_index(&self) -> &InvertedIndexCompressedImmutableRam<f32> {
        &self.sparse_postings
    }

    pub fn search<'a>(
        &'a self,
        queries: Vec<ExactChannelQuery<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
    ) -> Result<NChannelExactSearchResult, NChannelExactError> {
        self.search_with_policy(queries, top_k, rrf_k, weights, DynamicRrfPolicy::default())
    }

    pub fn search_with_policy<'a>(
        &'a self,
        queries: Vec<ExactChannelQuery<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        policy: DynamicRrfPolicy,
    ) -> Result<NChannelExactSearchResult, NChannelExactError> {
        self.search_with_sparse_strategy(
            queries,
            top_k,
            rrf_k,
            weights,
            policy,
            SparseStreamStrategy::default(),
        )
    }

    pub fn search_with_sparse_strategy<'a>(
        &'a self,
        queries: Vec<ExactChannelQuery<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        policy: DynamicRrfPolicy,
        sparse_strategy: SparseStreamStrategy,
    ) -> Result<NChannelExactSearchResult, NChannelExactError> {
        self.search_with_strategies(
            queries,
            top_k,
            rrf_k,
            weights,
            policy,
            DenseStreamStrategy::ReplayIdentical,
            sparse_strategy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_with_sparse_strategy_and_cancellation<'a>(
        &'a self,
        queries: Vec<ExactChannelQuery<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        policy: DynamicRrfPolicy,
        sparse_strategy: SparseStreamStrategy,
        stopped: &'a AtomicBool,
    ) -> Result<NChannelExactSearchResult, NChannelExactError> {
        self.search_with_strategies_and_cancellation(
            queries,
            top_k,
            rrf_k,
            weights,
            policy,
            DenseStreamStrategy::ReplayIdentical,
            sparse_strategy,
            stopped,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_with_strategies<'a>(
        &'a self,
        queries: Vec<ExactChannelQuery<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        policy: DynamicRrfPolicy,
        dense_strategy: DenseStreamStrategy,
        sparse_strategy: SparseStreamStrategy,
    ) -> Result<NChannelExactSearchResult, NChannelExactError> {
        let stopped = AtomicBool::new(false);
        self.search_with_strategies_and_cancellation(
            queries,
            top_k,
            rrf_k,
            weights,
            policy,
            dense_strategy,
            sparse_strategy,
            &stopped,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_with_strategies_and_cancellation<'a>(
        &'a self,
        queries: Vec<ExactChannelQuery<'a>>,
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
        policy: DynamicRrfPolicy,
        dense_strategy: DenseStreamStrategy,
        sparse_strategy: SparseStreamStrategy,
        stopped: &'a AtomicBool,
    ) -> Result<NChannelExactSearchResult, NChannelExactError> {
        if stopped.load(Relaxed) {
            return Err(NChannelExactError::Cancelled);
        }
        if queries.is_empty() {
            return Err(NChannelExactError::EmptyChannels);
        }
        let sparse_queries: Vec<_> = queries
            .iter()
            .filter_map(|query| match query {
                ExactChannelQuery::Dense(_) => None,
                ExactChannelQuery::Sparse(query) => Some(query.clone()),
            })
            .collect();
        let (
            sparse_strategy,
            sparse_auto_requested,
            sparse_auto_selected_shared,
            sparse_probe_then_shared_requested,
            sparse_estimate,
            sparse_estimated_deduplicated_posting_elements,
        ) = match sparse_strategy {
            SparseStreamStrategy::AutoShared {
                batch_size,
                max_unique_to_logical_milli,
            } => {
                let estimate = estimate_shared_sparse_cost(
                    &self.sparse_postings,
                    &sparse_queries,
                    &HardwareCounterCell::disposable(),
                )
                .map_err(NChannelExactError::SparsePosting)?;
                let selected_shared = estimate.logical_posting_elements > 0
                    && estimate.unique_posting_elements.saturating_mul(1_000)
                        <= estimate
                            .logical_posting_elements
                            .saturating_mul(max_unique_to_logical_milli as usize);
                (
                    if selected_shared {
                        SparseStreamStrategy::SharedFull
                    } else {
                        SparseStreamStrategy::PostingBlock { batch_size }
                    },
                    true,
                    selected_shared,
                    false,
                    estimate,
                    0,
                )
            }
            SparseStreamStrategy::AutoSharedLazy { batch_size } => {
                let deduplicated_queries = unique_sparse_queries(&queries, weights);
                let (estimate, deduplicated_estimate) = estimate_shared_sparse_cost_pair(
                    &self.sparse_postings,
                    &sparse_queries,
                    &deduplicated_queries,
                    &HardwareCounterCell::disposable(),
                )
                .map_err(NChannelExactError::SparsePosting)?;
                let selected_shared = estimate.unique_posting_elements
                    < deduplicated_estimate.logical_posting_elements;
                (
                    if selected_shared {
                        SparseStreamStrategy::SharedLazy { batch_size }
                    } else {
                        SparseStreamStrategy::PostingBlock { batch_size }
                    },
                    true,
                    selected_shared,
                    false,
                    estimate,
                    deduplicated_estimate.logical_posting_elements,
                )
            }
            strategy @ SparseStreamStrategy::ProbeThenShared { probe_depth, .. } => {
                if probe_depth == 0 {
                    return Err(OperationError::validation_error(
                        "Probe-then-Shared depth must be positive",
                    )
                    .into());
                }
                let estimate = estimate_shared_sparse_cost(
                    &self.sparse_postings,
                    &sparse_queries,
                    &HardwareCounterCell::disposable(),
                )
                .map_err(NChannelExactError::SparsePosting)?;
                (strategy, false, false, true, estimate, 0)
            }
            strategy @ (SparseStreamStrategy::DocumentBlock
            | SparseStreamStrategy::PostingBlock { .. }
            | SparseStreamStrategy::AdaptiveNative { .. }
            | SparseStreamStrategy::SharedFull
            | SparseStreamStrategy::SharedLazy { .. }) => (
                strategy,
                false,
                strategy == SparseStreamStrategy::SharedFull,
                false,
                Default::default(),
                0,
            ),
        };
        if stopped.load(Relaxed) {
            return Err(NChannelExactError::Cancelled);
        }
        let replay_identical_dense = dense_strategy == DenseStreamStrategy::ReplayIdentical;
        let dense_representative_flags = if replay_identical_dense {
            dense_representatives(&queries, weights)
        } else {
            vec![true; queries.len()]
        };
        let dense_channels = queries
            .iter()
            .filter(|query| matches!(query, ExactChannelQuery::Dense(_)))
            .count();
        let dense_queries: Vec<&'a [f32]> = queries
            .iter()
            .zip(&dense_representative_flags)
            .filter_map(|(query, representative)| match (query, representative) {
                (ExactChannelQuery::Dense(query), true) => Some(*query),
                _ => None,
            })
            .collect();
        let dense_physical_channels = dense_queries.len();
        let dense_preparation_started = Instant::now();
        let mut dense_streams = match dense_strategy {
            DenseStreamStrategy::Independent | DenseStreamStrategy::ReplayIdentical => {
                dense_queries
                    .into_iter()
                    .map(|query| self.dense.stream(query))
                    .collect::<Result<Vec<_>, _>>()?
            }
            DenseStreamStrategy::SharedDocumentMajor => {
                self.dense.streams_document_major(&dense_queries)?
            }
        }
        .into_iter();
        let dense_preparation_ns = dense_preparation_started.elapsed().as_nanos();
        if stopped.load(Relaxed) {
            return Err(NChannelExactError::Cancelled);
        }
        let dense_document_major_passes = match (dense_physical_channels, dense_strategy) {
            (0, _) => 0,
            (_, DenseStreamStrategy::Independent | DenseStreamStrategy::ReplayIdentical) => {
                dense_physical_channels
            }
            (_, DenseStreamStrategy::SharedDocumentMajor) => 1,
        };
        let physical = NChannelPhysicalTelemetry {
            dense_channels,
            dense_physical_channels,
            dense_document_major_passes,
            dense_document_rows_traversed: self.document_count() * dense_document_major_passes,
            dense_logical_dot_products: self.document_count() * dense_channels,
            dense_physical_dot_products: self.document_count() * dense_physical_channels,
            dense_preparation_ns,
            ..Default::default()
        };
        let telemetry: Vec<_> = queries
            .iter()
            .map(|query| {
                Rc::new(Cell::new(match query {
                    ExactChannelQuery::Dense(_) => {
                        ExactChannelTelemetry::Dense(DenseQuantizedTelemetry::default())
                    }
                    ExactChannelQuery::Sparse(_) => match sparse_strategy {
                        SparseStreamStrategy::DocumentBlock => {
                            ExactChannelTelemetry::Sparse(BlockMaxTelemetry::default())
                        }
                        SparseStreamStrategy::PostingBlock { .. } => {
                            ExactChannelTelemetry::SparsePosting(
                                PostingBlockStreamTelemetry::default(),
                            )
                        }
                        SparseStreamStrategy::ProbeThenShared { .. } => {
                            ExactChannelTelemetry::SparsePosting(
                                PostingBlockStreamTelemetry::default(),
                            )
                        }
                        SparseStreamStrategy::SharedLazy { .. } => {
                            ExactChannelTelemetry::SparseSharedPosting(
                                SharedPostingChannelTelemetry::default(),
                            )
                        }
                        SparseStreamStrategy::AdaptiveNative { .. } => {
                            ExactChannelTelemetry::SparseAdaptive(
                                AdaptiveTopKStreamTelemetry::default(),
                            )
                        }
                        SparseStreamStrategy::SharedFull => ExactChannelTelemetry::SparseShared(
                            SharedSparseChannelTelemetry::default(),
                        ),
                        SparseStreamStrategy::AutoShared { .. } => {
                            unreachable!("Auto Shared is resolved before stream construction")
                        }
                        SparseStreamStrategy::AutoSharedLazy { .. } => {
                            unreachable!("Auto Shared Lazy is resolved before stream construction")
                        }
                    },
                }))
            })
            .collect();
        let posting_arenas: Vec<_> = (0..queries.len())
            .map(|_| SearchScratchArena::new_slow())
            .collect();
        let pool = SearchScratchPool::new();
        let hardware_counter = HardwareCounterCell::disposable();
        let (mut shared_sparse_streams, sparse_shared, sparse_shared_preparation_ns) =
            match sparse_strategy {
                SparseStreamStrategy::SharedFull => {
                    let sparse_shared_started = Instant::now();
                    let prepared = evaluate_shared_sparse(
                        &self.sparse_postings,
                        &sparse_queries,
                        self.point_capacity,
                        &posting_arenas[0],
                        &hardware_counter,
                    )
                    .map_err(NChannelExactError::SparsePosting)?;
                    if stopped.load(Relaxed) {
                        return Err(NChannelExactError::Cancelled);
                    }
                    (
                        Some(prepared.rankings.into_iter().map(Vec::into_iter)),
                        prepared.telemetry,
                        sparse_shared_started.elapsed().as_nanos(),
                    )
                }
                SparseStreamStrategy::DocumentBlock
                | SparseStreamStrategy::PostingBlock { .. }
                | SparseStreamStrategy::AdaptiveNative { .. }
                | SparseStreamStrategy::AutoShared { .. }
                | SparseStreamStrategy::ProbeThenShared { .. }
                | SparseStreamStrategy::SharedLazy { .. }
                | SparseStreamStrategy::AutoSharedLazy { .. } => {
                    (None, SharedSparseTelemetry::default(), 0)
                }
            };
        let (mut shared_lazy_streams, shared_lazy_handle): (
            Option<std::vec::IntoIter<SharedPostingBlockStream<'_>>>,
            Option<SharedPostingBlockHandle<'_>>,
        ) = match sparse_strategy {
            SparseStreamStrategy::SharedLazy { batch_size } => {
                let (streams, handle) = build_shared_posting_block_streams(
                    &self.sparse_postings,
                    &sparse_queries,
                    batch_size,
                    &posting_arenas[0],
                    &hardware_counter,
                )
                .map_err(|error| NChannelExactError::SparsePosting(error.to_string()))?;
                (Some(streams.into_iter()), Some(handle))
            }
            SparseStreamStrategy::DocumentBlock
            | SparseStreamStrategy::PostingBlock { .. }
            | SparseStreamStrategy::AdaptiveNative { .. }
            | SparseStreamStrategy::SharedFull
            | SparseStreamStrategy::AutoShared { .. }
            | SparseStreamStrategy::ProbeThenShared { .. }
            | SparseStreamStrategy::AutoSharedLazy { .. } => (None, None),
        };
        let physical = NChannelPhysicalTelemetry {
            sparse_shared,
            sparse_shared_preparation_ns,
            sparse_auto_requested,
            sparse_auto_selected_shared,
            sparse_estimated_unique_posting_elements: sparse_estimate.unique_posting_elements,
            sparse_estimated_logical_posting_elements: sparse_estimate.logical_posting_elements,
            sparse_estimated_deduplicated_posting_elements,
            sparse_probe_then_shared_requested,
            sparse_probe_depth: match sparse_strategy {
                SparseStreamStrategy::ProbeThenShared { probe_depth, .. } => probe_depth,
                SparseStreamStrategy::DocumentBlock
                | SparseStreamStrategy::PostingBlock { .. }
                | SparseStreamStrategy::AdaptiveNative { .. }
                | SparseStreamStrategy::SharedFull
                | SparseStreamStrategy::AutoShared { .. }
                | SparseStreamStrategy::SharedLazy { .. }
                | SparseStreamStrategy::AutoSharedLazy { .. } => 0,
            },
            ..physical
        };
        let mut sources = Vec::<ExactRrfStream<'_>>::with_capacity(queries.len());
        let sparse_source_indices: Vec<_> = queries
            .iter()
            .enumerate()
            .filter_map(|(source, query)| {
                matches!(query, ExactChannelQuery::Sparse(_)).then_some(source)
            })
            .collect();
        for (channel_index, (query, progress)) in
            queries.iter().cloned().zip(&telemetry).enumerate()
        {
            match query {
                ExactChannelQuery::Dense(_) => {
                    if dense_representative_flags[channel_index] {
                        sources.push(Box::new(DenseIdentityStream {
                            inner: dense_streams
                                .next()
                                .expect("one prepared Dense stream per unique Dense query"),
                            telemetry: Rc::clone(progress),
                        }));
                    } else {
                        sources.push(Box::new(std::iter::empty()));
                    }
                }
                ExactChannelQuery::Sparse(query) => match sparse_strategy {
                    SparseStreamStrategy::DocumentBlock => {
                        sources.push(Box::new(SparseIdentityStream {
                            inner: self.sparse.stream(query)?,
                            telemetry: Rc::clone(progress),
                        }));
                    }
                    SparseStreamStrategy::PostingBlock { batch_size } => {
                        let stream = PostingBlockStream::new(
                            &self.sparse_postings,
                            query,
                            batch_size,
                            &posting_arenas[channel_index],
                            &hardware_counter,
                        )
                        .map_err(|error| NChannelExactError::SparsePosting(error.to_string()))?;
                        sources.push(Box::new(PostingSparseIdentityStream {
                            inner: stream,
                            telemetry: Rc::clone(progress),
                        }));
                    }
                    SparseStreamStrategy::ProbeThenShared { batch_size, .. } => {
                        let stream = PostingBlockStream::new(
                            &self.sparse_postings,
                            query,
                            batch_size,
                            &posting_arenas[channel_index],
                            &hardware_counter,
                        )
                        .map_err(|error| NChannelExactError::SparsePosting(error.to_string()))?;
                        sources.push(Box::new(PostingSparseIdentityStream {
                            inner: stream,
                            telemetry: Rc::clone(progress),
                        }));
                    }
                    SparseStreamStrategy::SharedLazy { .. } => {
                        sources.push(Box::new(SharedPostingIdentityStream {
                            inner: shared_lazy_streams
                                .as_mut()
                                .expect("Shared Lazy streams are prepared")
                                .next()
                                .expect("one Shared Lazy stream per Sparse query"),
                            telemetry: Rc::clone(progress),
                        }));
                    }
                    SparseStreamStrategy::AdaptiveNative {
                        initial_limit,
                        growth_factor,
                        batch_size,
                        endpoint_trend_router,
                    } => {
                        sources.push(Box::new(AdaptiveSparseIdentityStream {
                            inner: AdaptiveTopKStream::new(
                                &self.sparse_postings,
                                query,
                                initial_limit,
                                growth_factor,
                                false,
                                batch_size,
                                endpoint_trend_router,
                                &pool,
                                stopped,
                                &hardware_counter,
                            ),
                            telemetry: Rc::clone(progress),
                        }));
                    }
                    SparseStreamStrategy::SharedFull => {
                        sources.push(Box::new(SharedSparseIdentityStream {
                            inner: shared_sparse_streams
                                .as_mut()
                                .expect("Shared Sparse rankings are prepared")
                                .next()
                                .expect("one prepared ranking per Sparse query"),
                            telemetry: Rc::clone(progress),
                        }));
                    }
                    SparseStreamStrategy::AutoShared { .. } => {
                        unreachable!("Auto Shared is resolved before stream construction")
                    }
                    SparseStreamStrategy::AutoSharedLazy { .. } => {
                        unreachable!("Auto Shared Lazy is resolved before stream construction")
                    }
                },
            }
        }
        let replay_identical_sparse = matches!(
            sparse_strategy,
            SparseStreamStrategy::DocumentBlock
                | SparseStreamStrategy::PostingBlock { .. }
                | SparseStreamStrategy::AdaptiveNative { .. }
                | SparseStreamStrategy::ProbeThenShared { .. }
        );
        let (sources, rank_stream_observer) = share_identical_rank_streams(
            &queries,
            sources,
            weights,
            replay_identical_dense,
            replay_identical_sparse,
        );
        let sources = sources
            .into_iter()
            .map(|inner| {
                Box::new(CancellableIdentityStream { inner, stopped }) as ExactRrfStream<'_>
            })
            .collect();
        let mut session = DynamicRrfSession::new(sources, top_k, rrf_k, weights, policy)?;
        let mut physical = physical;
        let execution = match sparse_strategy {
            SparseStreamStrategy::ProbeThenShared { probe_depth, .. } => {
                let mut targets = vec![None; telemetry.len()];
                for &source in &sparse_source_indices {
                    targets[source] = Some(probe_depth);
                }
                match session.advance_until_each(&targets)? {
                    DynamicRrfAdvance::Fixed(execution) => execution,
                    DynamicRrfAdvance::Paused => {
                        let has_reusable_decode = sparse_estimate.unique_posting_elements
                            < sparse_estimate.logical_posting_elements;
                        let has_unfinished_sparse = sparse_source_indices
                            .iter()
                            .any(|&source| !session.source_is_exhausted(source).unwrap_or(true));
                        if has_reusable_decode && has_unfinished_sparse {
                            if stopped.load(Relaxed) {
                                return Err(NChannelExactError::Cancelled);
                            }
                            let started = Instant::now();
                            let prepared = evaluate_shared_sparse(
                                &self.sparse_postings,
                                &sparse_queries,
                                self.point_capacity,
                                &posting_arenas[0],
                                &hardware_counter,
                            )
                            .map_err(NChannelExactError::SparsePosting)?;
                            if stopped.load(Relaxed) {
                                return Err(NChannelExactError::Cancelled);
                            }
                            physical.sparse_shared = prepared.telemetry;
                            physical.sparse_shared_preparation_ns = started.elapsed().as_nanos();
                            physical.sparse_probe_selected_shared = true;
                            for (source, ranking) in
                                sparse_source_indices.iter().copied().zip(prepared.rankings)
                            {
                                if session.source_is_exhausted(source) == Some(true) {
                                    continue;
                                }
                                physical.sparse_prefix_identities_replayed +=
                                    session.source_pulls()[source];
                                session.replace_source(
                                    source,
                                    Box::new(CancellableIdentityStream {
                                        inner: Box::new(ranking.into_iter().map(|point| {
                                            ExtendedPointId::from(u64::from(point.idx))
                                        })),
                                        stopped,
                                    }),
                                )?;
                            }
                        }
                        session.run_to_completion()?
                    }
                }
            }
            SparseStreamStrategy::DocumentBlock
            | SparseStreamStrategy::PostingBlock { .. }
            | SparseStreamStrategy::AdaptiveNative { .. }
            | SparseStreamStrategy::SharedFull
            | SparseStreamStrategy::AutoShared { .. }
            | SparseStreamStrategy::SharedLazy { .. }
            | SparseStreamStrategy::AutoSharedLazy { .. } => session.run_to_completion()?,
        };
        if stopped.load(Relaxed) {
            return Err(NChannelExactError::Cancelled);
        }
        if let Some(handle) = shared_lazy_handle {
            physical.sparse_shared_lazy = handle.telemetry();
        }
        let rank_stream_telemetry = rank_stream_observer.telemetry(&execution.source_pulls);
        physical.logical_rank_streams = rank_stream_telemetry.logical_streams;
        physical.physical_rank_streams = rank_stream_telemetry.physical_streams;
        physical.shared_rank_stream_groups = rank_stream_telemetry.shared_groups;
        physical.logical_rank_pulls = rank_stream_telemetry.logical_pulls;
        physical.physical_rank_pulls = rank_stream_telemetry.physical_pulls;
        physical.rank_replay_buffered_identities = rank_stream_telemetry.buffered_identities;
        Ok(NChannelExactSearchResult {
            execution,
            channels: telemetry.iter().map(|telemetry| telemetry.get()).collect(),
            physical,
        })
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;
    use crate::common::reciprocal_rank_fusion::{DEFAULT_RRF_K, exact_rrf_scoring};
    use crate::types::ScoredPoint;

    fn documents() -> (Vec<DenseDocument>, Vec<SparseDocument>) {
        let dense = (0..97)
            .map(|id| DenseDocument {
                id,
                vector: vec![
                    ((id * 17) % 31) as f32 / 31.0,
                    ((id * 29 + 7) % 37) as f32 / 37.0,
                    ((id * 11 + 3) % 41) as f32 / 41.0,
                ],
            })
            .collect();
        let sparse = (0..97)
            .map(|id| SparseDocument {
                id,
                vector: RemappedSparseVector {
                    indices: vec![0, 1, 2],
                    values: vec![1.0, (id % 13) as f32, ((id * 7) % 17) as f32],
                },
            })
            .collect();
        (dense, sparse)
    }

    fn scored_point(id: u32) -> ScoredPoint {
        ScoredPoint {
            id: ExtendedPointId::from(u64::from(id)),
            version: 0,
            score: 0.0,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    #[test]
    fn pre_cancelled_search_does_not_return_a_partial_result() {
        let (dense_documents, sparse_documents) = documents();
        let index = NChannelExactIndex::build(dense_documents, sparse_documents, 11).unwrap();
        let query = vec![1.0, 0.25, -0.5];
        let stopped = AtomicBool::new(true);

        let error = index
            .search_with_sparse_strategy_and_cancellation(
                vec![ExactChannelQuery::Dense(&query)],
                20,
                DEFAULT_RRF_K,
                None,
                DynamicRrfPolicy::default(),
                SparseStreamStrategy::default(),
                &stopped,
            )
            .expect_err("a pre-cancelled search must fail");

        assert!(matches!(error, NChannelExactError::Cancelled));
    }

    #[test]
    fn one_to_eight_real_channel_streams_match_exhaustive_wrrf() {
        let (dense_documents, sparse_documents) = documents();
        let index =
            NChannelExactIndex::build(dense_documents.clone(), sparse_documents.clone(), 11)
                .unwrap();
        let dense_queries = [
            vec![1.0, 0.25, -0.5],
            vec![-0.25, 1.0, 0.5],
            vec![0.5, -0.75, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let sparse_queries = [
            RemappedSparseVector {
                indices: vec![0, 1],
                values: vec![1.0, 2.0],
            },
            RemappedSparseVector {
                indices: vec![0, 2],
                values: vec![1.0, 3.0],
            },
            RemappedSparseVector {
                indices: vec![1, 2],
                values: vec![2.0, 1.0],
            },
            RemappedSparseVector {
                indices: vec![0, 1, 2],
                values: vec![1.0, 1.0, 1.0],
            },
        ];

        for channel_count in 1usize..=8 {
            let mut queries = Vec::new();
            let mut rankings = Vec::new();
            for channel in 0..channel_count {
                if channel.is_multiple_of(2) {
                    let query = &dense_queries[(channel / 2) % dense_queries.len()];
                    queries.push(ExactChannelQuery::Dense(query));
                    let mut ranking: Vec<_> = dense_documents
                        .iter()
                        .map(|document| {
                            let score = document
                                .vector
                                .iter()
                                .zip(query)
                                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                                .sum::<f64>();
                            (document.id, score)
                        })
                        .collect();
                    ranking.sort_unstable_by(|left, right| {
                        OrderedFloat(right.1)
                            .cmp(&OrderedFloat(left.1))
                            .then_with(|| left.0.cmp(&right.0))
                    });
                    rankings.push(
                        ranking
                            .into_iter()
                            .map(|(id, _)| scored_point(id))
                            .collect(),
                    );
                } else {
                    let query = sparse_queries[(channel / 2) % sparse_queries.len()].clone();
                    queries.push(ExactChannelQuery::Sparse(query.clone()));
                    let mut ranking: Vec<_> = sparse_documents
                        .iter()
                        .filter_map(|document| {
                            document
                                .vector
                                .score(&query)
                                .filter(|score| *score != 0.0)
                                .map(|score| (document.id, score))
                        })
                        .collect();
                    ranking.sort_unstable_by(|left, right| {
                        OrderedFloat(right.1)
                            .cmp(&OrderedFloat(left.1))
                            .then_with(|| left.0.cmp(&right.0))
                    });
                    rankings.push(
                        ranking
                            .into_iter()
                            .map(|(id, _)| scored_point(id))
                            .collect(),
                    );
                }
            }
            let expected: Vec<_> = exact_rrf_scoring(rankings, DEFAULT_RRF_K, None)
                .unwrap()
                .into_iter()
                .take(20)
                .map(|point| point.id)
                .collect();

            let posting_actual = index
                .search_with_sparse_strategy(
                    queries.clone(),
                    20,
                    DEFAULT_RRF_K,
                    None,
                    DynamicRrfPolicy::default(),
                    SparseStreamStrategy::PostingBlock { batch_size: 17 },
                )
                .unwrap();
            let adaptive_actual = index
                .search_with_sparse_strategy(
                    queries.clone(),
                    20,
                    DEFAULT_RRF_K,
                    None,
                    DynamicRrfPolicy::default(),
                    SparseStreamStrategy::AdaptiveNative {
                        initial_limit: 5,
                        growth_factor: 2,
                        batch_size: 17,
                        endpoint_trend_router: false,
                    },
                )
                .unwrap();
            let shared_sparse_actual = index
                .search_with_sparse_strategy(
                    queries.clone(),
                    20,
                    DEFAULT_RRF_K,
                    None,
                    DynamicRrfPolicy::default(),
                    SparseStreamStrategy::SharedFull,
                )
                .unwrap();
            let auto_sparse_actual = index
                .search_with_sparse_strategy(
                    queries.clone(),
                    20,
                    DEFAULT_RRF_K,
                    None,
                    DynamicRrfPolicy::default(),
                    SparseStreamStrategy::AutoShared {
                        batch_size: 17,
                        max_unique_to_logical_milli: 1_000,
                    },
                )
                .unwrap();
            let probe_sparse_actual = index
                .search_with_sparse_strategy(
                    queries.clone(),
                    20,
                    DEFAULT_RRF_K,
                    None,
                    DynamicRrfPolicy::default(),
                    SparseStreamStrategy::ProbeThenShared {
                        batch_size: 17,
                        probe_depth: 2,
                    },
                )
                .unwrap();
            let shared_lazy_actual = index
                .search_with_sparse_strategy(
                    queries.clone(),
                    20,
                    DEFAULT_RRF_K,
                    None,
                    DynamicRrfPolicy::default(),
                    SparseStreamStrategy::SharedLazy { batch_size: 17 },
                )
                .unwrap();
            let auto_shared_lazy_actual = index
                .search_with_sparse_strategy(
                    queries.clone(),
                    20,
                    DEFAULT_RRF_K,
                    None,
                    DynamicRrfPolicy::default(),
                    SparseStreamStrategy::AutoSharedLazy { batch_size: 17 },
                )
                .unwrap();
            let actual = index.search(queries, 20, DEFAULT_RRF_K, None).unwrap();

            assert_eq!(
                actual.execution.point_ids, expected,
                "channels={channel_count}"
            );
            assert_eq!(
                posting_actual.execution.point_ids, expected,
                "posting channels={channel_count}"
            );
            assert_eq!(
                adaptive_actual.execution.point_ids, expected,
                "adaptive channels={channel_count}"
            );
            assert_eq!(
                shared_sparse_actual.execution.point_ids, expected,
                "shared Sparse channels={channel_count}"
            );
            assert_eq!(
                auto_sparse_actual.execution.point_ids, expected,
                "auto Sparse channels={channel_count}"
            );
            assert_eq!(
                probe_sparse_actual.execution.point_ids, expected,
                "probe Sparse channels={channel_count}"
            );
            assert!(
                probe_sparse_actual
                    .physical
                    .sparse_probe_then_shared_requested
            );
            assert_eq!(
                shared_lazy_actual.execution.point_ids, expected,
                "shared Lazy channels={channel_count}"
            );
            assert_eq!(
                auto_shared_lazy_actual.execution.point_ids, expected,
                "auto Shared Lazy channels={channel_count}"
            );
            assert_eq!(actual.channels.len(), channel_count);
        }
    }

    #[test]
    fn identical_dense_queries_share_only_with_identical_wrrf_weights() {
        let (dense_documents, sparse_documents) = documents();
        let query = vec![1.0, 0.25, -0.5];
        let mut expected: Vec<_> = dense_documents
            .iter()
            .map(|document| {
                let score = document
                    .vector
                    .iter()
                    .zip(&query)
                    .map(|(left, right)| f64::from(*left) * f64::from(*right))
                    .sum::<f64>();
                (document.id, score)
            })
            .collect();
        expected.sort_unstable_by(|left, right| {
            OrderedFloat(right.1)
                .cmp(&OrderedFloat(left.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        let expected: Vec<_> = expected
            .into_iter()
            .take(20)
            .map(|(id, _)| ExtendedPointId::from(u64::from(id)))
            .collect();
        let index = NChannelExactIndex::build(dense_documents, sparse_documents, 11).unwrap();
        let queries = vec![
            ExactChannelQuery::Dense(&query),
            ExactChannelQuery::Dense(&query),
        ];

        let shared = index
            .search(queries.clone(), 20, DEFAULT_RRF_K, None)
            .unwrap();
        assert_eq!(shared.execution.point_ids, expected);
        assert_eq!(shared.physical.dense_channels, 2);
        assert_eq!(shared.physical.dense_physical_channels, 1);
        assert_eq!(shared.physical.dense_physical_dot_products, 97);

        let document_major = index
            .search_with_strategies(
                queries.clone(),
                20,
                DEFAULT_RRF_K,
                None,
                DynamicRrfPolicy::default(),
                DenseStreamStrategy::SharedDocumentMajor,
                SparseStreamStrategy::PostingBlock { batch_size: 17 },
            )
            .unwrap();
        assert_eq!(document_major.execution.point_ids, expected);
        assert_eq!(document_major.physical.dense_physical_channels, 2);
        assert_eq!(document_major.physical.physical_rank_streams, 2);
        assert_eq!(document_major.physical.shared_rank_stream_groups, 0);

        let distinct_weights = index
            .search(queries, 20, DEFAULT_RRF_K, Some(&[1.0, 2.0]))
            .unwrap();
        assert_eq!(distinct_weights.execution.point_ids, expected);
        assert_eq!(distinct_weights.physical.dense_channels, 2);
        assert_eq!(distinct_weights.physical.dense_physical_channels, 2);
        assert_eq!(distinct_weights.physical.dense_physical_dot_products, 194);
    }
}
