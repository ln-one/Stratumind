// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::collections::HashMap;

use segment::common::reciprocal_rank_fusion::DynamicRrfStopReason;
use segment::index::dense_rank_state::DenseExecutionPolicy;
use segment::types::{ExtendedPointId, Filter, PointIdType, VectorNameBuf};
use sparse::common::sparse_vector::SparseVector;

pub const DEFAULT_EXACT_BATCH_SIZE: usize = 64;
pub const DEFAULT_SPARSE_POSTING_BATCH_SIZE: usize = 4_096;

#[derive(Clone, Debug)]
pub struct ExactRrfRequest {
    pub dense_query: Vec<f32>,
    pub dense_using: VectorNameBuf,
    pub sparse_query: SparseVector,
    pub sparse_using: VectorNameBuf,
    pub filter: Option<Filter>,
    pub limit: usize,
    pub rrf_k: usize,
    pub weights: [f32; 2],
    pub batch_size: usize,
    pub sparse_posting_batch_size: usize,
    pub dense_policy: DenseExecutionPolicy,
}

#[derive(Debug)]
pub struct ExactRrfResult {
    pub point_ids: Vec<ExtendedPointId>,
    pub versions: HashMap<PointIdType, u64>,
    pub stop_reason: DynamicRrfStopReason,
    pub source_pulls: Vec<usize>,
    pub source_exhausted: Vec<bool>,
    pub certification_checks: usize,
    pub source_points_materialized: Vec<usize>,
    /// Physical Segment-session reply batches fetched per channel.
    pub source_batch_requests: Vec<usize>,
    /// Physical scored points received per channel, including buffered
    /// lookahead not yet consumed by WRRF.
    pub source_points_received: Vec<usize>,
    pub exhaustive_fallback_sources: usize,
    pub visible_point_copies: usize,
    pub shard_count: usize,
}
