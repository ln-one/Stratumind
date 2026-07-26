// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use super::DynamicRrfSession;
use super::types::{DynamicRrfExecution, DynamicRrfPolicy, ExactRrfBatchStream, ExactRrfStream};
use crate::common::operation_error::OperationResult;

pub fn execute_dynamic_rrf(
    sources: Vec<ExactRrfStream<'_>>,
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
) -> OperationResult<DynamicRrfExecution> {
    execute_dynamic_rrf_with_policy(sources, top_k, rrf_k, weights, DynamicRrfPolicy::default())
}

pub fn execute_dynamic_rrf_with_policy(
    sources: Vec<ExactRrfStream<'_>>,
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
    policy: DynamicRrfPolicy,
) -> OperationResult<DynamicRrfExecution> {
    DynamicRrfSession::new(sources, top_k, rrf_k, weights, policy)?.run_to_completion()
}

/// Batch-oriented equivalent of [`execute_dynamic_rrf_with_policy`].
pub fn execute_dynamic_rrf_batches_with_policy(
    sources: Vec<ExactRrfBatchStream<'_>>,
    source_batch_size: usize,
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
    policy: DynamicRrfPolicy,
) -> OperationResult<DynamicRrfExecution> {
    DynamicRrfSession::new_batched(sources, source_batch_size, top_k, rrf_k, weights, policy)?
        .run_to_completion()
}
