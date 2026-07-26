// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

use segment::common::operation_error::{OperationError, OperationResult};
use segment::common::reciprocal_rank_fusion::{
    DynamicRrfExecution, DynamicRrfPolicy, DynamicRrfScheduler,
    execute_dynamic_rrf_batches_with_policy,
};
use segment::index::exact_rank_stream::{ExactScoreBatchStream, KWayExactScoreStream};
use segment::types::PointIdType;
use shard::dense_rank_plan::{DenseRankPlan, DenseShardTelemetry};
use shard::exact_shard_stream::ExactShardStream;
use shard::locked_segment::LockedSegment;
use shard::sparse_rank_plan::{SparseRankPlan, SparseShardTelemetry};

use super::source::{exact_score_batch_stream, observed_rank_batch_stream};
use super::{
    DEFAULT_EXACT_BATCH_SIZE, DEFAULT_SPARSE_POSTING_BATCH_SIZE, ExactRrfRequest, ExactRrfResult,
};

const EXACT_FUSION_BATCH_SIZE: usize = 16;

pub struct ExactHybridSession {
    snapshots: Vec<Vec<LockedSegment>>,
    request: ExactRrfRequest,
    stopped: Arc<AtomicBool>,
}

impl ExactHybridSession {
    pub fn new(
        snapshots: Vec<Vec<LockedSegment>>,
        request: ExactRrfRequest,
        stopped: Arc<AtomicBool>,
    ) -> Self {
        Self {
            snapshots,
            request,
            stopped,
        }
    }

    pub fn execute(self) -> OperationResult<ExactRrfResult> {
        let Self {
            snapshots,
            request,
            stopped,
        } = self;
        let batch_size = if request.batch_size == 0 {
            DEFAULT_EXACT_BATCH_SIZE
        } else {
            request.batch_size
        };
        let sparse_posting_batch_size = if request.sparse_posting_batch_size == 0 {
            DEFAULT_SPARSE_POSTING_BATCH_SIZE
        } else {
            request.sparse_posting_batch_size
        };
        // The Collection-owned update barriers freeze the Shard generation.
        // Channel state is owned; individual batches borrow Segment read views.
        let versions = Rc::new(RefCell::new(HashMap::new()));
        let sparse_physical = Rc::new(RefCell::new(Vec::<SparseShardTelemetry>::new()));
        let mut sparse_sources =
            Vec::<ExactScoreBatchStream<'static>>::with_capacity(snapshots.len());
        let mut exhaustive_fallback_sources = 0;
        let mut visible_point_copies = 0;
        for segments in snapshots.iter().cloned() {
            let plan = SparseRankPlan::new(
                &segments,
                request.sparse_using.clone(),
                request.sparse_query.clone(),
                request.filter.clone(),
                sparse_posting_batch_size,
                stopped.clone(),
            )?;
            let stream = ExactShardStream::open(segments, plan, batch_size, stopped.clone())?;
            let telemetry = stream.telemetry();
            exhaustive_fallback_sources += telemetry.exhaustive_fallback_sources;
            visible_point_copies += telemetry.visible_point_copies;
            let physical_source = {
                let mut telemetry = sparse_physical.borrow_mut();
                telemetry.push(stream.telemetry());
                telemetry.len() - 1
            };
            let source_versions = versions.clone();
            let source_physical = sparse_physical.clone();
            let mut stream = stream;
            sparse_sources.push(exact_score_batch_stream(
                move |max_results| {
                    let result = stream.next_batch(max_results);
                    source_physical.borrow_mut()[physical_source] = stream.telemetry();
                    result
                },
                source_versions,
            ));
        }
        let dense_physical = Rc::new(RefCell::new(Vec::<DenseShardTelemetry>::new()));
        let mut dense_sources =
            Vec::<ExactScoreBatchStream<'static>>::with_capacity(snapshots.len());
        for segments in snapshots {
            let plan = DenseRankPlan::new(
                request.dense_using.clone(),
                request.dense_query.clone(),
                request.filter.clone(),
                request.dense_policy,
            );
            let mut stream = ExactShardStream::open(segments, plan, batch_size, stopped.clone())?;
            let physical_source = {
                let mut telemetry = dense_physical.borrow_mut();
                telemetry.push(stream.telemetry());
                telemetry.len() - 1
            };
            let source_versions = versions.clone();
            let source_physical = dense_physical.clone();
            dense_sources.push(exact_score_batch_stream(
                move |max_results| {
                    let result = stream.next_batch(max_results);
                    source_physical.borrow_mut()[physical_source] = stream.telemetry();
                    result
                },
                source_versions,
            ));
        }

        let materialized = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let dense_stream = observed_rank_batch_stream(
            KWayExactScoreStream::new_batched(dense_sources, EXACT_FUSION_BATCH_SIZE)?,
            materialized[0].clone(),
        );
        let sparse_stream = observed_rank_batch_stream(
            KWayExactScoreStream::new_batched(sparse_sources, EXACT_FUSION_BATCH_SIZE)?,
            materialized[1].clone(),
        );
        let execution = execute_dynamic_rrf_batches_with_policy(
            vec![dense_stream, sparse_stream],
            EXACT_FUSION_BATCH_SIZE,
            request.limit,
            request.rrf_k,
            Some(&request.weights),
            DynamicRrfPolicy {
                scheduler: DynamicRrfScheduler::MaxNextContribution,
                ..DynamicRrfPolicy::default()
            },
        )?;
        let DynamicRrfExecution {
            point_ids,
            source_pulls,
            source_exhausted,
            certification_checks,
            stop_reason,
            ..
        } = execution;
        let observed_versions = Rc::try_unwrap(versions)
            .map_err(|_| OperationError::service_error_light("exact RRF version map leaked"))?
            .into_inner();
        let versions = point_ids
            .iter()
            .map(|id| {
                observed_versions
                    .get(id)
                    .copied()
                    .map(|version| (*id, version))
                    .ok_or_else(|| {
                        OperationError::inconsistent_storage(format!(
                            "exact RRF result {id} has no visible version"
                        ))
                    })
            })
            .collect::<OperationResult<HashMap<PointIdType, u64>>>()?;
        let source_points_materialized = materialized
            .into_iter()
            .map(|count| count.load(AtomicOrdering::Relaxed))
            .collect();
        let source_batch_requests = vec![
            dense_physical
                .borrow()
                .iter()
                .map(|telemetry| telemetry.batch_requests)
                .sum(),
            sparse_physical
                .borrow()
                .iter()
                .map(|telemetry| telemetry.batch_requests)
                .sum(),
        ];
        let source_points_received = vec![
            dense_physical
                .borrow()
                .iter()
                .map(|telemetry| telemetry.points_received)
                .sum(),
            sparse_physical
                .borrow()
                .iter()
                .map(|telemetry| telemetry.points_received)
                .sum(),
        ];
        Ok(ExactRrfResult {
            point_ids,
            versions,
            stop_reason,
            source_pulls,
            source_exhausted,
            certification_checks,
            source_points_materialized,
            source_batch_requests,
            source_points_received,
            exhaustive_fallback_sources,
            visible_point_copies,
            shard_count: 0,
        })
    }
}
