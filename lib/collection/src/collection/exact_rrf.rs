// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact Dense/Sparse channel execution followed by dynamic WRRF.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

use segment::common::operation_error::{OperationError, OperationResult};
use segment::common::reciprocal_rank_fusion::{
    DynamicRrfExecution, DynamicRrfPolicy, DynamicRrfScheduler, DynamicRrfStopReason,
    ExactRrfBatchStream, execute_dynamic_rrf_batches_with_policy,
};
use segment::index::dense_rank_state::DenseExecutionPolicy;
use segment::index::exact_rank_stream::{
    ExactScoreBatch, ExactScoreBatchStream, ExactScoredIdentity, KWayExactScoreStream,
};
use segment::types::{ExtendedPointId, Filter, PointIdType, ScoredPoint, VectorNameBuf};
use shard::dense_rank_plan::{DenseRankPlan, DenseShardTelemetry};
use shard::exact_shard_stream::ExactShardStream;
use shard::locked_segment::LockedSegment;
use shard::sparse_rank_plan::SparseRankPlan;
use sparse::common::sparse_vector::SparseVector;

use super::Collection;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::{CollectionError, CollectionResult};

pub const DEFAULT_EXACT_BATCH_SIZE: usize = 64;
pub const DEFAULT_SPARSE_POSTING_BATCH_SIZE: usize = 4_096;
const EXACT_FUSION_BATCH_SIZE: usize = 16;

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

struct ExactRrfCancellation {
    stopped: Arc<AtomicBool>,
    armed: bool,
}

impl ExactRrfCancellation {
    fn new(stopped: Arc<AtomicBool>) -> Self {
        Self {
            stopped,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ExactRrfCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.stopped.store(true, AtomicOrdering::Relaxed);
        }
    }
}

fn exact_score_batch_stream(
    mut next: impl FnMut(usize) -> OperationResult<Vec<ScoredPoint>> + 'static,
    versions: Rc<RefCell<HashMap<PointIdType, u64>>>,
) -> ExactScoreBatchStream<'static> {
    Box::new(move |max_results| {
        let points = next(max_results)?;
        let exhausted = points.len() < max_results;
        let observed_versions = versions.borrow();
        let mut batch_versions = HashMap::with_capacity(points.len());
        for point in &points {
            if let Some(observed) = observed_versions.get(&point.id)
                && *observed != point.version
            {
                return Err(OperationError::inconsistent_storage(format!(
                    "exact RRF observed point {} at conflicting versions {} and {}",
                    point.id, observed, point.version,
                )));
            }
            if let Some(observed) = batch_versions.insert(point.id, point.version)
                && observed != point.version
            {
                return Err(OperationError::inconsistent_storage(format!(
                    "exact RRF batch observed point {} at conflicting versions {} and {}",
                    point.id, observed, point.version,
                )));
            }
        }
        drop(observed_versions);
        versions.borrow_mut().extend(batch_versions);

        Ok(ExactScoreBatch {
            points: points
                .into_iter()
                .map(|point| ExactScoredIdentity {
                    id: point.id,
                    score: point.score,
                })
                .collect(),
            exhausted,
        })
    })
}

fn observed_rank_batch_stream(
    mut merged: KWayExactScoreStream<'static>,
    materialized: Arc<AtomicUsize>,
) -> ExactRrfBatchStream<'static> {
    Box::new(move |max_results| {
        let batch = merged.next_rank_batch(max_results)?;
        materialized.fetch_add(batch.point_ids.len(), AtomicOrdering::Relaxed);
        Ok(batch)
    })
}

/// Collection boundary for the frozen Production API exact-retrieval plan.
pub struct ExactRrfService<'a> {
    collection: &'a Collection,
}

impl<'a> ExactRrfService<'a> {
    pub fn new(collection: &'a Collection) -> Self {
        Self { collection }
    }

    /// Execute the local exact physical plan when every selected Shard has a
    /// readable local replica. Returns `None` when the safe router must use the
    /// ordinary replica/remote path instead. Plan selection cannot change the
    /// exact result contract.
    pub async fn execute(
        &self,
        request: ExactRrfRequest,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
    ) -> CollectionResult<Option<ExactRrfResult>> {
        let targets = {
            let shard_holder = self.collection.shards_holder.read().await;
            shard_holder
                .select_shards(shard_selection)?
                .into_iter()
                .map(|(shard, _)| Arc::clone(shard))
                .collect::<Vec<_>>()
        };

        let mut snapshots = Vec::with_capacity(targets.len());
        for target in targets {
            let Some(snapshot) = target.exact_segment_read_set().await? else {
                log::debug!("exact RRF skipped: selected Shard has no local snapshot");
                return Ok(None);
            };
            snapshots.push(snapshot);
        }

        let stopped = Arc::new(AtomicBool::new(false));
        let task_stopped = stopped.clone();
        let shard_count = snapshots.len();
        let segment_count = snapshots
            .iter()
            .map(|snapshot| snapshot.segment_count())
            .sum::<usize>();
        // ExactRankSession advances one Segment batch at a time with short
        // Qdrant-runtime tasks. Segment count no longer determines resident
        // reader capacity.
        let required_reader_slots = 1;
        let Some(reservation) = self
            .collection
            .search_runtime
            .try_reserve_exact_session(required_reader_slots)
        else {
            // The ordinary exact path remains available. Crucially, no
            // coordinator or cursor has started, so fallback cannot deadlock
            // behind a partially occupied blocking pool.
            log::debug!(
                "exact RRF skipped: reservation unavailable for {shard_count} Shards, {segment_count} Segments and {required_reader_slots} reader slots",
            );
            return Ok(None);
        };
        debug_assert_eq!(reservation.reader_slots(), required_reader_slots);
        let mut cancellation = ExactRrfCancellation::new(stopped.clone());
        let mut task = reservation
            .coordinator()
            .spawn_blocking(move || {
                let frozen_snapshots = snapshots;
                let pinned_generations = frozen_snapshots
                    .iter()
                    .map(|snapshot| snapshot.pin_generation())
                    .collect::<Vec<_>>();
                let segments = pinned_generations
                    .iter()
                    .map(|(_, segments)| segments.clone())
                    .collect();
                // `frozen_snapshots` stays alive in this closure, so every
                // Shard update guard remains held; `pinned_generations` also
                // keeps optimizer publish/rollback behind the fixed handles.
                let result = ExactHybridSession::new(segments, request, task_stopped).execute();
                drop(pinned_generations);
                drop(frozen_snapshots);
                result
            })
            .expect("fresh exact reservation includes its coordinator slot");
        let result = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, &mut task).await {
                Ok(result) => result,
                Err(_) => {
                    return Err(CollectionError::timeout(timeout, "exact RRF"));
                }
            }
        } else {
            task.await
        }
        .map_err(|error| {
            CollectionError::service_error(format!("exact RRF task failed: {error}"))
        })??;
        cancellation.disarm();

        Ok(Some(ExactRrfResult {
            shard_count,
            ..result
        }))
    }
}

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
        let sparse_physical = Rc::new(RefCell::new(Vec::<DenseShardTelemetry>::new()));
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
            .collect::<OperationResult<HashMap<_, _>>>()?;
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

#[cfg(test)]
#[path = "exact_rrf/tests.rs"]
mod tests;
