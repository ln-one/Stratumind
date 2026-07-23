// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Native exact Dense/Sparse channel execution followed by dynamic WRRF.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

use segment::common::operation_error::{OperationError, OperationResult};
use segment::common::reciprocal_rank_fusion::{
    DynamicRrfExecution, DynamicRrfPolicy, DynamicRrfScheduler, DynamicRrfStopReason,
    ExactRrfStream, execute_dynamic_rrf_with_policy,
};
use segment::index::exact_score_stream::{
    ExactScoreStream, ExactScoredIdentity, KWayExactScoreStream,
};
use segment::index::native_dense_stream::NativeDensePolicy;
use segment::types::{ExtendedPointId, Filter, PointIdType, ScoredPoint, VectorNameBuf};
use shard::locked_segment::LockedSegment;
use shard::native_dense_stream::{
    NativeDenseShardStream, NativeDenseShardTelemetry,
    materialize_shard_exact as materialize_dense_shard_exact,
};
use shard::native_sparse_stream::{
    NativeSparseShardStream, materialize_shard_exact as materialize_sparse_shard_exact,
};
use shard::{NativeShardPointVersionsCache, NativeWorkerSpawner};
use sparse::common::sparse_vector::SparseVector;

use super::Collection;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::{CollectionError, CollectionResult};
use crate::shards::local_shard::NativeSegmentSnapshot;

pub const DEFAULT_NATIVE_EXACT_BATCH_SIZE: usize = 64;
pub const DEFAULT_NATIVE_SPARSE_POSTING_BATCH_SIZE: usize = 4_096;

#[derive(Clone, Debug)]
pub struct NativeExactRrfRequest {
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
    pub dense_policy: NativeDensePolicy,
}

#[derive(Debug)]
pub struct NativeExactRrfResult {
    pub point_ids: Vec<ExtendedPointId>,
    pub versions: HashMap<PointIdType, u64>,
    pub stop_reason: DynamicRrfStopReason,
    pub source_pulls: Vec<usize>,
    pub source_exhausted: Vec<bool>,
    pub certification_checks: usize,
    pub source_points_materialized: Vec<usize>,
    /// Physical Segment-session reply batches fetched per channel.
    pub source_worker_pull_batches: Vec<usize>,
    /// Physical scored points received per channel, including buffered
    /// lookahead not yet consumed by WRRF.
    pub source_worker_points_received: Vec<usize>,
    pub exhaustive_fallback_sources: usize,
    pub visible_points: usize,
    pub shard_count: usize,
}

struct NativeExactCancellation {
    stopped: Arc<AtomicBool>,
    armed: bool,
}

impl NativeExactCancellation {
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

impl Drop for NativeExactCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.stopped.store(true, AtomicOrdering::Relaxed);
        }
    }
}

fn exact_score_stream(
    mut next: impl FnMut() -> OperationResult<Option<ScoredPoint>> + 'static,
    versions: Rc<RefCell<HashMap<PointIdType, u64>>>,
) -> ExactScoreStream<'static> {
    Box::new(std::iter::from_fn(move || match next() {
        Ok(Some(point)) => {
            let mut versions = versions.borrow_mut();
            if let Some(observed) = versions.get(&point.id)
                && *observed != point.version
            {
                return Some(Err(OperationError::inconsistent_storage(format!(
                    "native exact RRF observed point {} at conflicting versions {} and {}",
                    point.id, observed, point.version,
                ))));
            }
            versions.insert(point.id, point.version);
            Some(Ok(ExactScoredIdentity {
                id: point.id,
                score: point.score,
            }))
        }
        Ok(None) => None,
        Err(error) => Some(Err(error)),
    }))
}

fn observed_rank_stream(
    merged: KWayExactScoreStream<'static>,
    materialized: Arc<AtomicUsize>,
) -> ExactRrfStream<'static> {
    Box::new(merged.map(move |point| {
        point.map(|point| {
            materialized.fetch_add(1, AtomicOrdering::Relaxed);
            point.id
        })
    }))
}

struct NativeRrfCoreResult {
    point_ids: Vec<ExtendedPointId>,
    versions: HashMap<PointIdType, u64>,
    stop_reason: DynamicRrfStopReason,
    source_pulls: Vec<usize>,
    source_exhausted: Vec<bool>,
    certification_checks: usize,
    source_points_materialized: Vec<usize>,
}

struct NativeFrozenShard {
    segments: Vec<LockedSegment>,
    point_versions_cache: Arc<NativeShardPointVersionsCache>,
}

fn frozen_shards(snapshots: &[NativeSegmentSnapshot]) -> Vec<NativeFrozenShard> {
    snapshots
        .iter()
        .map(|snapshot| NativeFrozenShard {
            segments: snapshot.segments.clone(),
            point_versions_cache: snapshot.point_versions_cache.clone(),
        })
        .collect()
}

fn execute_rrf_sources(
    dense_sources: Vec<ExactScoreStream<'static>>,
    sparse_sources: Vec<ExactScoreStream<'static>>,
    versions: Rc<RefCell<HashMap<PointIdType, u64>>>,
    limit: usize,
    rrf_k: usize,
    weights: [f32; 2],
) -> OperationResult<NativeRrfCoreResult> {
    let materialized = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
    let dense_stream = observed_rank_stream(
        KWayExactScoreStream::new(dense_sources)?,
        materialized[0].clone(),
    );
    let sparse_stream = observed_rank_stream(
        KWayExactScoreStream::new(sparse_sources)?,
        materialized[1].clone(),
    );
    let DynamicRrfExecution {
        point_ids,
        source_pulls,
        source_exhausted,
        certification_checks,
        stop_reason,
        ..
    } = execute_dynamic_rrf_with_policy(
        vec![dense_stream, sparse_stream],
        limit,
        rrf_k,
        Some(&weights),
        DynamicRrfPolicy {
            scheduler: DynamicRrfScheduler::MaxNextContribution,
            ..DynamicRrfPolicy::default()
        },
    )?;
    let observed_versions = Rc::try_unwrap(versions)
        .map_err(|_| OperationError::service_error_light("native exact RRF version map leaked"))?
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
                        "native exact RRF result {id} has no visible version"
                    ))
                })
        })
        .collect::<OperationResult<HashMap<_, _>>>()?;
    let source_points_materialized = materialized
        .into_iter()
        .map(|count| count.load(AtomicOrdering::Relaxed))
        .collect();

    Ok(NativeRrfCoreResult {
        point_ids,
        versions,
        stop_reason,
        source_pulls,
        source_exhausted,
        certification_checks,
        source_points_materialized,
    })
}

impl Collection {
    /// Execute the local-native physical plan when every selected Shard has a
    /// readable local replica. Returns `None` when the safe router must use the
    /// ordinary replica/remote path instead. Plan selection cannot change the
    /// exact result contract.
    pub async fn native_exact_rrf(
        &self,
        request: NativeExactRrfRequest,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
    ) -> CollectionResult<Option<NativeExactRrfResult>> {
        let targets = {
            let shard_holder = self.shards_holder.read().await;
            shard_holder
                .select_shards(shard_selection)?
                .into_iter()
                .map(|(shard, _)| Arc::clone(shard))
                .collect::<Vec<_>>()
        };

        let mut snapshots = Vec::with_capacity(targets.len());
        for target in targets {
            let Some(snapshot) = target.native_segment_snapshot().await? else {
                log::debug!(
                    "native exact RRF skipped: selected Shard has no local native snapshot"
                );
                return Ok(None);
            };
            snapshots.push(snapshot);
        }

        let stopped = Arc::new(AtomicBool::new(false));
        let task_stopped = stopped.clone();
        let shard_count = snapshots.len();
        let segment_count = snapshots
            .iter()
            .map(|snapshot| snapshot.segments.len())
            .sum::<usize>();
        let Some(required_worker_slots) = segment_count.checked_mul(2) else {
            log::debug!("native exact RRF skipped: Segment worker count overflow");
            return Ok(None);
        };
        let mut cancellation = NativeExactCancellation::new(stopped.clone());
        let mut task = if let Some(reservation) = self
            .search_runtime
            .try_reserve_exact_workers(required_worker_slots)
        {
            debug_assert_eq!(reservation.worker_slots(), required_worker_slots);
            let worker_reservation = reservation.workers();
            let worker_spawner = NativeWorkerSpawner::new(move |_name, worker| {
                worker_reservation.spawn_blocking(worker).ok_or_else(|| {
                    OperationError::service_error_light(
                        "native exact session attempted to exceed its reserved worker capacity",
                    )
                })?;
                Ok(())
            });
            reservation
                .coordinator()
                .spawn_blocking(move || {
                    let shards = frozen_shards(&snapshots);
                    let result =
                        execute_native_exact_rrf(shards, request, task_stopped, worker_spawner);
                    drop(snapshots);
                    result
                })
                .expect("fresh exact reservation includes its coordinator slot")
        } else {
            // The frozen eager plan needs only one blocking coordinator. It
            // materializes each Segment through Qdrant's exact kernels and is
            // result-equivalent to the resumable cursor plan, so low runtime
            // capacity changes latency rather than correctness/availability.
            log::debug!(
                "native exact RRF using bounded eager fallback for {shard_count} Shards, {segment_count} Segments and {required_worker_slots} requested cursor workers",
            );
            tokio::task::spawn_blocking(move || {
                let shards = frozen_shards(&snapshots);
                let result = execute_materialized_exact_rrf(shards, request, task_stopped);
                drop(snapshots);
                result
            })
        };
        let result = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, &mut task).await {
                Ok(result) => result,
                Err(_) => {
                    return Err(CollectionError::timeout(timeout, "native exact RRF"));
                }
            }
        } else {
            task.await
        }
        .map_err(|error| {
            CollectionError::service_error(format!("native exact RRF task failed: {error}"))
        })??;
        cancellation.disarm();

        Ok(Some(NativeExactRrfResult {
            shard_count,
            ..result
        }))
    }
}

fn execute_native_exact_rrf(
    snapshots: Vec<NativeFrozenShard>,
    request: NativeExactRrfRequest,
    stopped: Arc<AtomicBool>,
    worker_spawner: NativeWorkerSpawner,
) -> OperationResult<NativeExactRrfResult> {
    let batch_size = if request.batch_size == 0 {
        DEFAULT_NATIVE_EXACT_BATCH_SIZE
    } else {
        request.batch_size
    };
    let sparse_posting_batch_size = if request.sparse_posting_batch_size == 0 {
        DEFAULT_NATIVE_SPARSE_POSTING_BATCH_SIZE
    } else {
        request.sparse_posting_batch_size
    };
    let point_versions = snapshots
        .iter()
        .map(|snapshot| {
            snapshot
                .point_versions_cache
                .get_or_build(&snapshot.segments, &stopped)
        })
        .collect::<OperationResult<Vec<_>>>()?;

    // Sparse workers are opened for every Segment first. Their pinned read
    // views freeze the corpus while Dense opens over the same identities.
    let versions = Rc::new(RefCell::new(HashMap::new()));
    let sparse_physical = Rc::new(RefCell::new(Vec::<NativeDenseShardTelemetry>::new()));
    let mut sparse_sources = Vec::<ExactScoreStream<'static>>::with_capacity(snapshots.len());
    let mut exhaustive_fallback_sources = 0;
    for (shard, snapshot) in snapshots.iter().enumerate() {
        let stream = NativeSparseShardStream::open_with_point_versions(
            snapshot.segments.clone(),
            request.sparse_using.clone(),
            request.sparse_query.clone(),
            request.filter.clone(),
            batch_size,
            sparse_posting_batch_size,
            stopped.clone(),
            worker_spawner.clone(),
            point_versions[shard].clone(),
        )?;
        exhaustive_fallback_sources += stream.telemetry().exhaustive_fallback_sources;
        let physical_source = {
            let mut telemetry = sparse_physical.borrow_mut();
            telemetry.push(stream.telemetry());
            telemetry.len() - 1
        };
        let source_versions = versions.clone();
        let source_physical = sparse_physical.clone();
        let mut stream = stream;
        sparse_sources.push(exact_score_stream(
            move || {
                let result = stream.next_result();
                source_physical.borrow_mut()[physical_source] = stream.telemetry();
                result
            },
            source_versions,
        ));
    }
    let dense_physical = Rc::new(RefCell::new(Vec::<NativeDenseShardTelemetry>::new()));
    let mut dense_sources = Vec::<ExactScoreStream<'static>>::with_capacity(snapshots.len());
    for (shard, snapshot) in snapshots.iter().enumerate() {
        let mut stream = NativeDenseShardStream::open_with_point_versions(
            snapshot.segments.clone(),
            request.dense_using.clone(),
            request.dense_query.clone(),
            request.filter.clone(),
            request.dense_policy,
            batch_size,
            stopped.clone(),
            worker_spawner.clone(),
            point_versions[shard].clone(),
        )?;
        let physical_source = {
            let mut telemetry = dense_physical.borrow_mut();
            telemetry.push(stream.telemetry());
            telemetry.len() - 1
        };
        let source_versions = versions.clone();
        let source_physical = dense_physical.clone();
        dense_sources.push(exact_score_stream(
            move || {
                let result = stream.next_result();
                source_physical.borrow_mut()[physical_source] = stream.telemetry();
                result
            },
            source_versions,
        ));
    }

    let NativeRrfCoreResult {
        point_ids,
        versions,
        stop_reason,
        source_pulls,
        source_exhausted,
        certification_checks,
        source_points_materialized,
    } = execute_rrf_sources(
        dense_sources,
        sparse_sources,
        versions,
        request.limit,
        request.rrf_k,
        request.weights,
    )?;
    let source_worker_pull_batches = vec![
        dense_physical
            .borrow()
            .iter()
            .map(|telemetry| telemetry.worker_pull_batches)
            .sum(),
        sparse_physical
            .borrow()
            .iter()
            .map(|telemetry| telemetry.worker_pull_batches)
            .sum(),
    ];
    let source_worker_points_received = vec![
        dense_physical
            .borrow()
            .iter()
            .map(|telemetry| telemetry.worker_points_received)
            .sum(),
        sparse_physical
            .borrow()
            .iter()
            .map(|telemetry| telemetry.worker_points_received)
            .sum(),
    ];
    let visible_points = dense_physical
        .borrow()
        .iter()
        .map(|telemetry| telemetry.visible_points)
        .sum();

    Ok(NativeExactRrfResult {
        point_ids,
        versions,
        stop_reason,
        source_pulls,
        source_exhausted,
        certification_checks,
        source_points_materialized,
        source_worker_pull_batches,
        source_worker_points_received,
        exhaustive_fallback_sources,
        visible_points,
        shard_count: 0,
    })
}

fn execute_materialized_exact_rrf(
    snapshots: Vec<NativeFrozenShard>,
    request: NativeExactRrfRequest,
    stopped: Arc<AtomicBool>,
) -> OperationResult<NativeExactRrfResult> {
    let point_versions = snapshots
        .iter()
        .map(|snapshot| {
            snapshot
                .point_versions_cache
                .get_or_build(&snapshot.segments, &stopped)
        })
        .collect::<OperationResult<Vec<_>>>()?;
    let visible_points = point_versions.iter().map(|versions| versions.len()).sum();
    let exhaustive_fallback_sources = snapshots
        .iter()
        .map(|snapshot| snapshot.segments.len())
        .sum::<usize>()
        .saturating_mul(2);

    let versions = Rc::new(RefCell::new(HashMap::new()));
    let mut source_worker_points_received = vec![0_usize, 0_usize];
    let mut dense_sources = Vec::<ExactScoreStream<'static>>::with_capacity(snapshots.len());
    let mut sparse_sources = Vec::<ExactScoreStream<'static>>::with_capacity(snapshots.len());

    for (shard, snapshot) in snapshots.iter().enumerate() {
        let dense = materialize_dense_shard_exact(
            &snapshot.segments,
            &request.dense_using,
            &request.dense_query,
            request.filter.as_ref(),
            stopped.clone(),
            &point_versions[shard],
        )?;
        source_worker_points_received[0] += dense.len();
        let mut dense = dense.into_iter();
        dense_sources.push(exact_score_stream(
            move || Ok(dense.next()),
            versions.clone(),
        ));

        let sparse = materialize_sparse_shard_exact(
            &snapshot.segments,
            &request.sparse_using,
            &request.sparse_query,
            request.filter.as_ref(),
            stopped.clone(),
            &point_versions[shard],
        )?;
        source_worker_points_received[1] += sparse.len();
        let mut sparse = sparse.into_iter();
        sparse_sources.push(exact_score_stream(
            move || Ok(sparse.next()),
            versions.clone(),
        ));
    }

    let NativeRrfCoreResult {
        point_ids,
        versions,
        stop_reason,
        source_pulls,
        source_exhausted,
        certification_checks,
        source_points_materialized,
    } = execute_rrf_sources(
        dense_sources,
        sparse_sources,
        versions,
        request.limit,
        request.rrf_k,
        request.weights,
    )?;

    Ok(NativeExactRrfResult {
        point_ids,
        versions,
        stop_reason,
        source_pulls,
        source_exhausted,
        certification_checks,
        source_points_materialized,
        source_worker_pull_batches: vec![0, 0],
        source_worker_points_received,
        exhaustive_fallback_sources,
        visible_points,
        shard_count: 0,
    })
}

#[cfg(test)]
#[path = "native_exact_rrf/tests.rs"]
mod tests;
