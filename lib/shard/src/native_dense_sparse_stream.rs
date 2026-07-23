// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Paired Dense/Sparse exact streams backed by one worker per frozen Segment.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::query_context::QueryContext;
use segment::entry::ReadSegmentEntry;
use segment::index::native_dense_stream::NativeDensePolicy;
use segment::segment::Segment;
use segment::types::{Filter, ScoredPoint, VectorNameBuf};
use sparse::common::sparse_vector::SparseVector;

use crate::locked_segment::LockedSegment;
use crate::native_dense_stream::{
    NativeDenseShardStream, materialize_exact as materialize_dense_exact,
};
use crate::native_score_stream::{
    NativeShardPointVersions, NativeStreamWorkerMode, NativeWorkerSpawner, SegmentScoreSource,
    WorkerCommand, WorkerCompletionSignal, serve_pair,
};
use crate::native_sparse_stream::{
    NativeSparseShardStream, build_query_context as build_sparse_query_context,
    materialize_exact as materialize_sparse_exact,
};

type PairedReady = OperationResult<(NativeStreamWorkerMode, NativeStreamWorkerMode)>;

pub struct NativeDenseSparseShardStreams {
    pub dense: NativeDenseShardStream,
    pub sparse: NativeSparseShardStream,
    pub paired_native_workers: usize,
    pub mixed_workers: usize,
    pub materialized_workers: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn open_native_dense_sparse_shard_streams(
    segments: Vec<LockedSegment>,
    dense_vector_name: VectorNameBuf,
    dense_query: Vec<f32>,
    dense_policy: NativeDensePolicy,
    sparse_vector_name: VectorNameBuf,
    sparse_query: SparseVector,
    filter: Option<Filter>,
    source_batch_size: usize,
    sparse_posting_batch_size: usize,
    stopped: Arc<AtomicBool>,
    worker_spawner: NativeWorkerSpawner,
    point_versions: Arc<NativeShardPointVersions>,
) -> OperationResult<NativeDenseSparseShardStreams> {
    if source_batch_size == 0 || sparse_posting_batch_size == 0 {
        return Err(OperationError::validation_error(
            "paired native Shard batch sizes must be positive",
        ));
    }

    let sparse_query_context = Arc::new(build_sparse_query_context(
        &segments,
        &sparse_vector_name,
        &sparse_query,
        stopped.clone(),
    )?);
    let mut dense_sources = Vec::with_capacity(segments.len());
    let mut sparse_sources = Vec::with_capacity(segments.len());
    let mut paired_native_workers = 0;
    let mut mixed_workers = 0;
    let mut materialized_workers = 0;
    let mut pending_workers = Vec::with_capacity(segments.len());
    for (source, segment) in segments.into_iter().enumerate() {
        pending_workers.push(launch_paired_segment_worker(
            source,
            segment,
            dense_vector_name.clone(),
            dense_query.clone(),
            dense_policy,
            sparse_vector_name.clone(),
            sparse_query.clone(),
            filter.clone(),
            sparse_posting_batch_size,
            sparse_query_context.clone(),
            stopped.clone(),
            &worker_spawner,
        )?);
    }
    for pending in pending_workers {
        let (dense, sparse, dense_mode, sparse_mode) = pending.finish()?;
        match (dense_mode, sparse_mode) {
            (NativeStreamWorkerMode::Native, NativeStreamWorkerMode::Native) => {
                paired_native_workers += 1;
            }
            (
                NativeStreamWorkerMode::ExhaustiveFallback,
                NativeStreamWorkerMode::ExhaustiveFallback,
            ) => {
                materialized_workers += 1;
            }
            _ => mixed_workers += 1,
        }
        dense_sources.push(dense);
        sparse_sources.push(sparse);
    }

    Ok(NativeDenseSparseShardStreams {
        dense: NativeDenseShardStream::from_sources(
            dense_sources,
            point_versions.clone(),
            source_batch_size,
            stopped.clone(),
        )?,
        sparse: NativeSparseShardStream::from_sources(
            sparse_sources,
            point_versions,
            source_batch_size,
            stopped,
        )?,
        paired_native_workers,
        mixed_workers,
        materialized_workers,
    })
}

struct PendingPairedSegmentWorker {
    commands: Option<SyncSender<WorkerCommand>>,
    ready: Receiver<PairedReady>,
    completion: Option<Receiver<()>>,
}

impl PendingPairedSegmentWorker {
    fn finish(
        mut self,
    ) -> OperationResult<(
        SegmentScoreSource,
        SegmentScoreSource,
        NativeStreamWorkerMode,
        NativeStreamWorkerMode,
    )> {
        let (dense_mode, sparse_mode) = match self.ready.recv() {
            Ok(result) => result?,
            Err(_) => {
                if let Some(completion) = self.completion.take() {
                    let _ = completion.recv();
                }
                return Err(OperationError::service_error_light(
                    "paired native Segment worker stopped during initialization",
                ));
            }
        };
        let commands = self
            .commands
            .take()
            .expect("pending paired worker owns its command sender");
        let completion = self
            .completion
            .take()
            .expect("pending paired worker owns its completion receiver");
        let (dense, sparse) =
            SegmentScoreSource::new_pair(commands, completion, dense_mode, sparse_mode);
        Ok((dense, sparse, dense_mode, sparse_mode))
    }
}

impl Drop for PendingPairedSegmentWorker {
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            let _ = commands.send(WorkerCommand::Stop);
        }
        if let Some(completion) = self.completion.take() {
            let _ = completion.recv();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_paired_segment_worker(
    source: usize,
    segment: LockedSegment,
    dense_vector_name: VectorNameBuf,
    dense_query: Vec<f32>,
    dense_policy: NativeDensePolicy,
    sparse_vector_name: VectorNameBuf,
    sparse_query: SparseVector,
    filter: Option<Filter>,
    sparse_posting_batch_size: usize,
    sparse_query_context: Arc<QueryContext>,
    stopped: Arc<AtomicBool>,
    worker_spawner: &NativeWorkerSpawner,
) -> OperationResult<PendingPairedSegmentWorker> {
    let (command_tx, command_rx) = sync_channel(1);
    let (ready_tx, ready_rx) = sync_channel(1);
    let (completion_tx, completion_rx) = sync_channel(1);
    worker_spawner.spawn(format!("native-dense-sparse-segment-{source}"), move || {
        let _completion = WorkerCompletionSignal::new(completion_tx);
        run_paired_segment_worker(
            segment,
            dense_vector_name,
            dense_query,
            dense_policy,
            sparse_vector_name,
            sparse_query,
            filter,
            sparse_posting_batch_size,
            sparse_query_context,
            stopped,
            command_rx,
            ready_tx,
        );
    })?;

    Ok(PendingPairedSegmentWorker {
        commands: Some(command_tx),
        ready: ready_rx,
        completion: Some(completion_rx),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_paired_segment_worker(
    segment: LockedSegment,
    dense_vector_name: VectorNameBuf,
    dense_query: Vec<f32>,
    dense_policy: NativeDensePolicy,
    sparse_vector_name: VectorNameBuf,
    sparse_query: SparseVector,
    filter: Option<Filter>,
    sparse_posting_batch_size: usize,
    sparse_query_context: Arc<QueryContext>,
    stopped: Arc<AtomicBool>,
    commands: Receiver<WorkerCommand>,
    ready: SyncSender<PairedReady>,
) {
    match segment {
        LockedSegment::Original(segment) => {
            let segment = segment.read();
            let mut dense_query_context =
                QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                    .with_is_stopped(stopped);
            if let Err(error) = segment.fill_query_context(&mut dense_query_context) {
                let _ = ready.send(Err(error));
                return;
            }
            let dense_segment_context = dense_query_context.get_segment_query_context();
            let sparse_segment_context = sparse_query_context.get_segment_query_context();
            let mut ready = Some(ready);
            let native = segment.with_view(|view| {
                view.with_native_dense_stream(
                    &dense_vector_name,
                    &dense_query,
                    filter.as_ref(),
                    dense_policy,
                    &dense_segment_context,
                    |dense_next| {
                        view.with_native_sparse_stream(
                            &sparse_vector_name,
                            &sparse_query,
                            filter.as_ref(),
                            sparse_posting_batch_size,
                            &sparse_segment_context,
                            |sparse_next| {
                                ready
                                    .take()
                                    .expect("paired native worker sends readiness once")
                                    .send(Ok((
                                        NativeStreamWorkerMode::Native,
                                        NativeStreamWorkerMode::Native,
                                    )))
                                    .map_err(|_| {
                                        OperationError::cancelled(
                                            "paired native Shard stream closed during initialization",
                                        )
                                    })?;
                                serve_pair(dense_next, sparse_next, &commands)
                            },
                        )?;
                        Ok(())
                    },
                )
            });

            if let Err(error) = native
                && let Some(ready) = ready
            {
                if matches!(error, OperationError::WrongSparse) {
                    run_native_dense_materialized_sparse(
                        &*segment,
                        &dense_vector_name,
                        &dense_query,
                        dense_policy,
                        &sparse_vector_name,
                        &sparse_query,
                        filter.as_ref(),
                        &dense_query_context,
                        &sparse_query_context,
                        &commands,
                        ready,
                    );
                } else {
                    let _ = ready.send(Err(error));
                }
            }
        }
        LockedSegment::Proxy(proxy) => {
            let proxy = proxy.read();
            let mut dense_query_context =
                QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                    .with_is_stopped(stopped);
            if let Err(error) = proxy.fill_query_context(&mut dense_query_context) {
                let _ = ready.send(Err(error));
                return;
            }
            run_materialized_pair(
                &*proxy,
                &dense_vector_name,
                &dense_query,
                &sparse_vector_name,
                &sparse_query,
                filter.as_ref(),
                &dense_query_context,
                &sparse_query_context,
                &commands,
                ready,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_native_dense_materialized_sparse(
    segment: &Segment,
    dense_vector_name: &str,
    dense_query: &[f32],
    dense_policy: NativeDensePolicy,
    sparse_vector_name: &str,
    sparse_query: &SparseVector,
    filter: Option<&Filter>,
    dense_query_context: &QueryContext,
    sparse_query_context: &QueryContext,
    commands: &Receiver<WorkerCommand>,
    ready: SyncSender<PairedReady>,
) {
    let sparse = match materialize_sparse_exact(
        segment,
        sparse_vector_name,
        sparse_query,
        filter,
        sparse_query_context,
    ) {
        Ok(points) => points,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let dense_segment_context = dense_query_context.get_segment_query_context();
    let mut ready = Some(ready);
    let result = segment.with_view(|view| {
        view.with_native_dense_stream(
            dense_vector_name,
            dense_query,
            filter,
            dense_policy,
            &dense_segment_context,
            |dense_next| {
                let mut sparse = materialized_next(sparse);
                ready
                    .take()
                    .expect("paired mixed worker sends readiness once")
                    .send(Ok((
                        NativeStreamWorkerMode::Native,
                        NativeStreamWorkerMode::ExhaustiveFallback,
                    )))
                    .map_err(|_| {
                        OperationError::cancelled(
                            "paired mixed Shard stream closed during initialization",
                        )
                    })?;
                serve_pair(dense_next, &mut sparse, commands)
            },
        )
    });
    if let Err(error) = result
        && let Some(ready) = ready
    {
        let _ = ready.send(Err(error));
    }
}

#[allow(clippy::too_many_arguments)]
fn run_materialized_pair(
    segment: &dyn ReadSegmentEntry,
    dense_vector_name: &str,
    dense_query: &[f32],
    sparse_vector_name: &str,
    sparse_query: &SparseVector,
    filter: Option<&Filter>,
    dense_query_context: &QueryContext,
    sparse_query_context: &QueryContext,
    commands: &Receiver<WorkerCommand>,
    ready: SyncSender<PairedReady>,
) {
    let dense = materialize_dense_exact(
        segment,
        dense_vector_name,
        dense_query,
        filter,
        dense_query_context,
    );
    let sparse = materialize_sparse_exact(
        segment,
        sparse_vector_name,
        sparse_query,
        filter,
        sparse_query_context,
    );
    match (dense, sparse) {
        (Ok(dense), Ok(sparse)) => {
            if ready
                .send(Ok((
                    NativeStreamWorkerMode::ExhaustiveFallback,
                    NativeStreamWorkerMode::ExhaustiveFallback,
                )))
                .is_ok()
            {
                let mut dense = materialized_next(dense);
                let mut sparse = materialized_next(sparse);
                let _ = serve_pair(&mut dense, &mut sparse, commands);
            }
        }
        (Err(error), _) | (_, Err(error)) => {
            let _ = ready.send(Err(error));
        }
    }
}

fn materialized_next(
    points: Vec<ScoredPoint>,
) -> impl FnMut() -> OperationResult<Option<ScoredPoint>> {
    let mut points = VecDeque::from(points);
    move || Ok(points.pop_front())
}
