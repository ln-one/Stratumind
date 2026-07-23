// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Dense stream merged across a frozen set of Segments.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use common::counter::hardware_accumulator::HwMeasurementAcc;
use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::entry::ReadSegmentEntry;
use segment::index::native_dense_stream::NativeDensePolicy;
use segment::types::{Filter, ScoredPoint, SearchParams, VectorNameBuf, WithPayload, WithVector};

use crate::locked_segment::LockedSegment;
use crate::native_score_stream::{
    NativeShardPointVersions, NativeShardScoreStream, NativeShardStreamTelemetry,
    NativeStreamWorkerMode, NativeWorkerSpawner, SegmentScoreSource, WorkerCommand,
    WorkerCompletionSignal, serve_materialized, serve_native,
};

pub type NativeDenseShardTelemetry = NativeShardStreamTelemetry;

/// One exact Dense rank stream for all Segments in a Shard snapshot.
///
/// Original Segments use the native certified cursor. Proxy Segments are
/// materialized exactly once as a safe fallback. Both plans are exhaustive
/// order equivalent and no repeated Top-N query is issued.
pub struct NativeDenseShardStream {
    inner: NativeShardScoreStream,
}

impl NativeDenseShardStream {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: Vec<f32>,
        filter: Option<Filter>,
        policy: NativeDensePolicy,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
        worker_spawner: NativeWorkerSpawner,
    ) -> OperationResult<Self> {
        let point_versions = NativeShardPointVersions::build(&segments, &stopped)?;
        Self::open_with_point_versions(
            segments,
            vector_name,
            query,
            filter,
            policy,
            batch_size,
            stopped,
            worker_spawner,
            point_versions,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_with_point_versions(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: Vec<f32>,
        filter: Option<Filter>,
        policy: NativeDensePolicy,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
        worker_spawner: NativeWorkerSpawner,
        point_versions: Arc<NativeShardPointVersions>,
    ) -> OperationResult<Self> {
        if batch_size == 0 {
            return Err(OperationError::validation_error(
                "native Dense Shard batch size must be positive",
            ));
        }

        let mut sources = Vec::with_capacity(segments.len());
        for (source, segment) in segments.into_iter().enumerate() {
            sources.push(spawn_segment_worker(
                source,
                segment,
                vector_name.clone(),
                query.clone(),
                filter.clone(),
                policy,
                stopped.clone(),
                &worker_spawner,
            )?);
        }

        Ok(Self {
            inner: NativeShardScoreStream::open(
                sources,
                point_versions,
                batch_size,
                stopped,
                "Dense",
            )?,
        })
    }

    pub fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        self.inner.next_result()
    }

    pub fn telemetry(&self) -> NativeDenseShardTelemetry {
        self.inner.telemetry()
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_segment_worker(
    source: usize,
    segment: LockedSegment,
    vector_name: VectorNameBuf,
    query: Vec<f32>,
    filter: Option<Filter>,
    policy: NativeDensePolicy,
    stopped: Arc<AtomicBool>,
    worker_spawner: &NativeWorkerSpawner,
) -> OperationResult<SegmentScoreSource> {
    let (command_tx, command_rx) = sync_channel(1);
    let (ready_tx, ready_rx) = sync_channel(1);
    let (completion_tx, completion_rx) = sync_channel(1);
    worker_spawner.spawn(format!("native-dense-segment-{source}"), move || {
        let _completion = WorkerCompletionSignal::new(completion_tx);
        run_segment_worker(
            segment,
            vector_name,
            query,
            filter,
            policy,
            stopped,
            command_rx,
            ready_tx,
        );
    })?;

    let mode = match ready_rx.recv() {
        Ok(result) => result?,
        Err(_) => {
            let _ = completion_rx.recv();
            return Err(OperationError::service_error_light(
                "native Dense Segment worker stopped during initialization",
            ));
        }
    };
    Ok(SegmentScoreSource::new(
        command_tx,
        completion_rx,
        mode,
        "Dense",
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_segment_worker(
    segment: LockedSegment,
    vector_name: VectorNameBuf,
    query: Vec<f32>,
    filter: Option<Filter>,
    policy: NativeDensePolicy,
    stopped: Arc<AtomicBool>,
    commands: Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<NativeStreamWorkerMode>>,
) {
    match segment {
        LockedSegment::Original(segment) => {
            let segment = segment.read();
            let mut query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                .with_is_stopped(stopped);
            if let Err(error) = segment.fill_query_context(&mut query_context) {
                let _ = ready.send(Err(error));
                return;
            }
            let segment_query_context = query_context.get_segment_query_context();
            let mut ready = Some(ready);
            let native = segment.with_view(|view| {
                view.with_native_dense_stream(
                    &vector_name,
                    &query,
                    filter.as_ref(),
                    policy,
                    &segment_query_context,
                    |next| {
                        ready
                            .take()
                            .expect("native worker sends readiness once")
                            .send(Ok(NativeStreamWorkerMode::Native))
                            .map_err(|_| {
                                OperationError::cancelled(
                                    "native Dense Shard stream closed during initialization",
                                )
                            })?;
                        serve_native(next, &commands)
                    },
                )
            });
            if let Err(error) = native
                && let Some(ready) = ready
            {
                let _ = ready.send(Err(error));
            }
        }
        LockedSegment::Proxy(proxy) => {
            let proxy = proxy.read();
            let query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                .with_is_stopped(stopped);
            run_materialized_worker(
                &*proxy,
                &vector_name,
                &query,
                filter.as_ref(),
                &query_context,
                &commands,
                ready,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_materialized_worker(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &[f32],
    filter: Option<&Filter>,
    query_context: &QueryContext,
    commands: &Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<NativeStreamWorkerMode>>,
) {
    match materialize_exact(segment, vector_name, query, filter, query_context) {
        Ok(points) => {
            if ready
                .send(Ok(NativeStreamWorkerMode::ExhaustiveFallback))
                .is_ok()
            {
                serve_materialized(points, commands);
            }
        }
        Err(error) => {
            let _ = ready.send(Err(error));
        }
    }
}

fn materialize_exact(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &[f32],
    filter: Option<&Filter>,
    query_context: &QueryContext,
) -> OperationResult<Vec<ScoredPoint>> {
    let query_vector: QueryVector = VectorInternal::Dense(query.to_vec()).into();
    let query_vectors = [&query_vector];
    let params = SearchParams {
        exact: true,
        ..Default::default()
    };
    let mut result = segment
        .search_batch(
            vector_name,
            &query_vectors,
            &WithPayload::default(),
            &WithVector::default(),
            filter,
            segment.available_point_count_without_deferred(),
            Some(&params),
            &query_context.get_segment_query_context(),
        )?
        .pop()
        .unwrap_or_default();
    result.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.id.cmp(&right.id))
            .then_with(|| right.version.cmp(&left.version))
    });
    Ok(result)
}

/// Materialize one Shard's complete authoritative Dense order without
/// keeping one blocking cursor worker per Segment. This is the exact bounded-
/// concurrency fallback when the native session cannot reserve every cursor.
#[allow(clippy::too_many_arguments)]
pub fn materialize_shard_exact(
    segments: &[LockedSegment],
    vector_name: &str,
    query: &[f32],
    filter: Option<&Filter>,
    stopped: Arc<AtomicBool>,
    point_versions: &NativeShardPointVersions,
) -> OperationResult<Vec<ScoredPoint>> {
    let mut points = Vec::new();
    for (source, segment) in segments.iter().enumerate() {
        let segment = segment.get().read();
        let mut query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
            .with_is_stopped(stopped.clone());
        segment.fill_query_context(&mut query_context)?;
        points.extend(
            materialize_exact(&*segment, vector_name, query, filter, &query_context)?
                .into_iter()
                .filter(|point| point_versions.contains(source, point)),
        );
    }
    points.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.id.cmp(&right.id))
            .then_with(|| right.version.cmp(&left.version))
    });
    let mut seen = HashSet::with_capacity(points.len());
    if let Some(duplicate) = points.iter().find(|point| !seen.insert(point.id)) {
        return Err(OperationError::inconsistent_storage(format!(
            "materialized Dense Shard order contains authoritative point {} more than once",
            duplicate.id,
        )));
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering as AtomicOrdering;

    use common::counter::hardware_counter::HardwareCounterCell;
    use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
    use segment::entry::SegmentEntry;
    use segment::json_path::JsonPath;
    use segment::payload_json;
    use segment::segment::Segment;
    use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
    use segment::types::{Condition, Distance, FieldCondition, PointIdType};
    use tempfile::TempDir;

    use super::*;

    fn make_segment(path: &TempDir, lane: u64) -> (Segment, Vec<(PointIdType, f32)>) {
        let mut segment = build_simple_segment(path.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let mut expected = Vec::new();
        for index in 0..256u64 {
            let id: PointIdType = (20_000 - (index * 2 + lane)).into();
            let score = 1.0 + (index % 5) as f32;
            segment
                .upsert_point(
                    index,
                    id,
                    only_default_vector(&[score, index as f32]),
                    &hardware_counter,
                )
                .unwrap();
            let visible = index % 3 != 1;
            segment
                .set_full_payload(
                    index,
                    id,
                    &payload_json! {"visible": visible},
                    &hardware_counter,
                )
                .unwrap();
            if visible {
                expected.push((id, score));
            }
        }
        (segment, expected)
    }

    #[test]
    fn shard_stream_merges_native_segments_exactly_and_resumes() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let (first, mut expected) = make_segment(&first_dir, 0);
        let (second, second_expected) = make_segment(&second_dir, 1);
        expected.extend(second_expected);
        expected.sort_unstable_by(|left, right| {
            OrderedFloat(right.1)
                .cmp(&OrderedFloat(left.1))
                .then_with(|| left.0.cmp(&right.0))
        });

        let filter = Filter::new_must(Condition::Field(FieldCondition::new_match(
            JsonPath::new("visible"),
            true.into(),
        )));
        let stopped = Arc::new(AtomicBool::new(false));
        let mut stream = NativeDenseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            Some(filter),
            NativeDensePolicy::default(),
            17,
            stopped,
            NativeWorkerSpawner::dedicated_threads_for_tests(),
        )
        .unwrap();
        assert_eq!(stream.telemetry().worker_pull_batches, 2);
        assert_eq!(stream.telemetry().worker_points_received, 2);

        let mut actual = Vec::new();
        for _ in 0..9 {
            actual.push(stream.next_result().unwrap().unwrap());
        }
        while let Some(point) = stream.next_result().unwrap() {
            actual.push(point);
        }

        assert_eq!(
            actual
                .iter()
                .map(|point| (point.id, point.score))
                .collect::<Vec<_>>(),
            expected
        );
        let telemetry = stream.telemetry();
        assert_eq!(telemetry.native_sources, 2);
        assert_eq!(telemetry.exhaustive_fallback_sources, 0);
        assert_eq!(telemetry.points_emitted, expected.len());
    }

    #[test]
    fn shard_stream_reports_cancellation_instead_of_eof() {
        let segment_dir = tempfile::tempdir().unwrap();
        let (segment, _) = make_segment(&segment_dir, 0);
        let stopped = Arc::new(AtomicBool::new(false));
        let mut stream = NativeDenseShardStream::open(
            vec![LockedSegment::new(segment)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            None,
            NativeDensePolicy::default(),
            8,
            stopped.clone(),
            NativeWorkerSpawner::dedicated_threads_for_tests(),
        )
        .unwrap();
        stopped.store(true, AtomicOrdering::Relaxed);
        assert!(matches!(
            stream.next_result(),
            Err(OperationError::Cancelled { .. })
        ));
        stopped.store(false, AtomicOrdering::Relaxed);
        assert!(matches!(
            stream.next_result(),
            Err(OperationError::Cancelled { .. })
        ));
    }

    #[test]
    fn stale_higher_scoring_segment_copy_is_removed_before_ranking() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let mut first = build_simple_segment(first_dir.path(), 2, Distance::Dot).unwrap();
        let mut second = build_simple_segment(second_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let shared: PointIdType = 7_u64.into();
        first
            .upsert_point(
                10,
                shared,
                only_default_vector(&[100.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();
        second
            .upsert_point(
                11,
                shared,
                only_default_vector(&[1.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();
        second
            .upsert_point(
                5,
                8_u64.into(),
                only_default_vector(&[2.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();

        let mut stream = NativeDenseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            None,
            NativeDensePolicy::default(),
            8,
            Arc::new(AtomicBool::new(false)),
            NativeWorkerSpawner::dedicated_threads_for_tests(),
        )
        .unwrap();
        let ranking = std::iter::from_fn(|| stream.next_result().transpose())
            .collect::<OperationResult<Vec<_>>>()
            .unwrap();
        assert_eq!(
            ranking
                .iter()
                .map(|point| (point.id, point.version, point.score))
                .collect::<Vec<_>>(),
            vec![(8_u64.into(), 5, 2.0), (shared, 11, 1.0)]
        );
        assert_eq!(stream.telemetry().duplicates_suppressed, 1);
    }

    #[test]
    fn equal_version_copies_in_multiple_segments_fail_closed() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let mut first = build_simple_segment(first_dir.path(), 2, Distance::Dot).unwrap();
        let mut second = build_simple_segment(second_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        for segment in [&mut first, &mut second] {
            segment
                .upsert_point(
                    10,
                    7_u64.into(),
                    only_default_vector(&[1.0, 0.0]),
                    &hardware_counter,
                )
                .unwrap();
        }
        let error = NativeDenseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            None,
            NativeDensePolicy::default(),
            8,
            Arc::new(AtomicBool::new(false)),
            NativeWorkerSpawner::dedicated_threads_for_tests(),
        )
        .err()
        .expect("duplicate authoritative versions must fail");
        assert!(error.to_string().contains("in multiple Segments"));
    }
}
