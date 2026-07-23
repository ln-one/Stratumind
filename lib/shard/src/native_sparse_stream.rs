// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Sparse stream merged across a frozen set of Segments.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use common::counter::hardware_accumulator::HwMeasurementAcc;
use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::modifier::Modifier;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::entry::ReadSegmentEntry;
use segment::types::{Filter, ScoredPoint, SearchParams, VectorNameBuf, WithPayload, WithVector};
use sparse::common::sparse_vector::SparseVector;

use crate::locked_segment::LockedSegment;
use crate::native_score_stream::{
    NativeScoreChannel, NativeShardPointVersions, NativeShardScoreStream,
    NativeShardStreamTelemetry, NativeStreamWorkerMode, NativeWorkerSpawner, SegmentScoreSource,
    WorkerCommand, WorkerCompletionSignal, serve_materialized, serve_native,
};

pub type NativeSparseShardTelemetry = NativeShardStreamTelemetry;

/// A single exact Sparse rank stream for all Segments in one Shard snapshot.
///
/// Original Segments keep their native posting cursor paused in a
/// capacity-reserved Qdrant search-runtime task. Proxy Segments, and index
/// representations without native support, are materialized exactly once as
/// an explicit safe fallback. No repeated Top-N query is issued.
pub struct NativeSparseShardStream {
    inner: NativeShardScoreStream,
}

impl NativeSparseShardStream {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: SparseVector,
        filter: Option<Filter>,
        source_batch_size: usize,
        posting_batch_size: usize,
        stopped: Arc<AtomicBool>,
        worker_spawner: NativeWorkerSpawner,
    ) -> OperationResult<Self> {
        let point_versions = NativeShardPointVersions::build(&segments, &stopped)?;
        Self::open_with_point_versions(
            segments,
            vector_name,
            query,
            filter,
            source_batch_size,
            posting_batch_size,
            stopped,
            worker_spawner,
            point_versions,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_with_point_versions(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: SparseVector,
        filter: Option<Filter>,
        source_batch_size: usize,
        posting_batch_size: usize,
        stopped: Arc<AtomicBool>,
        worker_spawner: NativeWorkerSpawner,
        point_versions: Arc<NativeShardPointVersions>,
    ) -> OperationResult<Self> {
        if source_batch_size == 0 || posting_batch_size == 0 {
            return Err(OperationError::validation_error(
                "native Sparse Shard batch sizes must be positive",
            ));
        }

        let query_context = Arc::new(build_query_context(
            &segments,
            &vector_name,
            &query,
            stopped.clone(),
        )?);
        let mut sources = Vec::with_capacity(segments.len());
        for (source, segment) in segments.into_iter().enumerate() {
            sources.push(spawn_segment_worker(
                source,
                segment,
                vector_name.clone(),
                query.clone(),
                filter.clone(),
                posting_batch_size,
                query_context.clone(),
                &worker_spawner,
            )?);
        }

        Self::from_sources(sources, point_versions, source_batch_size, stopped)
    }

    pub(crate) fn from_sources(
        sources: Vec<SegmentScoreSource>,
        point_versions: Arc<NativeShardPointVersions>,
        source_batch_size: usize,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<Self> {
        Ok(Self {
            inner: NativeShardScoreStream::open(
                sources,
                point_versions,
                source_batch_size,
                stopped,
                "Sparse",
            )?,
        })
    }

    pub fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        self.inner.next_result()
    }

    pub fn telemetry(&self) -> NativeSparseShardTelemetry {
        self.inner.telemetry()
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_segment_worker(
    source: usize,
    segment: LockedSegment,
    vector_name: VectorNameBuf,
    query: SparseVector,
    filter: Option<Filter>,
    posting_batch_size: usize,
    query_context: Arc<QueryContext>,
    worker_spawner: &NativeWorkerSpawner,
) -> OperationResult<SegmentScoreSource> {
    let (command_tx, command_rx) = sync_channel(1);
    let (ready_tx, ready_rx) = sync_channel(1);
    let (completion_tx, completion_rx) = sync_channel(1);
    worker_spawner.spawn(format!("native-sparse-segment-{source}"), move || {
        let _completion = WorkerCompletionSignal::new(completion_tx);
        run_segment_worker(
            segment,
            vector_name,
            query,
            filter,
            posting_batch_size,
            query_context,
            command_rx,
            ready_tx,
        );
    })?;

    let mode = match ready_rx.recv() {
        Ok(result) => result?,
        Err(_) => {
            let _ = completion_rx.recv();
            return Err(OperationError::service_error_light(
                "native Sparse Segment worker stopped during initialization",
            ));
        }
    };
    Ok(SegmentScoreSource::new(
        command_tx,
        completion_rx,
        mode,
        NativeScoreChannel::Sparse,
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_segment_worker(
    segment: LockedSegment,
    vector_name: VectorNameBuf,
    query: SparseVector,
    filter: Option<Filter>,
    posting_batch_size: usize,
    query_context: Arc<QueryContext>,
    commands: Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<NativeStreamWorkerMode>>,
) {
    match segment {
        LockedSegment::Original(segment) => {
            let segment = segment.read();
            let segment_query_context = query_context.get_segment_query_context();
            let mut ready = Some(ready);
            let native = segment.with_view(|view| {
                view.with_native_sparse_stream(
                    &vector_name,
                    &query,
                    filter.as_ref(),
                    posting_batch_size,
                    &segment_query_context,
                    |next| {
                        ready
                            .take()
                            .expect("native worker sends readiness once")
                            .send(Ok(NativeStreamWorkerMode::Native))
                            .map_err(|_| {
                                OperationError::cancelled(
                                    "native Sparse Shard stream closed during initialization",
                                )
                            })?;
                        serve_native(next, &commands, NativeScoreChannel::Sparse)
                    },
                )
            });

            if let Err(error) = native
                && let Some(ready) = ready
            {
                if matches!(error, OperationError::WrongSparse) {
                    run_materialized_worker(
                        &*segment,
                        &vector_name,
                        &query,
                        filter.as_ref(),
                        &query_context,
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

pub(crate) fn build_query_context(
    segments: &[LockedSegment],
    vector_name: &str,
    query: &SparseVector,
    stopped: Arc<AtomicBool>,
) -> OperationResult<QueryContext> {
    let requires_idf = segments.iter().any(|segment| {
        segment
            .get()
            .read()
            .config()
            .sparse_vector_data
            .get(vector_name)
            .is_some_and(|config| config.modifier == Some(Modifier::Idf))
    });
    let mut query_context =
        QueryContext::new(usize::MAX, HwMeasurementAcc::disposable()).with_is_stopped(stopped);
    if requires_idf {
        query_context.init_idf(vector_name, &query.indices);
    }
    for segment in segments {
        segment
            .get()
            .read()
            .fill_query_context(&mut query_context)?;
    }
    Ok(query_context)
}

#[allow(clippy::too_many_arguments)]
fn run_materialized_worker(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &SparseVector,
    filter: Option<&Filter>,
    query_context: &QueryContext,
    commands: &Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<NativeStreamWorkerMode>>,
) {
    let result = materialize_exact(segment, vector_name, query, filter, query_context);
    match result {
        Ok(points) => {
            if ready
                .send(Ok(NativeStreamWorkerMode::ExhaustiveFallback))
                .is_ok()
            {
                serve_materialized(points, commands, NativeScoreChannel::Sparse);
            }
        }
        Err(error) => {
            let _ = ready.send(Err(error));
        }
    }
}

pub(crate) fn materialize_exact(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &SparseVector,
    filter: Option<&Filter>,
    query_context: &QueryContext,
) -> OperationResult<Vec<ScoredPoint>> {
    let query_vector: QueryVector = VectorInternal::from(query.clone()).into();
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

/// Materialize one Shard's complete authoritative Sparse order without
/// keeping one blocking cursor worker per Segment. This is the exact bounded-
/// concurrency fallback when the native session cannot reserve every cursor.
pub fn materialize_shard_exact(
    segments: &[LockedSegment],
    vector_name: &str,
    query: &SparseVector,
    filter: Option<&Filter>,
    stopped: Arc<AtomicBool>,
    point_versions: &NativeShardPointVersions,
) -> OperationResult<Vec<ScoredPoint>> {
    let query_context = build_query_context(segments, vector_name, query, stopped)?;
    let mut points = Vec::new();
    for (source, segment) in segments.iter().enumerate() {
        let segment = segment.get().read();
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
            "materialized Sparse Shard order contains authoritative point {} more than once",
            duplicate.id,
        )));
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::Ordering as AtomicOrdering;

    use common::counter::hardware_counter::HardwareCounterCell;
    use ordered_float::OrderedFloat;
    use segment::data_types::named_vectors::NamedVectors;
    use segment::entry::SegmentEntry;
    use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
    use segment::json_path::JsonPath;
    use segment::payload_json;
    use segment::segment::Segment;
    use segment::segment_constructor::build_segment;
    use segment::types::{
        Condition, FieldCondition, PointIdType, SegmentConfig, SparseVectorDataConfig,
        SparseVectorStorageType, VectorStorageDatatype,
    };
    use tempfile::TempDir;

    use super::*;

    const VECTOR_NAME: &str = "sparse";

    fn make_segment(path: &TempDir, lane: u64) -> (Segment, Vec<(PointIdType, f32)>) {
        let config = SegmentConfig {
            vector_data: Default::default(),
            sparse_vector_data: HashMap::from([(
                VECTOR_NAME.to_owned(),
                SparseVectorDataConfig {
                    index: SparseIndexConfig {
                        full_scan_threshold: Some(1),
                        index_type: SparseIndexType::MutableRam,
                        datatype: Some(VectorStorageDatatype::Float32),
                    },
                    storage_type: SparseVectorStorageType::Mmap,
                    modifier: None,
                },
            )]),
            payload_storage_type: Default::default(),
        };
        let mut segment = build_segment(path.path(), &config, None, true).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let mut expected = Vec::new();
        for index in 0..256u64 {
            let id: PointIdType = (20_000 - (index * 2 + lane)).into();
            let score = 1.0 + (index % 5) as f32;
            let mut vectors = NamedVectors::default();
            vectors.insert(
                VECTOR_NAME.to_owned(),
                SparseVector {
                    indices: vec![11],
                    values: vec![score],
                }
                .into(),
            );
            segment
                .upsert_point(index, id, vectors, &hardware_counter)
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
        let mut stream = NativeSparseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            Some(filter),
            17,
            4_096,
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
        let mut stream = NativeSparseShardStream::open(
            vec![LockedSegment::new(segment)],
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            16,
            4_096,
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
    fn idf_config_uses_exact_qdrant_fallback() {
        let segment_dir = tempfile::tempdir().unwrap();
        let config = SegmentConfig {
            vector_data: Default::default(),
            sparse_vector_data: HashMap::from([(
                VECTOR_NAME.to_owned(),
                SparseVectorDataConfig {
                    index: SparseIndexConfig {
                        full_scan_threshold: Some(1),
                        index_type: SparseIndexType::MutableRam,
                        datatype: Some(VectorStorageDatatype::Float32),
                    },
                    storage_type: SparseVectorStorageType::Mmap,
                    modifier: Some(Modifier::Idf),
                },
            )]),
            payload_storage_type: Default::default(),
        };
        let mut segment = build_segment(segment_dir.path(), &config, None, true).unwrap();
        let mut vectors = NamedVectors::default();
        vectors.insert(
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![2.0],
            }
            .into(),
        );
        segment
            .upsert_point(0, 42u64.into(), vectors, &HardwareCounterCell::new())
            .unwrap();

        let mut stream = NativeSparseShardStream::open(
            vec![LockedSegment::new(segment)],
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            16,
            4_096,
            Arc::new(AtomicBool::new(false)),
            NativeWorkerSpawner::dedicated_threads_for_tests(),
        )
        .unwrap();

        assert_eq!(stream.telemetry().native_sources, 0);
        assert_eq!(stream.telemetry().exhaustive_fallback_sources, 1);
        assert_eq!(stream.next_result().unwrap().unwrap().id, 42u64.into());
        assert!(stream.next_result().unwrap().is_none());
    }

    #[test]
    fn idf_fallback_uses_one_shard_global_query_context() {
        fn build_idf_segment(
            path: &TempDir,
            base_id: u64,
            matching: usize,
            impact: f32,
        ) -> Segment {
            let config = SegmentConfig {
                vector_data: Default::default(),
                sparse_vector_data: HashMap::from([(
                    VECTOR_NAME.to_owned(),
                    SparseVectorDataConfig {
                        index: SparseIndexConfig {
                            full_scan_threshold: Some(1),
                            index_type: SparseIndexType::MutableRam,
                            datatype: Some(VectorStorageDatatype::Float32),
                        },
                        storage_type: SparseVectorStorageType::Mmap,
                        modifier: Some(Modifier::Idf),
                    },
                )]),
                payload_storage_type: Default::default(),
            };
            let mut segment = build_segment(path.path(), &config, None, true).unwrap();
            let hardware_counter = HardwareCounterCell::new();
            for index in 0..10_u64 {
                let mut vectors = NamedVectors::default();
                let matches = index < matching as u64;
                vectors.insert(
                    VECTOR_NAME.to_owned(),
                    SparseVector {
                        indices: vec![if matches { 11 } else { 99 }],
                        values: vec![if matches { impact } else { 1.0 }],
                    }
                    .into(),
                );
                segment
                    .upsert_point(index, (base_id + index).into(), vectors, &hardware_counter)
                    .unwrap();
            }
            segment
        }

        let rare_dir = tempfile::tempdir().unwrap();
        let common_dir = tempfile::tempdir().unwrap();
        let rare = build_idf_segment(&rare_dir, 100, 1, 1.0);
        let common = build_idf_segment(&common_dir, 200, 10, 2.0);
        let mut stream = NativeSparseShardStream::open(
            vec![LockedSegment::new(rare), LockedSegment::new(common)],
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            8,
            4_096,
            Arc::new(AtomicBool::new(false)),
            NativeWorkerSpawner::dedicated_threads_for_tests(),
        )
        .unwrap();
        let ranking = std::iter::from_fn(|| stream.next_result().transpose())
            .collect::<OperationResult<Vec<_>>>()
            .unwrap();

        assert_eq!(ranking.len(), 11);
        assert_eq!(ranking[0].id, 200_u64.into());
        assert_eq!(ranking.last().unwrap().id, 100_u64.into());
        assert_eq!(stream.telemetry().exhaustive_fallback_sources, 2);
    }
}
