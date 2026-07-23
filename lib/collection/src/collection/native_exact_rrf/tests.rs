use std::collections::HashMap;

use common::counter::hardware_counter::HardwareCounterCell;
use ordered_float::OrderedFloat;
use segment::common::reciprocal_rank_fusion::exact_rrf_scoring;
use segment::data_types::modifier::Modifier;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
use segment::entry::{NonAppendableSegmentEntry, SegmentEntry};
use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::types::{
    Distance, Indexes, SegmentConfig, SparseVectorDataConfig, SparseVectorStorageType,
    VectorDataConfig, VectorStorageType,
};

use super::*;
use crate::common::adaptive_handle::AdaptiveSearchHandle;

const SPARSE_NAME: &str = "sparse";

fn frozen(segments: Vec<LockedSegment>) -> NativeFrozenShard {
    NativeFrozenShard {
        segments,
        point_versions_cache: Arc::new(NativeShardPointVersionsCache::default()),
    }
}

fn scored(id: PointIdType, score: f32) -> ScoredPoint {
    ScoredPoint {
        id,
        version: 0,
        score,
        payload: None,
        vector: None,
        shard_key: None,
        order_value: None,
    }
}

#[test]
fn conflicting_channel_versions_fail_closed() {
    let id: PointIdType = 7_u64.into();
    let versions = Rc::new(RefCell::new(HashMap::new()));
    let mut first = Some(ScoredPoint {
        version: 41,
        ..scored(id, 2.0)
    });
    let mut first_stream = exact_score_stream(move || Ok(first.take()), versions.clone());
    assert!(first_stream.next().unwrap().is_ok());

    let mut second = Some(ScoredPoint {
        version: 42,
        ..scored(id, 1.0)
    });
    let mut second_stream = exact_score_stream(move || Ok(second.take()), versions);
    let error = second_stream.next().unwrap().unwrap_err();
    assert!(error.to_string().contains("conflicting versions 41 and 42"));
}

#[test]
fn dropping_native_execution_arms_cancellation() {
    let stopped = Arc::new(AtomicBool::new(false));
    {
        let _cancellation = NativeExactCancellation::new(stopped.clone());
    }
    assert!(stopped.load(AtomicOrdering::Relaxed));

    stopped.store(false, AtomicOrdering::Relaxed);
    {
        let mut cancellation = NativeExactCancellation::new(stopped.clone());
        cancellation.disarm();
    }
    assert!(!stopped.load(AtomicOrdering::Relaxed));
}

#[test]
fn paired_worker_experiment_flag_is_explicit() {
    for enabled in ["1", "true", "TRUE", " yes ", "On"] {
        assert!(parse_enabled_flag(enabled));
    }
    for disabled in ["", "0", "false", "enabled", "2"] {
        assert!(!parse_enabled_flag(disabled));
    }
}

fn make_segment(
    lane: u64,
) -> (
    tempfile::TempDir,
    Segment,
    Vec<ScoredPoint>,
    Vec<ScoredPoint>,
) {
    make_segment_with_sparse_modifier(lane, None)
}

fn make_segment_with_sparse_modifier(
    lane: u64,
    sparse_modifier: Option<Modifier>,
) -> (
    tempfile::TempDir,
    Segment,
    Vec<ScoredPoint>,
    Vec<ScoredPoint>,
) {
    let directory = tempfile::tempdir().unwrap();
    let config = SegmentConfig {
        vector_data: HashMap::from([(
            DEFAULT_VECTOR_NAME.to_owned(),
            VectorDataConfig {
                size: 2,
                distance: Distance::Dot,
                storage_type: VectorStorageType::default(),
                index: Indexes::Plain {},
                quantization_config: None,
                multivector_config: None,
                datatype: None,
            },
        )]),
        sparse_vector_data: HashMap::from([(
            SPARSE_NAME.to_owned(),
            SparseVectorDataConfig {
                index: SparseIndexConfig::new(Some(1), SparseIndexType::MutableRam, None),
                storage_type: SparseVectorStorageType::Mmap,
                modifier: sparse_modifier,
            },
        )]),
        payload_storage_type: Default::default(),
    };
    let mut segment = build_segment(directory.path(), &config, None, true).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    let mut dense = Vec::new();
    let mut sparse = Vec::new();
    for index in 0..128u64 {
        let id: PointIdType = (40_000 - (index * 2 + lane)).into();
        let dense_score = 1.0 + ((index * 5 + lane) % 17) as f32;
        let sparse_score = 1.0 + ((index * 7 + lane) % 19) as f32;
        let dense_vector = [dense_score, index as f32];
        let mut vectors = only_default_vector(&dense_vector);
        vectors.insert(
            SPARSE_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![sparse_score],
            }
            .into(),
        );
        segment
            .upsert_point(index, id, vectors, &hardware_counter)
            .unwrap();
        dense.push(scored(id, dense_score));
        sparse.push(scored(id, sparse_score));
    }
    (directory, segment, dense, sparse)
}

#[test]
fn native_execution_equals_exhaustive_dense_sparse_wrrf() {
    let (first_dir, first, mut dense, mut sparse) = make_segment(0);
    let (second_dir, second, second_dense, second_sparse) = make_segment(1);
    dense.extend(second_dense);
    sparse.extend(second_sparse);
    for ranking in [&mut dense, &mut sparse] {
        ranking.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.id.cmp(&right.id))
        });
    }
    let expected = exact_rrf_scoring(vec![dense, sparse], 60, Some(&[1.0, 1.0]))
        .unwrap()
        .into_iter()
        .take(20)
        .map(|point| point.id)
        .collect::<Vec<_>>();

    let first = LockedSegment::new(first);
    let second = LockedSegment::new(second);
    let request = NativeExactRrfRequest {
        dense_query: vec![1.0, 0.0],
        dense_using: DEFAULT_VECTOR_NAME.to_owned(),
        sparse_query: SparseVector {
            indices: vec![11],
            values: vec![1.0],
        },
        sparse_using: SPARSE_NAME.to_owned(),
        filter: None,
        limit: 20,
        rrf_k: 60,
        weights: [1.0, 1.0],
        batch_size: 13,
        sparse_posting_batch_size: 4_096,
        dense_policy: NativeDensePolicy::default(),
    };
    let actual = execute_native_exact_rrf(
        vec![frozen(vec![first.clone()]), frozen(vec![second.clone()])],
        request.clone(),
        Arc::new(AtomicBool::new(false)),
        NativeWorkerSpawner::dedicated_threads_for_tests(),
    )
    .unwrap();

    let eager = execute_materialized_exact_rrf(
        vec![frozen(vec![first]), frozen(vec![second])],
        request,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert_eq!(actual.point_ids, expected);
    assert_eq!(eager.point_ids, expected);
    assert_eq!(eager.versions, actual.versions);
    assert_eq!(actual.point_ids.len(), 20);
    assert_eq!(actual.versions.len(), actual.point_ids.len());
    assert!(
        actual
            .point_ids
            .iter()
            .all(|point| actual.versions.contains_key(point))
    );
    assert!(actual.source_pulls.iter().all(|pulls| *pulls < 256));
    for channel in 0..2 {
        assert!(
            actual.source_worker_points_received[channel]
                >= actual.source_points_materialized[channel]
        );
        assert!(
            actual.source_worker_points_received[channel]
                <= actual.source_points_materialized[channel] + 2 * 13
        );
        assert!(actual.source_worker_pull_batches[channel] > 0);
    }
    drop((first_dir, second_dir));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reserved_qdrant_runtime_starts_the_complete_exact_session() {
    let (directory, segment, _, _) = make_segment(0);
    let runtime = AdaptiveSearchHandle::new_with_session_capacities(
        tokio::runtime::Handle::current(),
        tokio::runtime::Handle::current(),
        3,
        3,
    );
    let reservation = runtime
        .try_reserve_exact_workers(1)
        .expect("coordinator plus the paired Segment worker fit atomically");
    let worker_reservation = reservation.workers();
    let worker_spawner = NativeWorkerSpawner::new(move |_name, worker| {
        worker_reservation.spawn_blocking(worker).ok_or_else(|| {
            OperationError::service_error_light(
                "test exact session exceeded its reserved worker capacity",
            )
        })?;
        Ok(())
    });
    let task = reservation
        .coordinator()
        .spawn_blocking(move || {
            execute_native_exact_rrf(
                vec![frozen(vec![LockedSegment::new(segment)])],
                NativeExactRrfRequest {
                    dense_query: vec![1.0, 0.0],
                    dense_using: DEFAULT_VECTOR_NAME.to_owned(),
                    sparse_query: SparseVector {
                        indices: vec![11],
                        values: vec![1.0],
                    },
                    sparse_using: SPARSE_NAME.to_owned(),
                    filter: None,
                    limit: 20,
                    rrf_k: 60,
                    weights: [1.0, 1.0],
                    batch_size: 13,
                    sparse_posting_batch_size: 4_096,
                    dense_policy: NativeDensePolicy::default(),
                },
                Arc::new(AtomicBool::new(false)),
                worker_spawner,
            )
        })
        .expect("reserved coordinator slot");
    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("reserved session must not starve its final worker")
        .expect("coordinator task joins")
        .expect("exact execution succeeds");
    assert_eq!(result.point_ids.len(), 20);
    drop(directory);
}

#[test]
fn four_segments_use_four_paired_workers_across_qdrant_runtimes() {
    let high_cpu = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let high_io = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let runtime = AdaptiveSearchHandle::new_with_session_capacities(
        high_cpu.handle().clone(),
        high_io.handle().clone(),
        4,
        4,
    );
    assert!(
        runtime.try_reserve_exact_workers(5).is_none(),
        "five paired workers cannot be partially started across four-slot runtimes",
    );
    let reservation = runtime
        .try_reserve_exact_workers(4)
        .expect("four paired workers and their coordinator fit across both runtimes");
    let spawned = Arc::new(AtomicUsize::new(0));
    let spawned_workers = spawned.clone();
    let worker_reservation = reservation.workers();
    let worker_spawner = NativeWorkerSpawner::new(move |_name, worker| {
        spawned_workers.fetch_add(1, AtomicOrdering::Relaxed);
        worker_reservation.spawn_blocking(worker).ok_or_else(|| {
            OperationError::service_error_light(
                "paired session exceeded its four reserved worker slots",
            )
        })?;
        Ok(())
    });

    let mut directories = Vec::new();
    let mut shards = Vec::new();
    for lane in 0..4 {
        let (directory, segment, _, _) = make_segment(lane * 1_000);
        directories.push(directory);
        shards.push(frozen(vec![LockedSegment::new(segment)]));
    }
    let task = reservation
        .coordinator()
        .spawn_blocking(move || {
            execute_native_exact_rrf(
                shards,
                NativeExactRrfRequest {
                    dense_query: vec![1.0, 0.0],
                    dense_using: DEFAULT_VECTOR_NAME.to_owned(),
                    sparse_query: SparseVector {
                        indices: vec![11],
                        values: vec![1.0],
                    },
                    sparse_using: SPARSE_NAME.to_owned(),
                    filter: None,
                    limit: 20,
                    rrf_k: 60,
                    weights: [1.0, 1.0],
                    batch_size: 13,
                    sparse_posting_batch_size: 4_096,
                    dense_policy: NativeDensePolicy::default(),
                },
                Arc::new(AtomicBool::new(false)),
                worker_spawner,
            )
        })
        .expect("reserved coordinator slot");
    let result = high_cpu
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(5), task)
                .await
                .expect("paired session must not starve")
                .expect("coordinator task joins")
        })
        .expect("paired exact execution succeeds");

    assert_eq!(spawned.load(AtomicOrdering::Relaxed), 4);
    assert_eq!(result.exhaustive_fallback_sources, 0);
    assert_eq!(result.point_ids.len(), 20);
    drop(directories);
}

#[test]
fn paired_worker_keeps_dense_native_when_sparse_requires_materialization() {
    let (directory, segment, _, _) = make_segment_with_sparse_modifier(0, Some(Modifier::Idf));
    let segment = LockedSegment::new(segment);
    let request = NativeExactRrfRequest {
        dense_query: vec![1.0, 0.0],
        dense_using: DEFAULT_VECTOR_NAME.to_owned(),
        sparse_query: SparseVector {
            indices: vec![11],
            values: vec![1.0],
        },
        sparse_using: SPARSE_NAME.to_owned(),
        filter: None,
        limit: 20,
        rrf_k: 60,
        weights: [1.0, 1.0],
        batch_size: 13,
        sparse_posting_batch_size: 4_096,
        dense_policy: NativeDensePolicy::default(),
    };
    let actual = execute_native_exact_rrf(
        vec![frozen(vec![segment.clone()])],
        request.clone(),
        Arc::new(AtomicBool::new(false)),
        NativeWorkerSpawner::dedicated_threads_for_tests(),
    )
    .unwrap();
    let eager = execute_materialized_exact_rrf(
        vec![frozen(vec![segment])],
        request,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert_eq!(actual.point_ids, eager.point_ids);
    assert_eq!(actual.versions, eager.versions);
    assert_eq!(actual.exhaustive_fallback_sources, 1);
    assert!(
        actual
            .source_worker_pull_batches
            .iter()
            .all(|pulls| *pulls > 0)
    );
    drop(directory);
}

#[test]
fn overwrite_and_delete_are_resolved_before_channel_ranking() {
    let (directory, mut segment, mut dense, mut sparse) = make_segment(0);
    let hardware_counter = HardwareCounterCell::new();

    let overwritten: PointIdType = 50_001_u64.into();
    for (version, score) in [(500_u64, 100.0_f32), (501_u64, 0.25_f32)] {
        let dense_vector = [score, 0.0];
        let mut vectors = only_default_vector(&dense_vector);
        vectors.insert(
            SPARSE_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![score],
            }
            .into(),
        );
        segment
            .upsert_point(version, overwritten, vectors, &hardware_counter)
            .unwrap();
    }
    dense.push(scored(overwritten, 0.25));
    sparse.push(scored(overwritten, 0.25));

    let deleted: PointIdType = 50_002_u64.into();
    let dense_vector = [200.0, 0.0];
    let mut vectors = only_default_vector(&dense_vector);
    vectors.insert(
        SPARSE_NAME.to_owned(),
        SparseVector {
            indices: vec![11],
            values: vec![200.0],
        }
        .into(),
    );
    segment
        .upsert_point(600, deleted, vectors, &hardware_counter)
        .unwrap();
    assert!(
        segment
            .delete_point(601, deleted, &hardware_counter)
            .unwrap()
    );

    for ranking in [&mut dense, &mut sparse] {
        ranking.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.id.cmp(&right.id))
        });
    }
    let expected = exact_rrf_scoring(vec![dense, sparse], 60, Some(&[1.0, 1.0]))
        .unwrap()
        .into_iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();

    let actual = execute_native_exact_rrf(
        vec![frozen(vec![LockedSegment::new(segment)])],
        NativeExactRrfRequest {
            dense_query: vec![1.0, 0.0],
            dense_using: DEFAULT_VECTOR_NAME.to_owned(),
            sparse_query: SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            sparse_using: SPARSE_NAME.to_owned(),
            filter: None,
            limit: expected.len(),
            rrf_k: 60,
            weights: [1.0, 1.0],
            batch_size: 13,
            sparse_posting_batch_size: 4_096,
            dense_policy: NativeDensePolicy::default(),
        },
        Arc::new(AtomicBool::new(false)),
        NativeWorkerSpawner::dedicated_threads_for_tests(),
    )
    .unwrap();

    assert_eq!(actual.point_ids, expected);
    assert!(!actual.point_ids.contains(&deleted));
    assert_eq!(actual.versions.get(&overwritten), Some(&501));
    drop(directory);
}
