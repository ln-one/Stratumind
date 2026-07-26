use std::collections::HashMap;
use std::time::Duration;

use common::counter::hardware_counter::HardwareCounterCell;
use ordered_float::OrderedFloat;
use segment::common::reciprocal_rank_fusion::exact_rrf_scoring;
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
    let mut first_stream = exact_rank_stream(move || Ok(first.take()), versions.clone());
    assert!(first_stream.next().unwrap().is_ok());

    let mut second = Some(ScoredPoint {
        version: 42,
        ..scored(id, 1.0)
    });
    let mut second_stream = exact_rank_stream(move || Ok(second.take()), versions);
    let error = second_stream.next().unwrap().unwrap_err();
    assert!(error.to_string().contains("conflicting versions 41 and 42"));
}

#[test]
fn dropping_exact_execution_arms_cancellation() {
    let stopped = Arc::new(AtomicBool::new(false));
    {
        let _cancellation = ExactRrfCancellation::new(stopped.clone());
    }
    assert!(stopped.load(AtomicOrdering::Relaxed));

    stopped.store(false, AtomicOrdering::Relaxed);
    {
        let mut cancellation = ExactRrfCancellation::new(stopped.clone());
        cancellation.disarm();
    }
    assert!(!stopped.load(AtomicOrdering::Relaxed));
}

fn make_segment(
    lane: u64,
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
                modifier: None,
            },
        )]),
        payload_storage_type: Default::default(),
        exact_rank_profile: Default::default(),
    };
    let mut segment = build_segment(directory.path(), &config, None, true).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    let mut dense = Vec::new();
    let mut sparse = Vec::new();
    for index in 0..128u64 {
        let id: PointIdType = (40_000 - (index * 8 + lane)).into();
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
fn exact_execution_equals_exhaustive_dense_sparse_wrrf() {
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

    let actual = ExactHybridSession::new(
        vec![
            vec![LockedSegment::new(first)],
            vec![LockedSegment::new(second)],
        ],
        ExactRrfRequest {
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
            dense_policy: DenseExecutionPolicy::default(),
        },
        Arc::new(AtomicBool::new(false)),
    )
    .execute()
    .unwrap();

    assert_eq!(actual.point_ids, expected);
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
            actual.source_points_received[channel] >= actual.source_points_materialized[channel]
        );
        assert!(
            actual.source_points_received[channel]
                <= actual.source_points_materialized[channel] + 2 * 13
        );
        assert!(actual.source_batch_requests[channel] > 0);
    }
    drop((first_dir, second_dir));
}

#[test]
fn reserved_qdrant_runtimes_serve_many_segments_with_one_reusable_reader_slot() {
    let mut directories = Vec::new();
    let mut segments = Vec::new();
    for lane in 0..8 {
        let (directory, segment, _, _) = make_segment(lane);
        directories.push(directory);
        segments.push(LockedSegment::new(segment));
    }
    let reader_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let coordinator_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let runtime = AdaptiveSearchHandle::new_with_session_capacities(
        reader_runtime.handle().clone(),
        coordinator_runtime.handle().clone(),
        2,
        2,
    );
    let reservation = runtime
        .try_reserve_exact_session(1)
        .expect("one reusable reader slot and a separate coordinator fit");
    let task = reservation
        .coordinator()
        .spawn_blocking(move || {
            ExactHybridSession::new(
                vec![segments],
                ExactRrfRequest {
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
                    dense_policy: DenseExecutionPolicy::default(),
                },
                Arc::new(AtomicBool::new(false)),
            )
            .execute()
        })
        .expect("reserved coordinator slot");
    let result = reader_runtime
        .block_on(async { tokio::time::timeout(Duration::from_secs(10), task).await })
        .expect("reserved session must not starve its reader")
        .expect("coordinator task joins")
        .expect("exact execution succeeds");
    assert_eq!(result.point_ids.len(), 20);
    assert_eq!(result.exhaustive_fallback_sources, 0);
    drop(directories);
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

    let actual = ExactHybridSession::new(
        vec![vec![LockedSegment::new(segment)]],
        ExactRrfRequest {
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
            dense_policy: DenseExecutionPolicy::default(),
        },
        Arc::new(AtomicBool::new(false)),
    )
    .execute()
    .unwrap();

    assert_eq!(actual.point_ids, expected);
    assert!(!actual.point_ids.contains(&deleted));
    assert_eq!(actual.versions.get(&overwritten), Some(&501));
    drop(directory);
}
