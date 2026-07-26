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
    Condition, FieldCondition, PointIdType, ScoredPoint, SegmentConfig, SparseVectorDataConfig,
    SparseVectorStorageType, VectorStorageDatatype,
};
use tempfile::TempDir;

use super::*;
use crate::exact_shard_stream::ExactShardStream;

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
        exact_rank_profile: Default::default(),
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

fn open_stream(
    segments: Vec<LockedSegment>,
    query: SparseVector,
    filter: Option<Filter>,
    batch_size: usize,
    stopped: Arc<AtomicBool>,
) -> OperationResult<ExactShardStream<SparseRankPlan>> {
    let plan = SparseRankPlan::new(
        &segments,
        VECTOR_NAME.to_owned(),
        query,
        filter,
        4_096,
        stopped.clone(),
    )?;
    ExactShardStream::open(segments, plan, batch_size, stopped)
}

fn next_result(
    stream: &mut ExactShardStream<SparseRankPlan>,
) -> OperationResult<Option<ScoredPoint>> {
    Ok(stream.next_batch(1)?.pop())
}

#[test]
fn shard_stream_merges_exact_segments_exactly_and_resumes() {
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
    let mut stream = open_stream(
        vec![LockedSegment::new(first), LockedSegment::new(second)],
        SparseVector {
            indices: vec![11],
            values: vec![1.0],
        },
        Some(filter),
        17,
        stopped,
    )
    .unwrap();
    assert_eq!(stream.telemetry().batch_requests, 2);
    assert_eq!(stream.telemetry().points_received, 2);

    let mut actual = Vec::new();
    for _ in 0..9 {
        actual.push(next_result(&mut stream).unwrap().unwrap());
    }
    while let Some(point) = next_result(&mut stream).unwrap() {
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
    assert_eq!(telemetry.exact_sources, 2);
    assert_eq!(telemetry.exhaustive_fallback_sources, 0);
    assert_eq!(telemetry.points_emitted, expected.len());
}

#[test]
fn shard_stream_reports_cancellation_instead_of_eof() {
    let segment_dir = tempfile::tempdir().unwrap();
    let (segment, _) = make_segment(&segment_dir, 0);
    let stopped = Arc::new(AtomicBool::new(false));
    let mut stream = open_stream(
        vec![LockedSegment::new(segment)],
        SparseVector {
            indices: vec![11],
            values: vec![1.0],
        },
        None,
        16,
        stopped.clone(),
    )
    .unwrap();
    stopped.store(true, AtomicOrdering::Relaxed);
    assert!(matches!(
        next_result(&mut stream),
        Err(OperationError::Cancelled { .. })
    ));
    stopped.store(false, AtomicOrdering::Relaxed);
    assert!(matches!(
        next_result(&mut stream),
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
        exact_rank_profile: Default::default(),
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

    let mut stream = open_stream(
        vec![LockedSegment::new(segment)],
        SparseVector {
            indices: vec![11],
            values: vec![1.0],
        },
        None,
        16,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert_eq!(stream.telemetry().exact_sources, 0);
    assert_eq!(stream.telemetry().exhaustive_fallback_sources, 1);
    assert_eq!(next_result(&mut stream).unwrap().unwrap().id, 42u64.into());
    assert!(next_result(&mut stream).unwrap().is_none());
}

#[test]
fn idf_fallback_uses_one_shard_global_query_context() {
    fn build_idf_segment(path: &TempDir, base_id: u64, matching: usize, impact: f32) -> Segment {
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
            exact_rank_profile: Default::default(),
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
    let mut stream = open_stream(
        vec![LockedSegment::new(rare), LockedSegment::new(common)],
        SparseVector {
            indices: vec![11],
            values: vec![1.0],
        },
        None,
        8,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let ranking = std::iter::from_fn(|| next_result(&mut stream).transpose())
        .collect::<OperationResult<Vec<_>>>()
        .unwrap();

    assert_eq!(ranking.len(), 11);
    assert_eq!(ranking[0].id, 200_u64.into());
    assert_eq!(ranking.last().unwrap().id, 100_u64.into());
    assert_eq!(stream.telemetry().exhaustive_fallback_sources, 2);
}
