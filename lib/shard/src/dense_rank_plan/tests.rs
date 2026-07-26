use std::sync::atomic::Ordering as AtomicOrdering;

use common::counter::hardware_counter::HardwareCounterCell;
use ordered_float::OrderedFloat;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
use segment::entry::SegmentEntry;
use segment::json_path::JsonPath;
use segment::payload_json;
use segment::segment::Segment;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Condition, Distance, FieldCondition, PointIdType, ScoredPoint};
use tempfile::TempDir;

use super::*;
use crate::exact_shard_stream::ExactShardStream;

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

fn open_stream(
    segments: Vec<LockedSegment>,
    query: Vec<f32>,
    filter: Option<Filter>,
    batch_size: usize,
    stopped: Arc<AtomicBool>,
) -> OperationResult<ExactShardStream<DenseRankPlan>> {
    ExactShardStream::open(
        segments,
        DenseRankPlan::new(
            DEFAULT_VECTOR_NAME.to_owned(),
            query,
            filter,
            DenseExecutionPolicy::default(),
        ),
        batch_size,
        stopped,
    )
}

fn next_result(
    stream: &mut ExactShardStream<DenseRankPlan>,
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
    let first = LockedSegment::new(first);
    let LockedSegment::Original(first_handle) = first.clone() else {
        unreachable!()
    };
    let mut stream = open_stream(
        vec![first, LockedSegment::new(second)],
        vec![1.0, 0.0],
        Some(filter),
        17,
        stopped,
    )
    .unwrap();
    assert!(
        first_handle.try_write().is_some(),
        "session initialization must release the Segment read guard"
    );
    assert_eq!(stream.telemetry().batch_requests, 2);
    assert_eq!(stream.telemetry().points_received, 2);

    let mut actual = Vec::new();
    for _ in 0..9 {
        actual.push(next_result(&mut stream).unwrap().unwrap());
        assert!(
            first_handle.try_write().is_some(),
            "each completed pull must release the Segment read guard"
        );
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
        vec![1.0, 0.0],
        None,
        8,
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
fn conflicting_version_copies_fail_closed_when_encountered() {
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

    let mut stream = open_stream(
        vec![LockedSegment::new(first), LockedSegment::new(second)],
        vec![1.0, 0.0],
        None,
        8,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let error = std::iter::from_fn(|| next_result(&mut stream).transpose())
        .collect::<OperationResult<Vec<_>>>()
        .unwrap_err();
    assert!(error.to_string().contains("emitted point 7 more than once"));
}

#[test]
fn equivalent_physical_copies_are_emitted_once() {
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
    let mut stream = open_stream(
        vec![LockedSegment::new(first), LockedSegment::new(second)],
        vec![1.0, 0.0],
        None,
        8,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let point = next_result(&mut stream).unwrap().unwrap();
    assert_eq!(
        (point.id, point.version, point.score),
        (7_u64.into(), 10, 1.0)
    );
    assert!(next_result(&mut stream).unwrap().is_none());
    assert_eq!(stream.telemetry().duplicates_suppressed, 1);
}
