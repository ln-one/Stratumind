use parking_lot::Mutex;

use super::*;

fn scored(id: u64, score: f32) -> ScoredPoint {
    ScoredPoint {
        id: id.into(),
        version: 0,
        score,
        payload: None,
        vector: None,
        shard_key: None,
        order_value: None,
    }
}

#[test]
fn producer_failure_is_sticky_and_never_becomes_eof() {
    let calls = Arc::new(Mutex::new(0usize));
    let source = SegmentScoreSource::on_demand(
        move |_| {
            let mut calls = calls.lock();
            *calls += 1;
            if *calls == 1 {
                Ok(BatchReply {
                    points: vec![scored(7, 1.0)],
                    eof: false,
                })
            } else {
                Err(OperationError::service_error_light(
                    "synthetic producer failure",
                ))
            }
        },
        ExactSourceMode::Exact,
    );
    let stopped = Arc::new(AtomicBool::new(false));
    let mut stream = ExactShardMergeState::open(vec![source], 0, 8, stopped, "Test").unwrap();

    let first = stream.next_result().unwrap_err();
    let second = stream.next_result().unwrap_err();
    assert!(first.to_string().contains("synthetic producer failure"));
    assert_eq!(first.to_string(), second.to_string());
}

#[test]
fn batch_merge_suppresses_equivalent_physical_copies() {
    let source = |points: Vec<ScoredPoint>| {
        let points = Arc::new(Mutex::new(VecDeque::from(points)));
        SegmentScoreSource::on_demand(
            move |limit| {
                let mut points = points.lock();
                let mut batch = Vec::with_capacity(limit);
                while batch.len() < limit
                    && let Some(point) = points.pop_front()
                {
                    batch.push(point);
                }
                Ok(BatchReply {
                    eof: points.is_empty(),
                    points: batch,
                })
            },
            ExactSourceMode::Exact,
        )
    };
    let mut duplicate = scored(7, 9.0);
    duplicate.version = 3;
    let mut first = duplicate.clone();
    let mut second = duplicate;
    first.payload = None;
    second.payload = None;

    let stopped = Arc::new(AtomicBool::new(false));
    let mut stream = ExactShardMergeState::open(
        vec![
            source(vec![first, scored(8, 7.0)]),
            source(vec![second, scored(9, 8.0)]),
        ],
        4,
        4,
        stopped,
        "Test",
    )
    .unwrap();

    let points = stream.next_batch(8).unwrap();
    assert_eq!(
        points.iter().map(|point| point.id).collect::<Vec<_>>(),
        vec![7.into(), 9.into(), 8.into()]
    );
    assert_eq!(stream.telemetry().duplicates_suppressed, 1);
}

#[test]
fn conflicting_versions_fail_before_duplicate_emission() {
    let source = |point: ScoredPoint| {
        let point = Arc::new(Mutex::new(Some(point)));
        SegmentScoreSource::on_demand(
            move |_| {
                let point = point.lock().take();
                Ok(BatchReply {
                    points: point.into_iter().collect(),
                    eof: true,
                })
            },
            ExactSourceMode::Exact,
        )
    };
    let mut old = scored(7, 9.0);
    old.version = 3;
    let mut new = old.clone();
    new.version = 4;

    let stopped = Arc::new(AtomicBool::new(false));
    let mut stream =
        ExactShardMergeState::open(vec![source(old), source(new)], 2, 8, stopped, "Test").unwrap();

    assert!(
        stream
            .next_batch(1)
            .unwrap_err()
            .to_string()
            .contains("conflicting versions")
    );
}

#[test]
fn every_output_batch_size_preserves_the_exhaustive_order() {
    fn source(points: Vec<ScoredPoint>) -> SegmentScoreSource {
        let points = Arc::new(Mutex::new(VecDeque::from(points)));
        SegmentScoreSource::on_demand(
            move |limit| {
                let mut points = points.lock();
                let mut batch = Vec::with_capacity(limit);
                while batch.len() < limit
                    && let Some(point) = points.pop_front()
                {
                    batch.push(point);
                }
                Ok(BatchReply {
                    eof: points.is_empty(),
                    points: batch,
                })
            },
            ExactSourceMode::Exact,
        )
    }

    let expected = (0..400u64).map(PointIdType::from).collect::<Vec<_>>();
    for output_batch_size in [1, 2, 7, 32, 64, 257] {
        let sources = (0..3u64)
            .map(|lane| {
                source(
                    (lane..400)
                        .step_by(3)
                        .map(|id| scored(id, 1_000.0 - id as f32))
                        .collect(),
                )
            })
            .collect();
        let mut state =
            ExactShardMergeState::open(sources, 400, 32, Arc::new(AtomicBool::new(false)), "Test")
                .unwrap();

        let mut actual = Vec::new();
        loop {
            let batch = state.next_batch(output_batch_size).unwrap();
            if batch.is_empty() {
                break;
            }
            actual.extend(batch.into_iter().map(|point| point.id));
        }

        assert_eq!(actual, expected, "output batch size {output_batch_size}");
    }
}
