use super::*;

#[test]
fn reader_independent_state_resumes_on_different_workers() {
    let state = DenseRankState::from_certificate_bounds_batched(
        vec![0, 1, 2, 3, 4],
        vec![
            (0, 5.0, 5.0),
            (1, 4.0, 4.0),
            (2, 3.0, 3.0),
            (3, 2.0, 2.0),
            (4, 1.0, 1.0),
        ],
        2,
        DensePhysicalPlan::ScalarCertificate,
        5,
    )
    .unwrap();

    let first = std::thread::spawn(move || {
        let mut state = state;
        let batch = state
            .next_batch_with(2, |ids, scores| {
                for (&id, score) in ids.iter().zip(scores) {
                    *score = 5.0 - id as f32;
                }
            })
            .unwrap();
        (state, batch)
    })
    .join()
    .unwrap();
    let second = std::thread::spawn(move || {
        let (mut state, mut points) = first;
        points.extend(
            state
                .next_batch_with(7, |ids, scores| {
                    for (&id, score) in ids.iter().zip(scores) {
                        *score = 5.0 - id as f32;
                    }
                })
                .unwrap(),
        );
        (state, points)
    })
    .join()
    .unwrap();

    assert_eq!(
        second.1.iter().map(|point| point.idx).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4],
    );
    assert_eq!(second.0.eligible_len(), 5);
}

#[test]
fn exact_scan_is_lazy_and_stably_ordered() {
    let mut state = build_exact_scan_state(vec![0, 1, 2]).unwrap();
    assert_eq!(state.telemetry().exact_scores, 0);

    let points = state
        .next_batch_with(3, |ids, scores| {
            assert_eq!(ids, [0, 1, 2]);
            scores.copy_from_slice(&[1.0, 2.0, 2.0]);
        })
        .unwrap();

    assert_eq!(
        points.iter().map(|point| point.idx).collect::<Vec<_>>(),
        vec![1, 2, 0],
    );
    assert_eq!(state.telemetry().exact_scores, 3);
}
