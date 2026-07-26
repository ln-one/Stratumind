// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use ordered_float::OrderedFloat;
use proptest::prelude::*;

use super::*;
use crate::common::operation_error::OperationError;
use crate::types::{ExtendedPointId, ScoredPoint};

fn make_scored_point(id: u64, score: f32) -> ScoredPoint {
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

fn exhaustive_top_k(
    sources: &[Vec<ExtendedPointId>],
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
) -> Vec<ExtendedPointId> {
    let responses = sources
        .iter()
        .map(|source| {
            source
                .iter()
                .map(|id| make_scored_point(id.as_u64(), 0.0))
                .collect()
        })
        .collect();
    exact_rrf_scoring(responses, rrf_k, weights)
        .unwrap()
        .into_iter()
        .take(top_k)
        .map(|point| point.id)
        .collect()
}

fn run_dynamic(
    sources: &[Vec<ExtendedPointId>],
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
) -> (Vec<ExtendedPointId>, Vec<usize>) {
    let mut state = DynamicRrfState::new(sources.len(), top_k, rrf_k, weights).unwrap();
    let mut observed = vec![0; sources.len()];

    loop {
        for (source, points) in sources.iter().enumerate() {
            if observed[source] == points.len() && !state.exhausted[source] {
                state.finish_source(source).unwrap();
            }
        }

        if let Some(fixed) = state.fixed_top_k() {
            return (fixed, observed);
        }

        let source = (0..sources.len())
            .filter(|source| !state.exhausted[*source])
            .max_by(|left, right| {
                let left_bound = state.next_possible_contribution(*left).unwrap();
                let right_bound = state.next_possible_contribution(*right).unwrap();
                OrderedFloat(left_bound)
                    .cmp(&OrderedFloat(right_bound))
                    .then_with(|| right.cmp(left))
            })
            .expect("unfinished Dynamic RRF must have a source to advance");
        let id = sources[source][observed[source]];
        state.observe(source, id).unwrap();
        observed[source] += 1;
    }
}

fn batch_stream(ids: Vec<ExtendedPointId>) -> ExactRrfBatchStream<'static> {
    let mut next_rank = 0_usize;
    Box::new(move |max_results| {
        let start_rank = next_rank;
        let end = next_rank.saturating_add(max_results).min(ids.len());
        let point_ids = ids[next_rank..end].to_vec();
        next_rank = end;
        Ok(ChannelRankBatch {
            start_rank,
            point_ids,
            exhausted: next_rank == ids.len(),
        })
    })
}

#[test]
fn batch_observation_is_atomic_and_rejects_invalid_envelopes() {
    let mut state = DynamicRrfState::new(1, 1, 60, None).unwrap();
    let invalid = ChannelRankBatch {
        start_rank: 0,
        point_ids: vec![1.into(), 1.into()],
        exhausted: false,
    };
    assert!(state.observe_batch(0, &invalid).is_err());

    let valid = ChannelRankBatch {
        start_rank: 0,
        point_ids: vec![1.into(), 2.into()],
        exhausted: true,
    };
    state.observe_batch(0, &valid).unwrap();
    assert_eq!(state.next_positions, vec![2]);
    assert!(state.all_sources_exhausted());
    assert_eq!(state.complete_order(), vec![1.into(), 2.into()]);
}

#[test]
fn batch_partition_does_not_change_exact_wrrf_order() {
    let sources = vec![
        (0_u64..47).map(ExtendedPointId::from).collect::<Vec<_>>(),
        (0_u64..47)
            .rev()
            .map(ExtendedPointId::from)
            .collect::<Vec<_>>(),
    ];
    let expected = exhaustive_top_k(&sources, 20, 60, Some(&[0.7, 1.3]));

    for batch_size in [1, 2, 7, 16, 20, 32, 64, 127] {
        let execution = execute_dynamic_rrf_batches_with_policy(
            sources.clone().into_iter().map(batch_stream).collect(),
            batch_size,
            20,
            60,
            Some(&[0.7, 1.3]),
            DynamicRrfPolicy {
                scheduler: DynamicRrfScheduler::MaxNextContribution,
                ..DynamicRrfPolicy::default()
            },
        )
        .unwrap();
        assert_eq!(execution.point_ids, expected, "batch size {batch_size}");
    }
}

fn random_rankings(
    allow_missing_points: bool,
) -> impl Strategy<Value = (Vec<Vec<ExtendedPointId>>, Vec<f32>, usize, usize)> {
    (1usize..48, 1usize..5)
        .prop_flat_map(|(point_count, source_count)| {
            (
                Just(point_count),
                Just(source_count),
                prop::collection::vec(any::<u16>(), point_count * source_count),
                prop::collection::vec(any::<bool>(), point_count * source_count),
                prop::collection::vec(0u8..5, source_count),
                1usize..point_count + 1,
                1usize..101,
            )
        })
        .prop_map(
            move |(point_count, source_count, keys, presence, weight_classes, top_k, rrf_k)| {
                let sources = (0..source_count)
                    .map(|source| {
                        let mut ids: Vec<_> = (0..point_count as u64)
                            .filter(|id| {
                                !allow_missing_points
                                    || presence[source * point_count + *id as usize]
                            })
                            .collect();
                        ids.sort_unstable_by_key(|id| {
                            (keys[source * point_count + *id as usize], *id)
                        });
                        ids.into_iter().map(ExtendedPointId::from).collect()
                    })
                    .collect();
                let weights = weight_classes
                    .into_iter()
                    .map(|weight| match weight {
                        0 => 0.0,
                        1 => 0.5,
                        2 => 1.0,
                        3 => 2.0,
                        _ => 4.0,
                    })
                    .collect();
                (sources, weights, top_k, rrf_k)
            },
        )
}

#[test]
fn test_rrf_scoring_empty() {
    let responses = vec![];
    let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();
    assert_eq!(scored_points.len(), 0);
}

#[test]
fn test_rrf_scoring_one() {
    let responses = vec![vec![make_scored_point(1, 0.9)]];
    let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();
    assert_eq!(scored_points.len(), 1);
    assert_eq!(scored_points[0].id, 1.into());
    assert_eq!(scored_points[0].score, 0.5); // 1 / (0 + 2)
}

#[test]
fn test_rrf_scoring() {
    let responses = vec![
        vec![make_scored_point(2, 0.9), make_scored_point(1, 0.8)],
        vec![
            make_scored_point(1, 0.7),
            make_scored_point(2, 0.6),
            make_scored_point(3, 0.5),
        ],
        vec![
            make_scored_point(5, 0.9),
            make_scored_point(3, 0.5),
            make_scored_point(1, 0.4),
        ],
    ];

    // top 10
    let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();
    assert_eq!(scored_points.len(), 4);
    // assert that the list is sorted
    assert!(
        scored_points
            .array_windows()
            .all(|[a, b]| a.score >= b.score),
    );

    assert_eq!(scored_points.len(), 4);
    assert_eq!(scored_points[0].id, 1.into());
    assert_eq!(scored_points[0].score, 1.0833334);

    assert_eq!(scored_points[1].id, 2.into());
    assert_eq!(scored_points[1].score, 0.8333334);

    assert_eq!(scored_points[2].id, 3.into());
    assert_eq!(scored_points[2].score, 0.5833334);

    assert_eq!(scored_points[3].id, 5.into());
    assert_eq!(scored_points[3].score, 0.5);
}

#[test]
fn test_rrf_scoring_weighted() {
    // Two sources: first with weight 3, second with weight 1
    // This should give 3x more influence to the first source
    let responses = vec![
        vec![make_scored_point(1, 0.9), make_scored_point(2, 0.8)],
        vec![make_scored_point(2, 0.9), make_scored_point(1, 0.8)],
    ];

    // Without weights - both equal
    let scored_points = rrf_scoring(responses.clone(), DEFAULT_RRF_K, None).unwrap();
    assert_eq!(scored_points[0].score, scored_points[1].score);

    // With weights [3.0, 1.0] - first source has 3x weight
    // Higher weight means positions are "compressed" - position N with weight W
    // contributes like position N/W would with weight 1.
    let weights = [3.0, 1.0];
    let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, Some(&weights)).unwrap();

    // Point 2 scores higher because:
    // - Being at pos 1 in high-weight source (w=3) costs less (effective pos = 1/3)
    // - Being at pos 0 in low-weight source still gives full 1/k score
    // So the weighted RRF favors items that rank well across sources,
    // with higher-weight sources having their position penalties reduced.
    assert!(scored_points[0].id == 2.into());
    assert!(scored_points[0].score > scored_points[1].score);
}

#[test]
fn test_rrf_scoring_weighted_ratio() {
    // Test that weight ratio of 3:1 means position 3 in source 1 equals position 1 in source 2
    let k = 60; // Use higher k for clearer demonstration

    // Source 1: item A at position 0, item B at position 3
    // Source 2: item B at position 0, item A at position 1
    let responses = vec![
        vec![
            make_scored_point(11, 0.0),
            make_scored_point(12, 0.0),
            make_scored_point(13, 0.0),
            make_scored_point(14, 0.0),
            make_scored_point(15, 0.0),
            make_scored_point(16, 0.0),
            make_scored_point(17, 0.0),
            make_scored_point(18, 0.0),
        ],
        vec![
            make_scored_point(21, 0.0),
            make_scored_point(22, 0.0),
            make_scored_point(23, 0.0),
            make_scored_point(24, 0.0),
            make_scored_point(25, 0.0),
            make_scored_point(26, 0.0),
            make_scored_point(27, 0.0),
            make_scored_point(28, 0.0),
        ],
    ];

    let weights = [3.0, 1.0];
    let scored_points = rrf_scoring(responses, k, Some(&weights)).unwrap();

    // Check that points from the first group appear 3 times more frequently in the top ranks than points from the second group
    let top_10 = &scored_points[..10];
    let count_source_1 = top_10
        .iter()
        .filter(|p| p.id.as_u64() >= 10 && p.id.as_u64() < 20)
        .count();
    let count_source_2 = top_10
        .iter()
        .filter(|p| p.id.as_u64() >= 20 && p.id.as_u64() < 30)
        .count();

    // With a 3:1 weight ratio, we expect the count of source 1 items in the top 10 to be roughly 3 times that of source 2
    assert!(count_source_1 >= 2 * count_source_2); // Allow some variance due to tie-breaking and small sample size
}

#[test]
fn test_rrf_scoring_weights_length_mismatch() {
    let responses = vec![
        vec![make_scored_point(1, 0.9)],
        vec![make_scored_point(2, 0.9)],
    ];

    // 3 weights for 2 responses should fail
    let weights = [1.0, 2.0, 3.0];
    let result = rrf_scoring(responses.clone(), DEFAULT_RRF_K, Some(&weights));
    assert!(result.is_err());

    // 1 weight for 2 responses should fail
    let weights = [1.0];
    let result = rrf_scoring(responses, DEFAULT_RRF_K, Some(&weights));
    assert!(result.is_err());
}

#[test]
fn test_rrf_scoring_zero_weight() {
    // Test that zero weight source contributes nothing
    let responses = vec![
        vec![make_scored_point(1, 0.9)],
        vec![make_scored_point(2, 0.9)],
    ];

    let weights = [1.0, 0.0];
    let scored_points = rrf_scoring(responses, DEFAULT_RRF_K, Some(&weights)).unwrap();

    // Only point 1 should have a score, point 2 should have 0
    let p1 = scored_points.iter().find(|p| p.id == 1.into()).unwrap();
    let p2 = scored_points.iter().find(|p| p.id == 2.into()).unwrap();

    assert_eq!(p1.score, 0.5); // 1/(0+2)
    assert_eq!(p2.score, 0.0); // zero weight
}

#[test]
fn exact_rrf_identity_breaks_score_ties() {
    let responses = vec![
        vec![make_scored_point(2, 0.0)],
        vec![make_scored_point(1, 0.0)],
    ];

    let scored_points = exact_rrf_scoring(responses, DEFAULT_RRF_K, None).unwrap();

    assert_eq!(scored_points[0].id, 1.into());
    assert_eq!(scored_points[1].id, 2.into());
}

#[test]
fn exact_rrf_excludes_zero_score_identities() {
    let responses = vec![
        vec![make_scored_point(1, 0.0)],
        vec![make_scored_point(2, 0.0)],
    ];

    let scored_points = exact_rrf_scoring(responses, DEFAULT_RRF_K, Some(&[1.0, 0.0])).unwrap();

    assert_eq!(scored_points.len(), 1);
    assert_eq!(scored_points[0].id, 1.into());
}

#[test]
fn test_dynamic_rrf_stops_early_for_identical_sources() {
    let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
    let sources = vec![source.clone(), source];

    let expected = exhaustive_top_k(&sources, 3, DEFAULT_RRF_K, None);
    let (actual, observed) = run_dynamic(&sources, 3, DEFAULT_RRF_K, None);

    assert_eq!(actual, expected);
    assert_eq!(observed, vec![3, 3]);
}

#[test]
fn executor_stops_early_for_identical_sources() {
    let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
    let sources: Vec<ExactRrfStream<'_>> = vec![
        infallible_exact_rrf_stream(source.clone().into_iter()),
        infallible_exact_rrf_stream(source.clone().into_iter()),
    ];

    let execution = execute_dynamic_rrf(sources, 3, DEFAULT_RRF_K, None).unwrap();

    assert_eq!(execution.point_ids, source[..3]);
    assert_eq!(execution.source_pulls, vec![3, 3]);
    assert_eq!(execution.source_exhausted, vec![false, false]);
    assert_eq!(execution.stop_reason, DynamicRrfStopReason::TopKFixed);
}

#[test]
fn pause_target_does_not_turn_an_incomplete_prefix_into_exact_eof() {
    let sources: Vec<ExactRrfStream<'_>> = vec![
        infallible_exact_rrf_stream([ExtendedPointId::from(1)].into_iter()),
        infallible_exact_rrf_stream(
            [
                ExtendedPointId::from(2),
                ExtendedPointId::from(3),
                ExtendedPointId::from(4),
            ]
            .into_iter(),
        ),
    ];
    let mut session = DynamicRrfSession::new(
        sources,
        10,
        DEFAULT_RRF_K,
        None,
        DynamicRrfPolicy::default(),
    )
    .unwrap();

    let advance = session.advance_until_each(&[Some(1), Some(3)]).unwrap();

    assert!(matches!(advance, DynamicRrfAdvance::Paused));
    assert_eq!(session.source_pulls(), &[1, 3]);
    assert_eq!(session.source_is_exhausted(0), Some(false));
    assert_eq!(session.source_is_exhausted(1), Some(false));
}

#[test]
fn recoverable_session_extends_an_exact_prefix_without_restarting_sources() {
    let first: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
    let second: Vec<_> = (0..100).rev().map(ExtendedPointId::from).collect();
    let expected = exhaustive_top_k(&[first.clone(), second.clone()], 12, DEFAULT_RRF_K, None);
    let sources: Vec<ExactRrfStream<'_>> = vec![
        infallible_exact_rrf_stream(first.into_iter()),
        infallible_exact_rrf_stream(second.into_iter()),
    ];
    let mut session =
        DynamicRrfSession::new(sources, 1, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
            .unwrap();
    let mut previous_pulls = vec![0; 2];

    for prefix in 1..=12 {
        let execution = session.run_to_prefix(prefix).unwrap();
        assert_eq!(execution.point_ids, expected[..prefix]);
        assert!(
            execution
                .source_pulls
                .iter()
                .zip(&previous_pulls)
                .all(|(current, previous)| current >= previous),
        );
        previous_pulls = execution.source_pulls;
    }
    assert!(session.run_to_prefix(11).is_err());
}

#[test]
fn exhausted_session_reuses_one_complete_order_for_later_prefixes() {
    let first = vec![ExtendedPointId::from(1)];
    let second = vec![ExtendedPointId::from(1)];
    let expected = exhaustive_top_k(&[first.clone(), second.clone()], 3, DEFAULT_RRF_K, None);
    let sources: Vec<ExactRrfStream<'_>> = vec![
        infallible_exact_rrf_stream(first.into_iter()),
        infallible_exact_rrf_stream(second.into_iter()),
    ];
    let mut session =
        DynamicRrfSession::new(sources, 1, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
            .unwrap();

    let first = session.run_to_prefix(2).unwrap();
    assert_eq!(first.stop_reason, DynamicRrfStopReason::AllSourcesExhausted);
    let extended = session.run_to_prefix(3).unwrap();

    assert_eq!(extended.point_ids, expected);
    assert_eq!(extended.source_pulls, first.source_pulls);
    assert_eq!(extended.certification_checks, first.certification_checks);
}

#[test]
fn resumable_session_replays_prefix_before_replacing_streams() {
    let first: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
    let second: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
    let expected = exhaustive_top_k(&[first.clone(), second.clone()], 5, DEFAULT_RRF_K, None);
    let sources: Vec<ExactRrfStream<'_>> = vec![
        infallible_exact_rrf_stream(first.clone().into_iter()),
        infallible_exact_rrf_stream(second.clone().into_iter()),
    ];
    let mut session =
        DynamicRrfSession::new(sources, 5, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
            .unwrap();

    let progress = session.advance_until_each(&[Some(2), Some(2)]).unwrap();

    assert_eq!(progress, DynamicRrfAdvance::Paused);
    assert_eq!(session.source_pulls(), &[2, 2]);
    session
        .replace_source(0, infallible_exact_rrf_stream(first.into_iter()))
        .unwrap();
    session
        .replace_source(1, infallible_exact_rrf_stream(second.into_iter()))
        .unwrap();
    assert_eq!(session.run_to_completion().unwrap().point_ids, expected);
}

#[test]
fn resumable_session_rejects_replacement_prefix_mismatch() {
    let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
    let sources: Vec<ExactRrfStream<'_>> = vec![infallible_exact_rrf_stream(source.into_iter())];
    let mut session =
        DynamicRrfSession::new(sources, 5, DEFAULT_RRF_K, None, DynamicRrfPolicy::default())
            .unwrap();
    assert_eq!(
        session.advance_until_each(&[Some(2)]).unwrap(),
        DynamicRrfAdvance::Paused
    );
    let invalid: Vec<_> = [0_u64, 999, 2, 3, 4]
        .into_iter()
        .map(ExtendedPointId::from)
        .collect();

    let error = session
        .replace_source(0, infallible_exact_rrf_stream(invalid.into_iter()))
        .unwrap_err();

    assert!(error.to_string().contains("disagrees at rank 1"));
}

#[test]
fn producer_failure_is_not_treated_as_exact_exhaustion() {
    let source: ExactRrfStream<'_> = Box::new(
        vec![
            Ok(ExtendedPointId::from(1_u64)),
            Err(OperationError::cancelled("test producer stopped")),
        ]
        .into_iter(),
    );
    let mut session = DynamicRrfSession::new(
        vec![source],
        2,
        DEFAULT_RRF_K,
        None,
        DynamicRrfPolicy::default(),
    )
    .unwrap();

    let error = session.run_to_completion().unwrap_err();

    assert!(matches!(error, OperationError::Cancelled { .. }));
    assert_eq!(session.source_pulls(), &[1]);
    assert_eq!(session.source_is_exhausted(0), Some(false));
}

#[test]
fn competitor_scheduler_prefers_equal_progress_at_lower_cost() {
    let source: Vec<_> = (0..100).map(ExtendedPointId::from).collect();
    let streams: Vec<ExactRrfStream<'_>> = vec![
        infallible_exact_rrf_stream(source.clone().into_iter()),
        infallible_exact_rrf_stream(source.clone().into_iter()),
    ];
    let mut session = DynamicRrfSession::new_with_source_costs(
        streams,
        3,
        DEFAULT_RRF_K,
        None,
        &[100.0, 1.0],
        DynamicRrfPolicy::default(),
    )
    .unwrap();

    let execution = session.run_to_completion().unwrap();

    assert_eq!(execution.point_ids, source[..3]);
    assert_eq!(session.source_schedule().first(), Some(&1));
}

#[test]
fn source_costs_are_validated_at_the_exact_boundary() {
    let stream = || infallible_exact_rrf_stream(std::iter::once(ExtendedPointId::from(1_u64)));

    assert!(
        DynamicRrfSession::new_with_source_costs(
            vec![stream()],
            1,
            DEFAULT_RRF_K,
            None,
            &[],
            DynamicRrfPolicy::default(),
        )
        .is_err()
    );
    assert!(
        DynamicRrfSession::new_with_source_costs(
            vec![stream()],
            1,
            DEFAULT_RRF_K,
            None,
            &[0.0],
            DynamicRrfPolicy::default(),
        )
        .is_err()
    );
}

#[test]
fn executor_fuses_exact_rank_streams_exactly() {
    let exhaustive_sources = vec![
        (0..64).map(ExtendedPointId::from).collect::<Vec<_>>(),
        (0..64).rev().map(ExtendedPointId::from).collect::<Vec<_>>(),
    ];
    let expected = exhaustive_top_k(&exhaustive_sources, 7, DEFAULT_RRF_K, None);
    let sources: Vec<ExactRrfStream<'_>> = exhaustive_sources
        .iter()
        .map(|source| Box::new(source.iter().copied().map(Ok)) as ExactRrfStream<'_>)
        .collect();

    let actual = execute_dynamic_rrf(sources, 7, DEFAULT_RRF_K, None).unwrap();

    assert_eq!(actual.point_ids, expected);
    assert_eq!(
        actual.source_pulls.iter().sum::<usize>(),
        exhaustive_sources.iter().map(Vec::len).sum::<usize>()
    );
}

#[test]
fn test_dynamic_rrf_handles_opposite_sources() {
    let ascending: Vec<_> = (0..20).map(ExtendedPointId::from).collect();
    let descending: Vec<_> = (0..20).rev().map(ExtendedPointId::from).collect();
    let sources = vec![ascending, descending];

    let expected = exhaustive_top_k(&sources, 5, DEFAULT_RRF_K, None);
    let (actual, _) = run_dynamic(&sources, 5, DEFAULT_RRF_K, None);

    assert_eq!(actual, expected);
}

#[test]
fn test_dynamic_rrf_zero_weights_produce_no_positive_fusion_result() {
    let sources = vec![
        vec![3.into(), 2.into(), 1.into()],
        vec![2.into(), 3.into(), 1.into()],
    ];
    let weights = [0.0, 0.0];

    let expected = exhaustive_top_k(&sources, 3, DEFAULT_RRF_K, Some(&weights));
    let (actual, observed) = run_dynamic(&sources, 3, DEFAULT_RRF_K, Some(&weights));

    assert_eq!(actual, expected);
    assert!(actual.is_empty());
    assert_eq!(observed, vec![3, 3]);
}

#[test]
fn test_dynamic_rrf_rejects_duplicate_point_in_source() {
    let mut state = DynamicRrfState::new(1, 1, DEFAULT_RRF_K, None).unwrap();
    state.observe(0, 1.into()).unwrap();

    let duplicate = state.observe(0, 1.into());

    assert!(duplicate.is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1_000))]

    #[test]
    fn dynamic_rrf_matches_exhaustive_rrf(
        (sources, weights, top_k, rrf_k) in random_rankings(false)
    ) {
        let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
        let (actual, _) = run_dynamic(&sources, top_k, rrf_k, Some(&weights));

        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn dynamic_rrf_matches_exhaustive_rrf_with_partial_sources(
        (sources, weights, top_k, rrf_k) in random_rankings(true)
    ) {
        let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
        let (actual, _) = run_dynamic(&sources, top_k, rrf_k, Some(&weights));

        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn batched_dynamic_rrf_executor_matches_exhaustive_rrf(
        (sources, weights, top_k, rrf_k) in random_rankings(true)
    ) {
        let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
        let streams: Vec<ExactRrfStream<'static>> = sources
            .into_iter()
            .map(|source| infallible_exact_rrf_stream(source.into_iter()))
            .collect();
        let actual = execute_dynamic_rrf(streams, top_k, rrf_k, Some(&weights)).unwrap();

        prop_assert_eq!(actual.point_ids, expected);
    }

    #[test]
    fn cost_aware_dynamic_rrf_matches_exhaustive_for_varied_costs(
        (sources, weights, top_k, rrf_k) in random_rankings(true),
    ) {
        let expected = exhaustive_top_k(&sources, top_k, rrf_k, Some(&weights));
        let costs: Vec<_> = (0..sources.len())
            .map(|source| 1.0 + ((source * 7_919 + top_k * 97 + rrf_k * 53) % 999) as f64)
            .collect();
        let streams: Vec<ExactRrfStream<'static>> = sources
            .into_iter()
            .map(|source| infallible_exact_rrf_stream(source.into_iter()))
            .collect();
        let actual = DynamicRrfSession::new_with_source_costs(
            streams,
            top_k,
            rrf_k,
            Some(&weights),
            &costs,
            DynamicRrfPolicy::default(),
        )
        .unwrap()
        .run_to_completion()
        .unwrap();

        prop_assert_eq!(actual.point_ids, expected);
    }
}
