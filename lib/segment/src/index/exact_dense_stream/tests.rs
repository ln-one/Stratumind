use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;

use super::*;
use crate::data_types::query_context::QueryContext;
use crate::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use crate::entry::{ReadSegmentEntry, SegmentEntry};
use crate::segment_constructor::simple_segment_constructor::build_simple_segment;
use crate::types::{
    Distance, QuantizationConfig, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    SearchParams,
};
use crate::vector_storage::quantized::quantized_vectors::{
    QUANTIZED_COMPACT_CERTIFICATE_PATH, QuantizedVectors, QuantizedVectorsStorageType,
};

#[test]
fn reader_independent_state_resumes_on_different_workers() {
    let mut cursor = ExactDenseCursor::from_certificate_bounds_batched(
        vec![0, 1, 2, 3, 4],
        vec![
            (0, 5.0, 5.0),
            (1, 4.0, 4.0),
            (2, 3.0, 3.0),
            (3, 2.0, 2.0),
            (4, 1.0, 1.0),
        ],
        |_, _| unreachable!("detached state must not retain its original reader"),
        2,
        DensePhysicalPlan::ScalarCertificate,
        5,
    )
    .unwrap();
    let state = cursor.take_rank_state();

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
#[cfg(feature = "stratumind-research")]
fn seeded_exact_scores_are_validated_and_reused_by_rank_probe() {
    let exact_calls = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let exact_calls_for_scorer = exact_calls.clone();
    let mut cursor = ExactDenseCursor::from_certificate_bounds(
        vec![0, 1, 2],
        vec![(0, 3.0, 3.0), (1, 2.0, 2.0), (2, 1.0, 1.0)],
        move |id| {
            exact_calls_for_scorer.set(exact_calls_for_scorer.get() + 1);
            3.0 - id as f32
        },
        DensePhysicalPlan::ScalarCertificate,
        3,
    )
    .unwrap();

    assert_eq!(cursor.seed_exact_scores(&[(1, 2.0)]).unwrap(), 1);
    assert_eq!(cursor.seed_exact_scores(&[(1, 2.0)]).unwrap(), 0);
    let probes = cursor.probe_exact_ranks(&[1]).unwrap();

    assert_eq!(probes[0].rank, 1);
    assert_eq!(probes[0].score, 2.0);
    assert_eq!(exact_calls.get(), 0);
    assert_eq!(cursor.telemetry().seeded_exact_scores, 1);
    assert_eq!(cursor.telemetry().exact_scores, 0);
}

#[test]
#[cfg(feature = "stratumind-research")]
fn invalid_seed_fails_closed() {
    let mut cursor = ExactDenseCursor::from_certificate_bounds(
        vec![0],
        vec![(0, -0.5, 0.5)],
        |_| 0.0,
        DensePhysicalPlan::ScalarCertificate,
        1,
    )
    .unwrap();
    assert!(cursor.seed_exact_scores(&[(0, 0.75)]).is_err());
    assert!(cursor.seed_exact_scores(&[(1, 0.0)]).is_err());
}

#[test]
#[cfg(feature = "stratumind-research")]
fn exact_rank_probe_uses_authoritative_tie_identity() {
    let mut cursor = ExactDenseCursor::from_certificate_bounds(
        vec![0, 1, 2],
        vec![(0, 1.0, 1.0), (1, 1.0, 1.0), (2, 0.0, 0.0)],
        |id| if id < 2 { 1.0 } else { 0.0 },
        DensePhysicalPlan::ScalarCertificate,
        3,
    )
    .unwrap();
    let external_key = |internal| match internal {
        0 => 20_u64,
        1 => 10_u64,
        2 => 30_u64,
        _ => unreachable!(),
    };

    let probes = cursor
        .probe_exact_ranks_by_key(&[0, 1], external_key)
        .unwrap();

    assert_eq!(probes[0].rank, 1);
    assert_eq!(probes[1].rank, 0);
}

#[test]
fn default_compact_build_limit_keeps_scalar_fallback_without_extra_file() {
    let segment_dir = tempfile::tempdir().unwrap();
    let quantized_dir = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..=DEFAULT_DENSE_COMPACT_MAX_POINTS as u64 {
        segment
            .upsert_point(
                id,
                id.into(),
                only_default_vector(&[id as f32, 1.0]),
                &hardware_counter,
            )
            .unwrap();
    }
    let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
        scalar: ScalarQuantizationConfig {
            r#type: ScalarType::Int8,
            quantile: None,
            always_ram: Some(true),
        },
    });
    let quantized = QuantizedVectors::create(
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        &scalar_config,
        QuantizedVectorsStorageType::Immutable,
        quantized_dir.path(),
        1,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(quantized.scalar_reconstruction_table().is_some());
    assert!(quantized.compact_dense_certificate().is_none());
    assert!(
        !quantized_dir
            .path()
            .join(QUANTIZED_COMPACT_CERTIFICATE_PATH)
            .exists()
    );
}

#[test]
#[cfg(feature = "stratumind-research")]
fn exact_prefix_matches_full_scan_and_avoids_certificate_startup() {
    let segment_dir = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..8_192u64 {
        segment
            .upsert_point(
                id,
                (20_000 - id).into(),
                only_default_vector(&[id as f32, 1.0]),
                &hardware_counter,
            )
            .unwrap();
    }
    let query = vec![1.0, 0.0];
    let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
    let expected = segment
        .search(
            DEFAULT_VECTOR_NAME,
            &query_vector,
            &Default::default(),
            &Default::default(),
            None,
            20,
            Some(&SearchParams {
                exact: true,
                ..Default::default()
            }),
        )
        .unwrap()
        .into_iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();
    let mut query_context = QueryContext::default();
    segment.fill_query_context(&mut query_context).unwrap();
    let segment_query_context = query_context.get_segment_query_context();
    let (actual, telemetry) = segment
        .with_view(|view| {
            view.with_exact_dense_stream(
                DEFAULT_VECTOR_NAME,
                &query,
                None,
                64,
                DenseExecutionPolicy::default(),
                &segment_query_context,
                |next| {
                    (0..20)
                        .map(|_| next().map(Option::unwrap))
                        .collect::<OperationResult<Vec<_>>>()
                },
            )
        })
        .unwrap();
    assert_eq!(
        actual.into_iter().map(|point| point.id).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(telemetry.plan, Some(DensePhysicalPlan::ExactPrefix));
    assert_eq!(telemetry.accepted_prefix_points, 64);
    assert_eq!(telemetry.exact_prefix_fallbacks, 0);
}

#[test]
#[cfg(feature = "stratumind-research")]
fn tied_exact_prefix_falls_back_before_emitting_and_freezes_identity_order() {
    let segment_dir = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    let mut identities = Vec::new();
    for id in 0..128u64 {
        let external = (1_000 - id).into();
        identities.push(external);
        segment
            .upsert_point(
                id,
                external,
                only_default_vector(&[1.0, id as f32]),
                &hardware_counter,
            )
            .unwrap();
    }
    identities.sort_unstable();
    identities.truncate(20);
    let mut query_context = QueryContext::default();
    segment.fill_query_context(&mut query_context).unwrap();
    let segment_query_context = query_context.get_segment_query_context();
    let (actual, telemetry) = segment
        .with_view(|view| {
            view.with_exact_dense_stream(
                DEFAULT_VECTOR_NAME,
                &[1.0, 0.0],
                None,
                16,
                DenseExecutionPolicy::default(),
                &segment_query_context,
                |next| {
                    (0..20)
                        .map(|_| next().map(Option::unwrap))
                        .collect::<OperationResult<Vec<_>>>()
                },
            )
        })
        .unwrap();
    assert_eq!(
        actual.into_iter().map(|point| point.id).collect::<Vec<_>>(),
        identities
    );
    assert_eq!(telemetry.plan, Some(DensePhysicalPlan::ExactScan));
    assert_eq!(telemetry.accepted_prefix_points, 0);
    assert_eq!(telemetry.exact_prefix_fallbacks, 1);
}

#[test]
fn persisted_compact_certificate_matches_segment_exact_top_k() {
    let segment_dir = tempfile::tempdir().unwrap();
    let quantized_dir = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..8_192u64 {
        segment
            .upsert_point(
                id,
                (20_000 - id).into(),
                only_default_vector(&[id as f32, 1.0]),
                &hardware_counter,
            )
            .unwrap();
    }

    let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
        scalar: ScalarQuantizationConfig {
            r#type: ScalarType::Int8,
            quantile: None,
            always_ram: Some(true),
        },
    });
    let quantized = QuantizedVectors::create(
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        &scalar_config,
        QuantizedVectorsStorageType::Immutable,
        quantized_dir.path(),
        1,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        quantized.scalar_reconstruction_table().unwrap().len(),
        8_192
    );
    assert_eq!(quantized.compact_dense_certificate().unwrap().len(), 8_192);
    drop(quantized);
    let quantized = QuantizedVectors::load(
        &scalar_config,
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        quantized_dir.path(),
        &AtomicBool::new(false),
    )
    .unwrap()
    .unwrap();
    *segment.vector_data[DEFAULT_VECTOR_NAME]
        .quantized_vectors
        .borrow_mut() = Some(quantized);

    let query = vec![1.0, 0.0];
    let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
    let mut expected = segment
        .search(
            DEFAULT_VECTOR_NAME,
            &query_vector,
            &Default::default(),
            &Default::default(),
            None,
            20,
            Some(&SearchParams {
                exact: true,
                ..Default::default()
            }),
        )
        .unwrap();
    expected.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut query_context = QueryContext::default();
    segment.fill_query_context(&mut query_context).unwrap();
    let segment_query_context = query_context.get_segment_query_context();
    let (actual, telemetry) = segment
        .with_view(|view| {
            view.with_exact_dense_stream(
                DEFAULT_VECTOR_NAME,
                &query,
                None,
                0,
                DenseExecutionPolicy {
                    scalar_min_points: 0,
                    compact_max_points: usize::MAX,
                    disable_compact_certificate: false,
                    disable_per_vector_scalar_certificate: true,
                    ..DenseExecutionPolicy::default()
                },
                &segment_query_context,
                |next| {
                    (0..20)
                        .map(|_| next().map(Option::unwrap))
                        .collect::<OperationResult<Vec<_>>>()
                },
            )
        })
        .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(telemetry.plan, Some(DensePhysicalPlan::CompactCertificate));
    assert_eq!(telemetry.quantized_scores, 8_192);
    assert!(telemetry.exact_scores < telemetry.eligible_points);
}

#[test]
fn persisted_compact_certificate_is_exact_for_mixed_sign_dot() {
    check_mixed_sign_persisted_certificate(Distance::Dot);
}

#[test]
fn persisted_compact_certificate_is_exact_for_mixed_sign_cosine() {
    check_mixed_sign_persisted_certificate(Distance::Cosine);
}

fn check_mixed_sign_persisted_certificate(distance: Distance) {
    const DIMENSION: usize = 16;
    const POINTS: usize = 1_024;
    const QUERIES: usize = 12;
    const TOP_K: usize = 32;

    let segment_dir = tempfile::tempdir().unwrap();
    let quantized_dir = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_dir.path(), DIMENSION, distance).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..POINTS {
        let vector = (0..DIMENSION)
            .map(|coordinate| {
                let mixed = (id * 131 + coordinate * 47 + id * coordinate * 3) % 257;
                (mixed as f32 - 128.0) / 37.0 + id as f32 * 0.000_001 + coordinate as f32 * 0.000_01
            })
            .collect::<Vec<_>>();
        segment
            .upsert_point(
                id as u64,
                (50_000 - id as u64).into(),
                only_default_vector(&vector),
                &hardware_counter,
            )
            .unwrap();
    }

    let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
        scalar: ScalarQuantizationConfig {
            r#type: ScalarType::Int8,
            quantile: None,
            always_ram: Some(true),
        },
    });
    let quantized = QuantizedVectors::create(
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        &scalar_config,
        QuantizedVectorsStorageType::Immutable,
        quantized_dir.path(),
        1,
        &AtomicBool::new(false),
    )
    .unwrap();
    drop(quantized);
    let quantized = QuantizedVectors::load(
        &scalar_config,
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        quantized_dir.path(),
        &AtomicBool::new(false),
    )
    .unwrap()
    .unwrap();
    *segment.vector_data[DEFAULT_VECTOR_NAME]
        .quantized_vectors
        .borrow_mut() = Some(quantized);

    let mut query_context = QueryContext::default();
    segment.fill_query_context(&mut query_context).unwrap();
    let segment_query_context = query_context.get_segment_query_context();
    for query_id in 0..QUERIES {
        let query = (0..DIMENSION)
            .map(|coordinate| {
                let mixed = (query_id * 73 + coordinate * 29 + query_id * coordinate * 11) % 193;
                (mixed as f32 - 96.0) / 31.0 + coordinate as f32 * 0.000_03
            })
            .collect::<Vec<_>>();
        let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
        let mut expected = segment
            .search(
                DEFAULT_VECTOR_NAME,
                &query_vector,
                &Default::default(),
                &Default::default(),
                None,
                TOP_K,
                Some(&SearchParams {
                    exact: true,
                    ..Default::default()
                }),
            )
            .unwrap();
        expected.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.id.cmp(&right.id))
        });

        let (actual, telemetry) = segment
            .with_view(|view| {
                view.with_exact_dense_stream(
                    DEFAULT_VECTOR_NAME,
                    &query,
                    None,
                    0,
                    DenseExecutionPolicy {
                        scalar_min_points: 0,
                        compact_max_points: usize::MAX,
                        disable_compact_certificate: false,
                        disable_per_vector_scalar_certificate: true,
                        ..DenseExecutionPolicy::default()
                    },
                    &segment_query_context,
                    |next| {
                        (0..TOP_K)
                            .map(|_| next().map(Option::unwrap))
                            .collect::<OperationResult<Vec<_>>>()
                    },
                )
            })
            .unwrap();

        assert_eq!(actual, expected, "distance={distance:?}, query={query_id}");
        assert_eq!(telemetry.plan, Some(DensePhysicalPlan::CompactCertificate));
    }
}

#[test]
fn scalar_exact_rank_probes_share_the_exact_scan_and_exact_score_cache() {
    const DIMENSION: usize = 16;
    const POINTS: usize = 512;

    let segment_dir = tempfile::tempdir().unwrap();
    let quantized_dir = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_dir.path(), DIMENSION, Distance::Dot).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..POINTS {
        let vector = (0..DIMENSION)
            .map(|coordinate| {
                let mixed = (id * 97 + coordinate * 43 + id * coordinate * 5) % 251;
                (mixed as f32 - 125.0) / 41.0 + id as f32 * 0.000_001 + coordinate as f32 * 0.000_01
            })
            .collect::<Vec<_>>();
        segment
            .upsert_point(
                id as u64,
                (id as u64).into(),
                only_default_vector(&vector),
                &hardware_counter,
            )
            .unwrap();
    }

    let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
        scalar: ScalarQuantizationConfig {
            r#type: ScalarType::Int8,
            quantile: None,
            always_ram: Some(true),
        },
    });
    let quantized = QuantizedVectors::create(
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        &scalar_config,
        QuantizedVectorsStorageType::Immutable,
        quantized_dir.path(),
        1,
        &AtomicBool::new(false),
    )
    .unwrap();

    let query = (0..DIMENSION)
        .map(|coordinate| (coordinate as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
    let eligible = (0..POINTS as PointOffsetType).collect::<Vec<_>>();
    let mut expected = {
        let vector_storage = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow();
        let exact_scorer =
            new_raw_scorer(query_vector, &vector_storage, hardware_counter.fork()).unwrap();
        let mut exact_scores = vec![0.0; eligible.len()];
        exact_scorer.score_points(&eligible, &mut exact_scores);
        eligible
            .iter()
            .copied()
            .zip(exact_scores)
            .map(|(idx, score)| ScoredPointOffset { idx, score })
            .collect::<Vec<_>>()
    };
    expected.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.idx.cmp(&right.idx))
    });
    *segment.vector_data[DEFAULT_VECTOR_NAME]
        .quantized_vectors
        .borrow_mut() = Some(quantized);

    let target_ranks = [3usize, 29, 173];
    let target_ids = target_ranks
        .iter()
        .map(|&rank| expected[rank].idx)
        .collect::<Vec<_>>();
    let mut query_context = QueryContext::default();
    segment.fill_query_context(&mut query_context).unwrap();
    let segment_query_context = query_context.get_segment_query_context();
    let ((probes, repeated, exact_scores_after_first_probe, actual), telemetry) = segment
        .with_view(|view| {
            view.with_exact_dense_session(
                DEFAULT_VECTOR_NAME,
                &query,
                None,
                DenseExecutionPolicy {
                    scalar_min_points: 0,
                    disable_compact_certificate: true,
                    disable_per_vector_scalar_certificate: true,
                    ..DenseExecutionPolicy::default()
                },
                &segment_query_context,
                |cursor| {
                    let probes = cursor.probe_exact_ranks(&target_ids)?;
                    let exact_scores_after_first_probe = cursor.telemetry().exact_scores;
                    let repeated = cursor.probe_exact_ranks(&target_ids)?;
                    assert_eq!(
                        cursor.telemetry().exact_scores,
                        exact_scores_after_first_probe,
                        "repeating a probe batch must reuse exact scores"
                    );
                    let actual = (0..POINTS)
                        .map(|_| {
                            cursor.next_result().and_then(|point| {
                                point.ok_or_else(|| {
                                    OperationError::inconsistent_storage(
                                        "exact Dense session ended before the frozen universe",
                                    )
                                })
                            })
                        })
                        .collect::<OperationResult<Vec<_>>>()?;
                    assert_eq!(cursor.next_result()?, None);
                    Ok((probes, repeated, exact_scores_after_first_probe, actual))
                },
            )
        })
        .unwrap();
    assert_eq!(telemetry.plan, Some(DensePhysicalPlan::ScalarCertificate));
    for (probe, &rank) in probes.iter().zip(&target_ranks) {
        assert_eq!(probe.id, expected[rank].idx);
        assert_eq!(probe.score, expected[rank].score);
        assert_eq!(probe.rank, rank);
    }
    assert_eq!(repeated, probes);
    assert!(exact_scores_after_first_probe <= POINTS);
    assert_eq!(actual, expected);
}
