use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use ordered_float::OrderedFloat;

use super::*;
use crate::data_types::vectors::{
    DEFAULT_VECTOR_NAME, QueryVector, VectorInternal, only_default_vector,
};
use crate::entry::SegmentEntry;
use crate::segment_constructor::simple_segment_constructor::build_simple_segment;
use crate::types::{QuantizationConfig, ScalarQuantization, ScalarQuantizationConfig, ScalarType};
use crate::vector_storage::new_raw_scorer;
use crate::vector_storage::quantized::quantized_vectors::{
    QuantizedVectors, QuantizedVectorsStorageType,
};

fn scalar_config(always_ram: bool) -> QuantizationConfig {
    QuantizationConfig::Scalar(ScalarQuantization {
        scalar: ScalarQuantizationConfig {
            r#type: ScalarType::Int8,
            quantile: None,
            always_ram: Some(always_ram),
        },
    })
}

fn populate_segment(segment: &mut crate::segment::Segment, dimension: usize, points: usize) {
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..points as u64 {
        let vector = (0..dimension)
            .map(|coordinate| {
                (((id as usize * 131 + coordinate * 37) % 2003) as f32 - 1001.0) / 1009.0
            })
            .collect::<Vec<_>>();
        segment
            .upsert_point(
                id,
                id.into(),
                only_default_vector(&vector),
                &hardware_counter,
            )
            .unwrap();
    }
}

fn authoritative_order(
    storage: &VectorStorageEnum,
    query: &[f32],
    eligible: &[PointOffsetType],
    hardware_counter: &HardwareCounterCell,
) -> Vec<ScoredPointOffset> {
    let query: QueryVector = VectorInternal::Dense(query.to_vec()).into();
    let scorer = new_raw_scorer(query, storage, hardware_counter.fork()).unwrap();
    let mut scores = vec![0.0; eligible.len()];
    scorer.score_points(eligible, &mut scores);
    let mut points = eligible
        .iter()
        .copied()
        .zip(scores)
        .map(|(idx, score)| ScoredPointOffset { idx, score })
        .collect::<Vec<_>>();
    points.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.idx.cmp(&right.idx))
    });
    points
}

#[test]
fn persisted_pvs_is_selected_by_production_auto_and_preserves_exact_order() {
    const DIMENSION: usize = 17;
    const POINTS: usize = 257;

    let segment_directory = tempfile::tempdir().unwrap();
    let quantized_directory = tempfile::tempdir().unwrap();
    let mut segment =
        build_simple_segment(segment_directory.path(), DIMENSION, Distance::Dot).unwrap();
    populate_segment(&mut segment, DIMENSION, POINTS);

    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let config = scalar_config(true);
    let stopped = AtomicBool::new(false);
    let quantized = QuantizedVectors::create(
        &storage,
        &config,
        QuantizedVectorsStorageType::Immutable,
        quantized_directory.path(),
        1,
        true,
        &stopped,
    )
    .unwrap();
    assert!(quantized.per_vector_scalar().is_some());
    assert!(quantized_directory.path().join(DATA_FILE).is_file());
    assert!(quantized_directory.path().join(METADATA_FILE).is_file());
    drop(quantized);

    let quantized = QuantizedVectors::load(
        &config,
        &storage,
        quantized_directory.path(),
        true,
        &stopped,
    )
    .unwrap()
    .unwrap();
    assert!(quantized.per_vector_scalar().is_some());

    let query = (0..DIMENSION)
        .map(|coordinate| ((coordinate * 53 % 997) as f32 - 498.0) / 503.0)
        .collect::<Vec<_>>();
    let eligible = (0..POINTS as PointOffsetType).collect::<Vec<_>>();
    let hardware_counter = HardwareCounterCell::new();
    let expected = authoritative_order(&storage, &query, &eligible, &hardware_counter);
    let mut state = DenseRankState::new(
        &storage,
        Some(&quantized),
        eligible,
        &query,
        DenseExecutionPolicy {
            scalar_min_points: 0,
            disable_compact_certificate: true,
            ..Default::default()
        },
        &hardware_counter,
        &stopped,
    )
    .unwrap();
    let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
    let scorer = new_raw_scorer(query_vector, &storage, hardware_counter.fork()).unwrap();
    let mut actual = Vec::new();
    loop {
        let batch = state
            .next_batch_with(64, |ids, scores| scorer.score_points(ids, scores))
            .unwrap();
        if batch.is_empty() {
            break;
        }
        actual.extend(batch);
    }

    assert_eq!(actual, expected);
    assert_eq!(
        state.telemetry().plan,
        Some(DensePhysicalPlan::PerVectorScalarCertificate),
    );
}

#[test]
fn invalid_pvs_is_quarantined_and_auto_falls_back_to_scalar() {
    const DIMENSION: usize = 17;
    const POINTS: usize = 129;

    let segment_directory = tempfile::tempdir().unwrap();
    let quantized_directory = tempfile::tempdir().unwrap();
    let mut segment =
        build_simple_segment(segment_directory.path(), DIMENSION, Distance::Dot).unwrap();
    populate_segment(&mut segment, DIMENSION, POINTS);

    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let config = scalar_config(true);
    let stopped = AtomicBool::new(false);
    drop(
        QuantizedVectors::create(
            &storage,
            &config,
            QuantizedVectorsStorageType::Immutable,
            quantized_directory.path(),
            1,
            true,
            &stopped,
        )
        .unwrap(),
    );
    fs_err::write(
        quantized_directory.path().join(METADATA_FILE),
        b"{not-valid-json",
    )
    .unwrap();

    let quantized = QuantizedVectors::load(
        &config,
        &storage,
        quantized_directory.path(),
        true,
        &stopped,
    )
    .unwrap()
    .unwrap();
    assert!(quantized.per_vector_scalar().is_none());

    let query = vec![0.25; DIMENSION];
    let eligible = (0..POINTS as PointOffsetType).collect::<Vec<_>>();
    let hardware_counter = HardwareCounterCell::new();
    let state = DenseRankState::new(
        &storage,
        Some(&quantized),
        eligible,
        &query,
        DenseExecutionPolicy {
            scalar_min_points: 0,
            disable_compact_certificate: true,
            ..Default::default()
        },
        &hardware_counter,
        &stopped,
    )
    .unwrap();
    assert_eq!(
        state.telemetry().plan,
        Some(DensePhysicalPlan::ScalarCertificate),
    );
}

#[test]
fn disabled_profile_neither_builds_nor_loads_pvs() {
    const DIMENSION: usize = 17;

    let segment_directory = tempfile::tempdir().unwrap();
    let quantized_directory = tempfile::tempdir().unwrap();
    let mut segment =
        build_simple_segment(segment_directory.path(), DIMENSION, Distance::Dot).unwrap();
    populate_segment(&mut segment, DIMENSION, 65);

    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let config = scalar_config(true);
    let stopped = AtomicBool::new(false);
    let quantized = QuantizedVectors::create(
        &storage,
        &config,
        QuantizedVectorsStorageType::Immutable,
        quantized_directory.path(),
        1,
        false,
        &stopped,
    )
    .unwrap();
    assert!(quantized.per_vector_scalar().is_none());
    assert!(!quantized_directory.path().join(DATA_FILE).exists());
    assert!(!quantized_directory.path().join(METADATA_FILE).exists());
    drop(quantized);

    let quantized = QuantizedVectors::load(
        &config,
        &storage,
        quantized_directory.path(),
        false,
        &stopped,
    )
    .unwrap()
    .unwrap();
    assert!(quantized.per_vector_scalar().is_none());
}

#[test]
fn pvs_mmap_backend_survives_reload() {
    const DIMENSION: usize = 17;

    let segment_directory = tempfile::tempdir().unwrap();
    let quantized_directory = tempfile::tempdir().unwrap();
    let mut segment =
        build_simple_segment(segment_directory.path(), DIMENSION, Distance::Cosine).unwrap();
    populate_segment(&mut segment, DIMENSION, 97);
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let stopped = AtomicBool::new(false);

    drop(
        PerVectorScalarMmapIndex::build_mmap(
            quantized_directory.path(),
            &storage,
            DIMENSION,
            73,
            &stopped,
        )
        .unwrap(),
    );
    let index = PerVectorScalarMmapIndex::load_mmap(quantized_directory.path(), 73).unwrap();
    assert_eq!(index.encoded().metadata().vector_count(), 97);
    assert_eq!(index.encoded().metadata().dimension(), DIMENSION);
}
