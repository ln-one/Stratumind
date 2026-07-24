use quantization::EncodedStorage;

use super::cursor::compact_style_guard_factor;
use super::*;
use crate::data_types::vectors::{
    DEFAULT_VECTOR_NAME, QueryVector, VectorInternal, only_default_vector,
};
use crate::entry::SegmentEntry;
use crate::segment_constructor::simple_segment_constructor::build_simple_segment;
use crate::vector_storage::new_raw_scorer;

#[test]
fn exact_cursor_matches_authoritative_cosine_order() {
    let segment_directory = tempfile::tempdir().unwrap();
    let index_directory = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_directory.path(), 17, Distance::Cosine).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..257u64 {
        let vector: Vec<_> = (0..17)
            .map(|coordinate| {
                (((id as usize * 131 + coordinate * 37) % 2003) as f32 - 1001.0) / 1009.0
            })
            .collect();
        segment
            .upsert_point(
                id,
                id.into(),
                only_default_vector(&vector),
                &hardware_counter,
            )
            .unwrap();
    }

    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let index = PerVectorScalarIndex::build_ram(
        index_directory.path(),
        &storage,
        17,
        1,
        &AtomicBool::new(false),
    )
    .unwrap();
    let query: Vec<_> = (0..17)
        .map(|coordinate| ((coordinate * 53 % 997) as f32 - 498.0) / 503.0)
        .collect();
    let eligible: Vec<_> = (0..257).collect();
    let mut cursor = index
        .cursor_with_refine_batch(
            &storage,
            eligible.clone(),
            &query,
            16,
            &hardware_counter,
            &AtomicBool::new(false),
        )
        .unwrap();
    let mut actual = Vec::new();
    while let Some(point) = cursor.next_result().unwrap() {
        actual.push(point);
    }
    let (mut contiguous, profile) = index
        .cursor_contiguous_with_refine_batch_profiled(
            &storage,
            &query,
            16,
            &hardware_counter,
            &AtomicBool::new(false),
        )
        .unwrap();
    let mut contiguous_actual = Vec::new();
    while let Some(point) = contiguous.next_result().unwrap() {
        contiguous_actual.push(point);
    }
    assert_eq!(contiguous_actual, actual);
    assert_eq!(profile.bound_id_reserved_bytes, 0);
    assert_eq!(profile.bounds_reserved_bytes, 0);
    assert_eq!(profile.pending_construction_ns, 0);
    assert!(profile.eligible_reserved_bytes <= eligible.len() * std::mem::size_of::<u32>());

    let query_vector: QueryVector = VectorInternal::Dense(query).into();
    let exact_scorer = new_raw_scorer(query_vector, &storage, hardware_counter.fork()).unwrap();
    let mut scores = vec![0.0; eligible.len()];
    exact_scorer.score_points(&eligible, &mut scores);
    let mut expected: Vec<_> = eligible
        .into_iter()
        .zip(scores)
        .map(|(idx, score)| common::types::ScoredPointOffset { idx, score })
        .collect();
    expected.sort_unstable_by(|left, right| {
        ordered_float::OrderedFloat(right.score)
            .cmp(&ordered_float::OrderedFloat(left.score))
            .then_with(|| left.idx.cmp(&right.idx))
    });

    assert_eq!(actual, expected);
    assert_eq!(
        cursor.telemetry().plan,
        Some(DensePhysicalPlan::PerVectorScalarCertificate),
    );
}

#[test]
fn storage_backends_survive_reload_and_preserve_exact_order() {
    let segment_directory = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(segment_directory.path(), 17, Distance::Cosine).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..97u64 {
        let vector: Vec<_> = (0..17)
            .map(|coordinate| {
                (((id as usize * 131 + coordinate * 37) % 2003) as f32 - 1001.0) / 1009.0
            })
            .collect();
        segment
            .upsert_point(
                id,
                id.into(),
                only_default_vector(&vector),
                &hardware_counter,
            )
            .unwrap();
    }
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let query: Vec<_> = (0..17)
        .map(|coordinate| ((coordinate * 53 % 997) as f32 - 498.0) / 503.0)
        .collect();
    let eligible: Vec<_> = (0..97).collect();
    let expected = authoritative_order(&storage, &query, &eligible, &hardware_counter);
    let stopped = AtomicBool::new(false);

    let ram_directory = tempfile::tempdir().unwrap();
    drop(
        PerVectorScalarIndex::build_ram(ram_directory.path(), &storage, 17, 41, &stopped).unwrap(),
    );
    assert!(PerVectorScalarIndex::load_ram(ram_directory.path(), 99).is_err());
    let ram = PerVectorScalarIndex::load_ram(ram_directory.path(), 41).unwrap();
    assert_eq!(ram.encoded().metadata().source_generation(), 41);
    assert_eq!(
        collect_order(
            &ram,
            &storage,
            &query,
            &eligible,
            &hardware_counter,
            &stopped,
        ),
        expected,
    );

    let mmap_directory = tempfile::tempdir().unwrap();
    drop(
        PerVectorScalarIndex::build_mmap(mmap_directory.path(), &storage, 17, 42, &stopped)
            .unwrap(),
    );
    let mmap = PerVectorScalarIndex::load_mmap(mmap_directory.path(), 42).unwrap();
    assert_eq!(mmap.encoded().metadata().source_generation(), 42);
    assert_eq!(
        collect_order(
            &mmap,
            &storage,
            &query,
            &eligible,
            &hardware_counter,
            &stopped,
        ),
        expected,
    );

    let chunked_directory = tempfile::tempdir().unwrap();
    drop(
        PerVectorScalarIndex::build_chunked_mmap(
            chunked_directory.path(),
            &storage,
            17,
            43,
            false,
            &stopped,
        )
        .unwrap(),
    );
    let chunked =
        PerVectorScalarIndex::load_chunked_mmap(chunked_directory.path(), 43, false).unwrap();
    assert_eq!(chunked.encoded().metadata().source_generation(), 43);
    assert_eq!(
        collect_order(
            &chunked,
            &storage,
            &query,
            &eligible,
            &hardware_counter,
            &stopped,
        ),
        expected,
    );
}

#[test]
fn auto_ram_path_matches_authoritative_order_across_chunk_boundary() {
    const DIMENSION: usize = 384;
    const POINTS: usize = 1_500;

    let segment_directory = tempfile::tempdir().unwrap();
    let index_directory = tempfile::tempdir().unwrap();
    let mut segment =
        build_simple_segment(segment_directory.path(), DIMENSION, Distance::Dot).unwrap();
    let hardware_counter = HardwareCounterCell::new();
    for id in 0..POINTS as u64 {
        let vector: Vec<_> = (0..DIMENSION)
            .map(|coordinate| {
                (((id as usize * 131 + coordinate * 37) % 2003) as f32 - 1001.0) / 1009.0
            })
            .collect();
        segment
            .upsert_point(
                id,
                id.into(),
                only_default_vector(&vector),
                &hardware_counter,
            )
            .unwrap();
    }
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let query: Vec<_> = (0..DIMENSION)
        .map(|coordinate| ((coordinate * 53 % 997) as f32 - 498.0) / 503.0)
        .collect();
    let eligible: Vec<_> = (0..POINTS as PointOffsetType).collect();
    let stopped = AtomicBool::new(false);
    let expected = authoritative_order(&storage, &query, &eligible, &hardware_counter);
    let index =
        PerVectorScalarIndex::build_ram(index_directory.path(), &storage, DIMENSION, 51, &stopped)
            .unwrap();

    let auto = collect_order(
        &index,
        &storage,
        &query,
        &eligible,
        &hardware_counter,
        &stopped,
    );

    assert_eq!(auto, expected);
}

#[test]
fn guard_factor_is_finite_for_target_dimensions() {
    for dimension in [1, 384, 768, 1024, 1536] {
        let guard = compact_style_guard_factor(dimension).unwrap();
        assert!(guard.is_finite() && guard > 0.0);
    }
}

#[test]
fn auto_fused_bounds_contain_authoritative_scores() {
    for dimension in [1, 17, 384, 1536] {
        let segment_directory = tempfile::tempdir().unwrap();
        let index_directory = tempfile::tempdir().unwrap();
        let mut segment =
            build_simple_segment(segment_directory.path(), dimension, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let query: Vec<_> = (0..dimension)
            .map(|coordinate| ((coordinate * 53 % 997) as f32 - 498.0) / 503.0)
            .collect();
        for id in 0..129u64 {
            let vector = match id {
                0 => vec![0.0; dimension],
                1 => query.clone(),
                2 => query.iter().map(|value| -*value).collect(),
                _ => (0..dimension)
                    .map(|coordinate| {
                        (((id as usize * 131 + coordinate * 37) % 2003) as f32 - 1001.0) / 1009.0
                    })
                    .collect(),
            };
            segment
                .upsert_point(
                    id,
                    id.into(),
                    only_default_vector(&vector),
                    &hardware_counter,
                )
                .unwrap();
        }

        let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow();
        let index = PerVectorScalarIndex::build_ram(
            index_directory.path(),
            &storage,
            dimension,
            71,
            &AtomicBool::new(false),
        )
        .unwrap();
        let encoded_query = index.encoded().try_encode_query(&query).unwrap();
        let guard_factor = compact_style_guard_factor(dimension).unwrap();
        let eligible: Vec<_> = (0..129).collect();
        let mut bounds = Vec::new();
        index
            .encoded()
            .for_each_final_certified_batch(
                &encoded_query,
                &eligible,
                guard_factor,
                &hardware_counter,
                |point_id, lower, upper| bounds.push((point_id, lower, upper)),
            )
            .unwrap();
        let mut range_bounds = Vec::new();
        index
            .encoded()
            .for_each_final_certified_range(
                &encoded_query,
                0,
                eligible.len(),
                guard_factor,
                &hardware_counter,
                |point_id, lower, upper| range_bounds.push((point_id, lower, upper)),
            )
            .unwrap();
        assert_eq!(range_bounds, bounds);

        let query_vector: QueryVector = VectorInternal::Dense(query).into();
        let exact_scorer = new_raw_scorer(query_vector, &storage, hardware_counter.fork()).unwrap();
        let mut exact = vec![0.0; eligible.len()];
        exact_scorer.score_points(&eligible, &mut exact);
        assert_eq!(bounds.len(), exact.len());
        for ((point_id, lower, upper), exact) in bounds.into_iter().zip(exact) {
            let exact = f64::from(exact);
            assert!(
                lower <= exact && exact <= upper,
                "dimension={dimension}, point={point_id}, bounds=[{lower}, {upper}], exact={exact}",
            );
        }
    }
}

fn authoritative_order(
    storage: &VectorStorageEnum,
    query: &[f32],
    eligible: &[PointOffsetType],
    hardware_counter: &HardwareCounterCell,
) -> Vec<common::types::ScoredPointOffset> {
    let query_vector: QueryVector = VectorInternal::Dense(query.to_vec()).into();
    let exact_scorer = new_raw_scorer(query_vector, storage, hardware_counter.fork()).unwrap();
    let mut scores = vec![0.0; eligible.len()];
    exact_scorer.score_points(eligible, &mut scores);
    let mut expected: Vec<_> = eligible
        .iter()
        .copied()
        .zip(scores)
        .map(|(idx, score)| common::types::ScoredPointOffset { idx, score })
        .collect();
    expected.sort_unstable_by(|left, right| {
        ordered_float::OrderedFloat(right.score)
            .cmp(&ordered_float::OrderedFloat(left.score))
            .then_with(|| left.idx.cmp(&right.idx))
    });
    expected
}

fn collect_order<TStorage: EncodedStorage>(
    index: &PerVectorScalarIndex<TStorage>,
    storage: &VectorStorageEnum,
    query: &[f32],
    eligible: &[PointOffsetType],
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
) -> Vec<common::types::ScoredPointOffset> {
    let mut cursor = index
        .cursor_with_refine_batch(
            storage,
            eligible.to_vec(),
            query,
            16,
            hardware_counter,
            stopped,
        )
        .unwrap();
    let mut actual = Vec::new();
    while let Some(point) = cursor.next_result().unwrap() {
        actual.push(point);
    }
    actual
}
