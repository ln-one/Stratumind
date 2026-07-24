use super::bounds::*;
use super::format::*;
use super::*;
use crate::encoded_storage::{TestEncodedStorage, TestEncodedStorageBuilder};

fn parameters(dimension: usize) -> VectorParameters {
    VectorParameters {
        dim: dimension,
        distance_type: DistanceType::Cosine,
        invert: false,
        deprecated_count: None,
    }
}

fn encode_test(
    vectors: &[Vec<f32>],
    metadata_path: Option<&Path>,
) -> EncodedVectorsPerVectorScalar<TestEncodedStorage> {
    let dimension = vectors.first().map_or(384, Vec::len);
    let row_bytes = HEADER_BYTES + aligned_dimension(dimension);
    EncodedVectorsPerVectorScalar::encode(
        vectors.iter(),
        TestEncodedStorageBuilder::new(None, row_bytes),
        &parameters(dimension),
        vectors.len(),
        7,
        metadata_path,
        &AtomicBool::new(false),
    )
    .unwrap()
}

#[test]
fn branch_light_finite_neighbors_match_standard_library() {
    let mut values = vec![
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        1.0,
        -1.0,
        f64::MAX,
        -f64::MAX,
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    while values.len() < 10_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let value = f64::from_bits(state);
        if value.is_finite() {
            values.push(value);
        }
    }
    for value in values {
        assert_eq!(
            next_up_finite(value).to_bits(),
            value.next_up().to_bits(),
            "next_up mismatch for {value:?}",
        );
        assert_eq!(
            next_down_finite(value).to_bits(),
            value.next_down().to_bits(),
            "next_down mismatch for {value:?}",
        );
    }
}

#[test]
fn row_size_matches_design_at_384_dimensions() {
    let vectors = vec![vec![0.0; 384]];
    let encoded = encode_test(&vectors, None);
    assert_eq!(encoded.quantized_vector_size(), 400);
    assert_eq!(encoded.metadata().source_generation(), 7);
}

#[test]
fn quantization_interval_contains_mathematical_dot() {
    for dimension in [1, 15, 16, 17, 128, 192, 256, 384, 768, 1024, 1536] {
        let vectors: Vec<Vec<f32>> = (0..5)
            .map(|row| {
                (0..dimension)
                    .map(|column| {
                        (((row * 131 + column * 37 + dimension) % 2003) as f32 - 1001.0) / 1009.0
                    })
                    .collect()
            })
            .collect();
        let query: Vec<f32> = (0..dimension)
            .map(|column| (((column * 53 + 17) % 997) as f32 - 498.0) / 503.0)
            .collect();
        let encoded = encode_test(&vectors, None);
        let encoded_query = encoded.try_encode_query(&query).unwrap();
        let ids: Vec<_> = (0..vectors.len() as PointOffsetType).collect();
        let mut bounds = Vec::new();
        encoded
            .score_certified_batch(
                &encoded_query,
                &ids,
                &mut bounds,
                &HardwareCounterCell::disposable(),
            )
            .unwrap();
        for (vector, bound) in vectors.iter().zip(bounds) {
            let exact: f64 = query
                .iter()
                .zip(vector)
                .map(|(&query, &value)| f64::from(query) * f64::from(value))
                .sum();
            assert!(
                exact >= bound.quantization_lower && exact <= bound.quantization_upper,
                "dimension={dimension} exact={exact} bounds={bound:?}",
            );
        }
    }
}

#[test]
fn deterministic_format_golden() {
    let vectors = vec![vec![0.0, 1.0, -1.0], vec![0.5, -0.25, 0.75]];
    let encoded = encode_test(&vectors, None);
    let rows: Vec<Vec<u8>> = (0..vectors.len() as PointOffsetType)
        .map(|id| encoded.storage().get_vector_data(id).into_owned())
        .collect();

    assert_eq!(encoded.metadata().magic, MAGIC);
    assert_eq!(encoded.metadata().format_version, 1);
    assert_eq!(encoded.metadata().row_bytes, 32);
    assert_eq!(encoded.metadata().actual_dimension, 16);
    assert_eq!(
        encoded.metadata().data_checksum,
        [
            22, 16, 71, 51, 45, 137, 101, 22, 255, 177, 233, 103, 229, 221, 248, 198, 43, 77, 110,
            210, 110, 124, 133, 177, 102, 123, 146, 69, 146, 179, 129, 70,
        ],
    );
    assert_eq!(
        rows,
        [
            vec![
                4, 2, 1, 60, 244, 4, 181, 63, 244, 4, 181, 63, 244, 4, 181, 49, 0, 127, 129, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                6, 131, 193, 59, 81, 119, 111, 63, 15, 154, 111, 63, 213, 113, 54, 59, 85, 214,
                127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ],
    );
}

#[test]
fn auto_and_scalar_kernels_return_identical_integer_center() {
    let vectors = vec![
        (0..384)
            .map(|index| ((index * 37 % 251) as f32 - 125.0) / 127.0)
            .collect::<Vec<_>>(),
    ];
    let query: Vec<_> = (0..384)
        .map(|index| ((index * 53 % 241) as f32 - 120.0) / 127.0)
        .collect();
    let encoded = encode_test(&vectors, None);
    let encoded_query = encoded.try_encode_query(&query).unwrap();
    let hardware_counter = HardwareCounterCell::disposable();
    let auto = encoded.score_point(&encoded_query, 0, &hardware_counter);
    let row = encoded.storage().get_vector_data(0);
    let (stats, codes) = parse_validated_row(&row, encoded.metadata()).unwrap();
    let scalar_integer_dot = simd::dot_i8_scalar(encoded_query.codes(), codes);
    let scalar_center =
        scalar_integer_dot as f64 * f64::from(encoded_query.scale()) * f64::from(stats.scale);
    assert_eq!(auto, scalar_center as f32);

    let mut auto_bounds = Vec::new();
    encoded
        .score_certified_batch(&encoded_query, &[0], &mut auto_bounds, &hardware_counter)
        .unwrap();
    assert_eq!(auto_bounds[0].center, scalar_center);
}

#[test]
fn subnormal_and_zero_vectors_are_supported() {
    let vectors = vec![
        vec![0.0; 17],
        vec![f32::from_bits(1), -f32::from_bits(1)]
            .into_iter()
            .cycle()
            .take(17)
            .collect(),
    ];
    let encoded = encode_test(&vectors, None);
    for id in 0..2 {
        let row = encoded.storage().get_vector_data(id);
        let (stats, _) = parse_row(&row, encoded.metadata()).unwrap();
        assert!(stats.scale > 0.0);
    }
}

#[test]
fn load_rejects_checksum_corruption() {
    let directory = tempfile::tempdir().unwrap();
    let data_path = directory.path().join("vectors.bin");
    let metadata_path = directory.path().join("metadata.json");
    let vectors = [vec![0.25; 17], vec![-0.75; 17]];
    let row_bytes = HEADER_BYTES + aligned_dimension(17);
    let encoded = EncodedVectorsPerVectorScalar::encode(
        vectors.iter(),
        TestEncodedStorageBuilder::new(Some(&data_path), row_bytes),
        &parameters(17),
        vectors.len(),
        11,
        Some(&metadata_path),
        &AtomicBool::new(false),
    )
    .unwrap();
    drop(encoded);

    let mut bytes = fs::read(&data_path).unwrap();
    bytes[HEADER_BYTES] ^= 1;
    fs::write(&data_path, bytes).unwrap();
    let storage = TestEncodedStorage::from_file(&data_path, row_bytes).unwrap();
    let Err(error) = EncodedVectorsPerVectorScalar::load(storage, &metadata_path) else {
        panic!("corrupt PerVectorScalar data unexpectedly loaded");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn load_rejects_stale_floating_bound_version() {
    let directory = tempfile::tempdir().unwrap();
    let data_path = directory.path().join("vectors.bin");
    let metadata_path = directory.path().join("metadata.json");
    let vectors = [vec![0.25; 17], vec![-0.75; 17]];
    let row_bytes = HEADER_BYTES + aligned_dimension(17);
    let encoded = EncodedVectorsPerVectorScalar::encode(
        vectors.iter(),
        TestEncodedStorageBuilder::new(Some(&data_path), row_bytes),
        &parameters(17),
        vectors.len(),
        11,
        Some(&metadata_path),
        &AtomicBool::new(false),
    )
    .unwrap();
    drop(encoded);

    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
    metadata["floating_bound_version"] = serde_json::Value::from(1);
    fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

    let storage = TestEncodedStorage::from_file(&data_path, row_bytes).unwrap();
    let Err(error) = EncodedVectorsPerVectorScalar::load(storage, &metadata_path) else {
        panic!("stale PerVectorScalar floating bound unexpectedly loaded");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}
