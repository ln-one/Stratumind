use super::*;

pub(super) fn select_dense_plan(
    vector_storage: &VectorStorageEnum,
    quantized: Option<&QuantizedVectors>,
    eligible: &[PointOffsetType],
    query: &[f32],
    policy: DenseExecutionPolicy,
) -> DensePhysicalPlan {
    let supported_distance = matches!(vector_storage.distance(), Distance::Dot | Distance::Cosine);
    let per_vector_scalar_available = !policy.force_exact_scan
        && !policy.disable_per_vector_scalar_certificate
        && eligible.len() >= policy.scalar_min_points
        && supported_distance
        && vector_storage.datatype() == VectorStorageDatatype::Float32
        && quantized.is_some_and(|quantized| quantized.per_vector_scalar().is_some());
    if per_vector_scalar_available {
        return DensePhysicalPlan::PerVectorScalarCertificate;
    }
    let compact_available = !policy.force_exact_scan
        && !policy.disable_compact_certificate
        && eligible.len() >= policy.scalar_min_points
        && eligible.len() <= policy.compact_max_points
        && supported_distance
        && vector_storage.datatype() == VectorStorageDatatype::Float32
        && quantized.is_some_and(|quantized| {
            quantized
                .compact_dense_certificate()
                .is_some_and(|compact| {
                    compact.dimension() == query.len()
                        && eligible.iter().all(|id| compact.row(*id).is_some())
                })
        });
    if compact_available {
        return DensePhysicalPlan::CompactCertificate;
    }
    let scalar_available = !policy.force_exact_scan
        && eligible.len() >= policy.scalar_min_points
        && supported_distance
        && quantized.is_some_and(|quantized| {
            matches!(quantized.distance(), Distance::Dot | Distance::Cosine)
                && quantized.datatype() == VectorStorageDatatype::Float32
                && quantized
                    .scalar_reconstruction_table()
                    .is_some_and(|table| eligible.iter().all(|id| (*id as usize) < table.len()))
        });
    if scalar_available {
        DensePhysicalPlan::ScalarCertificate
    } else {
        DensePhysicalPlan::ExactScan
    }
}

#[expect(clippy::too_many_arguments)]
pub(super) fn build_per_vector_scalar_certificate_cursor<'a>(
    vector_storage: &'a VectorStorageEnum,
    quantized: &QuantizedVectors,
    eligible: Vec<PointOffsetType>,
    query: &[f32],
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
    exact_refine_batch: usize,
) -> OperationResult<ExactDenseCursor<'a>> {
    const PRODUCTION_PVS_EXACT_REFINE_BATCH: usize = 16;
    quantized
        .per_vector_scalar()
        .ok_or_else(|| {
            OperationError::inconsistent_storage(
                "PerVectorScalar plan was selected without a persisted index",
            )
        })?
        .cursor_with_refine_batch(
            vector_storage,
            eligible,
            query,
            exact_refine_batch.max(PRODUCTION_PVS_EXACT_REFINE_BATCH),
            hardware_counter,
            stopped,
        )
}

#[expect(clippy::too_many_arguments)]
pub(super) fn build_compact_certificate_cursor<'a>(
    vector_storage: &'a VectorStorageEnum,
    quantized: &QuantizedVectors,
    eligible: Vec<PointOffsetType>,
    query: &[f32],
    query_vector: QueryVector,
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
    exact_refine_batch: usize,
) -> OperationResult<ExactDenseCursor<'a>> {
    let compact = quantized
        .compact_dense_certificate()
        .expect("compact plan availability checked");
    let processed_query = vector_storage
        .distance()
        .preprocess_vector::<VectorElementType>(query.to_vec());
    let (query_codes, query_metadata) = compact_encode(&processed_query);
    let mut bounds = Vec::with_capacity(eligible.len());
    for (ordinal, &id) in eligible.iter().enumerate() {
        if ordinal.is_multiple_of(NATIVE_DENSE_SCORE_CHUNK_SIZE) {
            check_stopped(stopped)?;
        }
        let (document_codes, document) = compact
            .row(id)
            .expect("compact certificate availability checked");
        let approximate = integer_dot(&query_codes, document_codes) as f64
            * query_metadata.scale
            * document.scale;
        let error = query_metadata.residual_norm * document.original_norm
            + query_metadata.reconstructed_norm * document.residual_norm;
        let guard = floating_guard(
            approximate.abs() + error + query_metadata.original_norm * document.original_norm,
            query.len(),
        );
        bounds.push((
            id,
            (approximate - error - guard).next_down(),
            (approximate + error + guard).next_up(),
        ));
    }
    let point_count = eligible.len();
    let exact_scorer = new_raw_scorer(query_vector, vector_storage, hardware_counter.fork())?;
    ExactDenseCursor::from_certificate_bounds_batched(
        eligible,
        bounds,
        move |ids, scores| exact_scorer.score_points(ids, scores),
        exact_refine_batch,
        DensePhysicalPlan::CompactCertificate,
        point_count,
    )
}

#[expect(clippy::too_many_arguments)]
pub(super) fn build_scalar_certificate_cursor<'a>(
    vector_storage: &'a VectorStorageEnum,
    quantized: &QuantizedVectors,
    eligible: Vec<PointOffsetType>,
    query: &[f32],
    query_vector: QueryVector,
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
    exact_refine_batch: usize,
) -> OperationResult<ExactDenseCursor<'a>> {
    let reconstruction = quantized
        .scalar_reconstruction_table()
        .expect("Scalar plan availability checked");
    let processed_query = vector_storage
        .distance()
        .preprocess_vector::<VectorElementType>(query.to_vec());
    let query_stats = quantized
        .scalar_query_reconstruction_stats(&processed_query)
        .ok_or_else(|| {
            OperationError::inconsistent_storage(
                "Scalar quantization did not expose Query reconstruction statistics",
            )
        })?;
    let approximate_scorer = quantized.raw_scorer(query_vector.clone(), hardware_counter.fork())?;
    let mut approximate = vec![0.0; eligible.len()];
    for (points, scores) in eligible
        .chunks(NATIVE_DENSE_SCORE_CHUNK_SIZE)
        .zip(approximate.chunks_mut(NATIVE_DENSE_SCORE_CHUNK_SIZE))
    {
        check_stopped(stopped)?;
        approximate_scorer.score_points(points, scores);
    }
    let bounds = eligible
        .iter()
        .zip(approximate)
        .map(|(&id, approximate)| {
            let document = reconstruction[id as usize];
            let error = query_stats.residual_norm * document.original_norm
                + query_stats.reconstructed_norm * document.residual_norm;
            let guard = floating_guard(
                f64::from(approximate).abs()
                    + error
                    + query_stats.original_norm * document.original_norm,
                query.len(),
            );
            (
                id,
                (f64::from(approximate) - error - guard).next_down(),
                (f64::from(approximate) + error + guard).next_up(),
            )
        })
        .collect();
    drop(approximate_scorer);
    let point_count = eligible.len();
    let exact_scorer = new_raw_scorer(query_vector, vector_storage, hardware_counter.fork())?;
    ExactDenseCursor::from_certificate_bounds_batched(
        eligible,
        bounds,
        move |ids, scores| exact_scorer.score_points(ids, scores),
        exact_refine_batch,
        DensePhysicalPlan::ScalarCertificate,
        point_count,
    )
}

pub(super) fn build_exact_scan_cursor<'a>(
    vector_storage: &'a VectorStorageEnum,
    eligible: Vec<PointOffsetType>,
    query_vector: QueryVector,
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
) -> OperationResult<ExactDenseCursor<'a>> {
    let scorer = new_raw_scorer(query_vector, vector_storage, hardware_counter.fork())?;
    let mut scores = vec![0.0; eligible.len()];
    for (points, scores) in eligible
        .chunks(NATIVE_DENSE_SCORE_CHUNK_SIZE)
        .zip(scores.chunks_mut(NATIVE_DENSE_SCORE_CHUNK_SIZE))
    {
        check_stopped(stopped)?;
        scorer.score_points(points, scores);
    }
    let mut points: Vec<_> = eligible
        .iter()
        .copied()
        .zip(scores)
        .map(|(idx, score)| ScoredPointOffset { idx, score })
        .collect();
    points.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.idx.cmp(&right.idx))
    });
    let point_count = points.len();
    Ok(ExactDenseCursor {
        inner: ExactDenseCursorInner::Scan(ExactScanCursor { points, next: 0 }),
        eligible: EligibleUniverse::explicit(eligible),
        telemetry: DenseExecutionTelemetry {
            plan: Some(DensePhysicalPlan::ExactScan),
            eligible_points: point_count,
            exact_scores: point_count,
            ..Default::default()
        },
    })
}

fn floating_guard(scale: f64, dimension: usize) -> f64 {
    let gamma = f64::from(f32::EPSILON) * (8.0 * dimension as f64 + 64.0);
    if gamma < 1.0 {
        scale * gamma / (1.0 - gamma)
    } else {
        f64::INFINITY
    }
}

fn compact_encode(vector: &[f32]) -> (Vec<i8>, CompactDenseVectorMetadata) {
    const MAX_CODE: f64 = 127.0;
    let original_norm = vector
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let max_abs = vector
        .iter()
        .map(|value| f64::from(*value).abs())
        .fold(0.0, f64::max);
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / MAX_CODE
    };
    let codes: Vec<_> = vector
        .iter()
        .map(|value| {
            (f64::from(*value) / scale)
                .round()
                .clamp(-MAX_CODE, MAX_CODE) as i8
        })
        .collect();
    let mut reconstructed_squared = 0.0;
    let mut residual_squared = 0.0;
    for (&value, &code) in vector.iter().zip(&codes) {
        let reconstructed = f64::from(code) * scale;
        reconstructed_squared += reconstructed * reconstructed;
        let residual = f64::from(value) - reconstructed;
        residual_squared += residual * residual;
    }
    (
        codes,
        CompactDenseVectorMetadata {
            scale,
            reconstructed_norm: reconstructed_squared.sqrt(),
            residual_norm: residual_squared.sqrt().next_up(),
            original_norm,
        },
    )
}

fn integer_dot(left: &[i8], right: &[i8]) -> i64 {
    debug_assert_eq!(left.len(), right.len());
    const MAX_I32_DOT_DIMENSION: usize = i32::MAX as usize / (127 * 127);
    if left.len() <= MAX_I32_DOT_DIMENSION {
        i64::from(
            left.iter()
                .zip(right)
                .map(|(&left, &right)| i32::from(left) * i32::from(right))
                .sum::<i32>(),
        )
    } else {
        left.iter()
            .zip(right)
            .map(|(&left, &right)| i64::from(left) * i64::from(right))
            .sum()
    }
}
