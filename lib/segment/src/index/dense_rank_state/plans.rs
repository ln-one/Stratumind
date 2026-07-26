use super::*;

pub(super) fn select_dense_plan(
    vector_storage: &VectorStorageEnum,
    quantized: Option<&QuantizedVectors>,
    eligible: &[PointOffsetType],
    policy: DenseExecutionPolicy,
) -> DensePhysicalPlan {
    let supported_distance = matches!(vector_storage.distance(), Distance::Dot | Distance::Cosine);
    let per_vector_scalar_available = !policy.force_exact_scan
        && eligible.len() >= policy.scalar_min_points
        && supported_distance
        && vector_storage.datatype() == VectorStorageDatatype::Float32
        && quantized.is_some_and(|quantized| quantized.per_vector_scalar().is_some());
    if per_vector_scalar_available {
        return DensePhysicalPlan::PerVectorScalarCertificate;
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
pub(super) fn build_per_vector_scalar_certificate_state(
    vector_storage: &VectorStorageEnum,
    quantized: &QuantizedVectors,
    eligible: Vec<PointOffsetType>,
    query: &[f32],
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
    exact_refine_batch: usize,
) -> OperationResult<DenseRankState> {
    const PRODUCTION_PVS_EXACT_REFINE_BATCH: usize = 16;
    quantized
        .per_vector_scalar()
        .ok_or_else(|| {
            OperationError::inconsistent_storage(
                "PerVectorScalar plan was selected without a persisted index",
            )
        })?
        .rank_state_with_refine_batch(
            vector_storage,
            eligible,
            query,
            exact_refine_batch.max(PRODUCTION_PVS_EXACT_REFINE_BATCH),
            hardware_counter,
            stopped,
        )
}

#[expect(clippy::too_many_arguments)]
pub(super) fn build_scalar_certificate_state(
    vector_storage: &VectorStorageEnum,
    quantized: &QuantizedVectors,
    eligible: Vec<PointOffsetType>,
    query: &[f32],
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
    exact_refine_batch: usize,
) -> OperationResult<DenseRankState> {
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
    let query_vector: QueryVector = VectorInternal::Dense(query.to_vec()).into();
    let approximate_scorer = quantized.raw_scorer(query_vector, hardware_counter.fork())?;
    let mut approximate = vec![0.0; eligible.len()];
    for (points, scores) in eligible
        .chunks(DENSE_SCORE_CHUNK_SIZE)
        .zip(approximate.chunks_mut(DENSE_SCORE_CHUNK_SIZE))
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
    let point_count = eligible.len();
    DenseRankState::from_certificate_bounds_batched(
        eligible,
        bounds,
        exact_refine_batch,
        DensePhysicalPlan::ScalarCertificate,
        point_count,
    )
}

pub(super) fn build_exact_scan_state(
    eligible: Vec<PointOffsetType>,
) -> OperationResult<DenseRankState> {
    let point_count = eligible.len();
    Ok(DenseRankState {
        inner: DenseRankStateInner::Scan(ExactScanState {
            points: None,
            next: 0,
        }),
        eligible: EligibleUniverse::explicit(eligible),
        telemetry: DenseExecutionTelemetry {
            plan: Some(DensePhysicalPlan::ExactScan),
            eligible_points: point_count,
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
