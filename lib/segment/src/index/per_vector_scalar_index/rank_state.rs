use super::storage::validate_storage;
use super::*;

impl<TStorage: EncodedStorage> PerVectorScalarIndex<TStorage> {
    pub fn encoded(&self) -> &EncodedVectorsPerVectorScalar<TStorage> {
        &self.encoded
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rank_state_with_refine_batch(
        &self,
        vector_storage: &VectorStorageEnum,
        eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<DenseRankState> {
        let mut eligible = eligible;
        check_stopped(stopped)?;
        validate_storage(vector_storage, self.encoded.metadata().dimension())?;
        if vector_storage.total_vector_count() != self.encoded.metadata().vector_count() {
            return Err(OperationError::inconsistent_storage(
                "PerVectorScalar index does not match the frozen Segment vector count",
            ));
        }
        if raw_query.len() != self.encoded.metadata().dimension()
            || raw_query.iter().any(|coordinate| !coordinate.is_finite())
        {
            return Err(OperationError::validation_error(
                "PerVectorScalar rank state requires a finite Query of matching dimension",
            ));
        }
        eligible.sort_unstable();
        if eligible.windows(2).any(|pair| pair[0] == pair[1])
            || eligible
                .iter()
                .any(|id| *id as usize >= self.encoded.metadata().vector_count())
        {
            return Err(OperationError::inconsistent_storage(
                "PerVectorScalar eligible universe is invalid",
            ));
        }

        let prepared_query = vector_storage
            .distance()
            .preprocess_vector::<VectorElementType>(raw_query.to_vec());

        let encoded_query = self
            .encoded
            .try_encode_query(&prepared_query)
            .map_err(|error| {
                OperationError::validation_error(format!(
                    "failed to encode PerVectorScalar Query: {error}",
                ))
            })?;
        let fused_final_guard_factor = compact_style_guard_factor(prepared_query.len())
            .ok_or_else(|| {
                OperationError::validation_error(
                    "PerVectorScalar fused floating guard is unsupported for this dimension",
                )
            })?;

        let mut bounds = Vec::with_capacity(eligible.len());
        for chunk in eligible.chunks(SCORE_CHUNK_SIZE) {
            check_stopped(stopped)?;
            self.encoded
                .for_each_final_certified_batch(
                    &encoded_query,
                    chunk,
                    fused_final_guard_factor,
                    hardware_counter,
                    |point_id, lower, upper| bounds.push((point_id, lower, upper)),
                )
                .map_err(|error| {
                    OperationError::inconsistent_storage(format!(
                        "PerVectorScalar fused certificate failed: {error}",
                    ))
                })?;
        }

        let point_count = eligible.len();
        DenseRankState::from_certificate_bounds_batched(
            eligible,
            bounds,
            exact_refine_batch,
            DensePhysicalPlan::PerVectorScalarCertificate,
            point_count,
        )
    }

    pub fn exact_fallback(
        vector_storage: &VectorStorageEnum,
        eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<DenseRankState> {
        DenseRankState::new(
            vector_storage,
            None,
            eligible,
            raw_query,
            DenseExecutionPolicy {
                force_exact_scan: true,
                ..Default::default()
            },
            hardware_counter,
            stopped,
        )
    }
}

pub(super) fn compact_style_guard_factor(dimension: usize) -> Option<f64> {
    let operations = dimension.checked_mul(8)?.checked_add(64)? as f64;
    let numerator = mul_up(operations, f64::from(f32::EPSILON));
    if numerator >= 1.0 {
        return None;
    }
    let denominator = sub_down(1.0, numerator);
    Some(div_up(numerator, denominator))
}

#[inline]
fn sub_down(left: f64, right: f64) -> f64 {
    if right == 0.0 {
        return left;
    }
    (left - right).next_down()
}

#[inline]
fn mul_up(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        return 0.0;
    }
    (left * right).next_up()
}

#[inline]
fn div_up(left: f64, right: f64) -> f64 {
    if left == 0.0 {
        return 0.0;
    }
    (left / right).next_up()
}
