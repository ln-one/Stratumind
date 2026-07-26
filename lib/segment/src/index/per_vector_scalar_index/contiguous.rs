use super::rank_state::compact_style_guard_factor;
use super::storage::validate_storage;
use super::*;

impl<TStorage: EncodedStorage> PerVectorScalarIndex<TStorage> {
    #[allow(clippy::too_many_arguments)]
    pub fn rank_state_contiguous_with_refine_batch(
        &self,
        vector_storage: &VectorStorageEnum,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<DenseRankState> {
        check_stopped(stopped)?;
        validate_storage(vector_storage, self.encoded.metadata().dimension())?;
        let point_count = self.encoded.metadata().vector_count();
        if vector_storage.total_vector_count() != point_count {
            return Err(OperationError::inconsistent_storage(
                "PerVectorScalar index does not match the frozen Segment vector count",
            ));
        }
        if point_count > PointOffsetType::MAX as usize + 1 {
            return Err(OperationError::inconsistent_storage(
                "PerVectorScalar contiguous universe exceeds PointOffsetType",
            ));
        }
        if raw_query.len() != self.encoded.metadata().dimension()
            || raw_query.iter().any(|coordinate| !coordinate.is_finite())
        {
            return Err(OperationError::validation_error(
                "PerVectorScalar rank state requires a finite Query of matching dimension",
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

        let mut pending = Vec::with_capacity(point_count);
        for start in (0..point_count).step_by(SCORE_CHUNK_SIZE) {
            check_stopped(stopped)?;
            let end = (start + SCORE_CHUNK_SIZE).min(point_count);
            let start = PointOffsetType::try_from(start).expect("validated point count");
            self.encoded
                .for_each_final_certified_range(
                    &encoded_query,
                    start,
                    end - start as usize,
                    fused_final_guard_factor,
                    hardware_counter,
                    |point_id, lower, upper| {
                        pending.push(PendingBound {
                            id: point_id,
                            lower,
                            value: upper,
                        });
                    },
                )
                .map_err(|error| {
                    OperationError::inconsistent_storage(format!(
                        "PerVectorScalar fused certificate failed: {error}",
                    ))
                })?;
        }

        DenseRankState::from_contiguous_pending_batched(
            point_count,
            pending,
            exact_refine_batch,
            DensePhysicalPlan::PerVectorScalarCertificate,
            point_count,
        )
    }
}
