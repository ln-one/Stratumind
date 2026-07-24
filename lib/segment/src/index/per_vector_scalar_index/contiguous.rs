use std::time::Instant;

use super::cursor::{PerVectorScalarCursorProfile, compact_style_guard_factor};
use super::storage::validate_storage;
use super::*;

impl<TStorage: EncodedStorage> PerVectorScalarIndex<TStorage> {
    #[allow(clippy::too_many_arguments)]
    pub fn cursor_contiguous_with_refine_batch<'a>(
        &self,
        vector_storage: &'a VectorStorageEnum,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<ExactDenseCursor<'a>> {
        self.cursor_contiguous_with_refine_batch_impl(
            vector_storage,
            raw_query,
            exact_refine_batch,
            hardware_counter,
            stopped,
            None,
        )
    }

    #[cfg(feature = "stratumind-research")]
    #[allow(clippy::too_many_arguments)]
    pub fn cursor_contiguous_with_refine_batch_profiled<'a>(
        &self,
        vector_storage: &'a VectorStorageEnum,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<(ExactDenseCursor<'a>, PerVectorScalarCursorProfile)> {
        let mut profile = PerVectorScalarCursorProfile::default();
        let cursor = self.cursor_contiguous_with_refine_batch_impl(
            vector_storage,
            raw_query,
            exact_refine_batch,
            hardware_counter,
            stopped,
            Some(&mut profile),
        )?;
        Ok((cursor, profile))
    }

    #[allow(clippy::too_many_arguments)]
    fn cursor_contiguous_with_refine_batch_impl<'a>(
        &self,
        vector_storage: &'a VectorStorageEnum,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
        mut profile: Option<&mut PerVectorScalarCursorProfile>,
    ) -> OperationResult<ExactDenseCursor<'a>> {
        let total_started = profile.as_ref().map(|_| Instant::now());
        let phase_started = profile.as_ref().map(|_| Instant::now());
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
                "PerVectorScalar cursor requires a finite Query of matching dimension",
            ));
        }
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.validation_ns = started.elapsed().as_nanos();
            profile.kernel = self.encoded.selected_kernel_name();
            profile.storage_residency = self.encoded.storage_residency();
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
        let prepared_query = vector_storage
            .distance()
            .preprocess_vector::<VectorElementType>(raw_query.to_vec());
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.query_preprocess_ns = started.elapsed().as_nanos();
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
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
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.query_quantization_ns = started.elapsed().as_nanos();
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
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
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.scan_and_bound_ns = started.elapsed().as_nanos();
            profile.scanned_points = point_count;
            profile.int8_dot_products = point_count;
            profile.encoded_bytes = self
                .encoded
                .metadata()
                .row_bytes()
                .saturating_mul(point_count);
            profile.eligible_reserved_bytes = 0;
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
        let exact_scorer =
            new_raw_scorer_preprocessed(prepared_query, vector_storage, hardware_counter.fork())?;
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.exact_scorer_construction_ns = started.elapsed().as_nanos();
        }

        let cursor = if profile.is_some() {
            let (cursor, native) = ExactDenseCursor::from_contiguous_pending_batched_profiled(
                point_count,
                pending,
                move |ids, scores| exact_scorer.score_points(ids, scores),
                exact_refine_batch,
                DensePhysicalPlan::PerVectorScalarCertificate,
                point_count,
            )?;
            let profile = profile.as_deref_mut().expect("profile checked above");
            profile.eligible_validation_ns = native.eligible_validation_ns;
            profile.bound_validation_ns = native.bound_validation_ns;
            profile.pending_construction_ns = native.pending_construction_ns;
            profile.heapify_ns = native.heapify_ns;
            profile.bound_count = native.bound_count;
            profile.initial_heap_items = native.initial_heap_items;
            profile.bound_id_reserved_bytes = native.bound_id_reserved_bytes;
            profile.pending_reserved_bytes = native.pending_reserved_bytes;
            profile.total_temporary_reserved_bytes = profile
                .eligible_reserved_bytes
                .saturating_add(profile.bounds_reserved_bytes)
                .saturating_add(profile.bound_id_reserved_bytes)
                .saturating_add(profile.pending_reserved_bytes);
            cursor
        } else {
            ExactDenseCursor::from_contiguous_pending_batched(
                point_count,
                pending,
                move |ids, scores| exact_scorer.score_points(ids, scores),
                exact_refine_batch,
                DensePhysicalPlan::PerVectorScalarCertificate,
                point_count,
            )?
        };
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), total_started) {
            profile.total_cursor_ns = started.elapsed().as_nanos();
        }
        Ok(cursor)
    }
}
