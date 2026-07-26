use std::time::Instant;

use super::storage::validate_storage;
use super::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PerVectorScalarBuildProfile {
    pub total_build_ns: u128,
    pub validation_ns: u128,
    pub query_preprocess_ns: u128,
    pub query_quantization_ns: u128,
    pub scan_and_bound_ns: u128,
    pub eligible_validation_ns: u128,
    pub bound_validation_ns: u128,
    pub pending_construction_ns: u128,
    pub heapify_ns: u128,
    pub scanned_points: usize,
    pub int8_dot_products: usize,
    pub encoded_bytes: usize,
    pub bound_count: usize,
    pub initial_heap_items: usize,
    pub eligible_reserved_bytes: usize,
    pub bounds_reserved_bytes: usize,
    pub bound_id_reserved_bytes: usize,
    pub pending_reserved_bytes: usize,
    pub total_temporary_reserved_bytes: usize,
    pub kernel: &'static str,
    pub storage_residency: &'static str,
}

impl<TStorage: EncodedStorage> PerVectorScalarIndex<TStorage> {
    pub fn encoded(&self) -> &EncodedVectorsPerVectorScalar<TStorage> {
        &self.encoded
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rank_state(
        &self,
        vector_storage: &VectorStorageEnum,
        eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<DenseRankState> {
        self.rank_state_with_refine_batch(
            vector_storage,
            eligible,
            raw_query,
            1,
            hardware_counter,
            stopped,
        )
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
        self.rank_state_with_refine_batch_impl(
            vector_storage,
            eligible,
            raw_query,
            exact_refine_batch,
            hardware_counter,
            stopped,
            None,
        )
    }

    #[cfg(feature = "stratumind-research")]
    #[allow(clippy::too_many_arguments)]
    pub fn rank_state_with_refine_batch_profiled(
        &self,
        vector_storage: &VectorStorageEnum,
        eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<(DenseRankState, PerVectorScalarBuildProfile)> {
        let mut profile = PerVectorScalarBuildProfile::default();
        let state = self.rank_state_with_refine_batch_impl(
            vector_storage,
            eligible,
            raw_query,
            exact_refine_batch,
            hardware_counter,
            stopped,
            Some(&mut profile),
        )?;
        Ok((state, profile))
    }

    #[allow(clippy::too_many_arguments)]
    fn rank_state_with_refine_batch_impl(
        &self,
        vector_storage: &VectorStorageEnum,
        mut eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
        mut profile: Option<&mut PerVectorScalarBuildProfile>,
    ) -> OperationResult<DenseRankState> {
        let total_started = profile.as_ref().map(|_| Instant::now());
        let phase_started = profile.as_ref().map(|_| Instant::now());
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
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.validation_ns = started.elapsed().as_nanos();
            profile.eligible_reserved_bytes = eligible
                .capacity()
                .saturating_mul(std::mem::size_of::<PointOffsetType>());
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
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.scan_and_bound_ns = started.elapsed().as_nanos();
            profile.scanned_points = eligible.len();
            profile.int8_dot_products = eligible.len();
            profile.encoded_bytes = self
                .encoded
                .metadata()
                .row_bytes()
                .saturating_mul(eligible.len());
            profile.bounds_reserved_bytes =
                bounds
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(PointOffsetType, f64, f64)>());
        }

        let point_count = eligible.len();
        let state = if profile.is_some() {
            let (state, rank_build) = DenseRankState::from_certificate_bounds_batched_profiled(
                eligible,
                bounds,
                exact_refine_batch,
                DensePhysicalPlan::PerVectorScalarCertificate,
                point_count,
            )?;
            let profile = profile.as_deref_mut().expect("profile checked above");
            profile.eligible_validation_ns = rank_build.eligible_validation_ns;
            profile.bound_validation_ns = rank_build.bound_validation_ns;
            profile.pending_construction_ns = rank_build.pending_construction_ns;
            profile.heapify_ns = rank_build.heapify_ns;
            profile.bound_count = rank_build.bound_count;
            profile.initial_heap_items = rank_build.initial_heap_items;
            profile.bound_id_reserved_bytes = rank_build.bound_id_reserved_bytes;
            profile.pending_reserved_bytes = rank_build.pending_reserved_bytes;
            profile.total_temporary_reserved_bytes = profile
                .eligible_reserved_bytes
                .saturating_add(profile.bounds_reserved_bytes)
                .saturating_add(profile.bound_id_reserved_bytes)
                .saturating_add(profile.pending_reserved_bytes);
            state
        } else {
            DenseRankState::from_certificate_bounds_batched(
                eligible,
                bounds,
                exact_refine_batch,
                DensePhysicalPlan::PerVectorScalarCertificate,
                point_count,
            )?
        };
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), total_started) {
            profile.total_build_ns = started.elapsed().as_nanos();
        }
        Ok(state)
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
