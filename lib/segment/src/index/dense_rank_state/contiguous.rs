use super::*;

impl DenseRankState {
    pub(crate) fn from_contiguous_pending_batched(
        point_count: usize,
        pending: Vec<PendingBound>,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
    ) -> OperationResult<Self> {
        Self::from_contiguous_pending_batched_impl(
            point_count,
            pending,
            exact_refine_batch,
            plan,
            quantized_scores,
            None,
        )
    }

    pub(crate) fn from_contiguous_pending_batched_profiled(
        point_count: usize,
        pending: Vec<PendingBound>,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
    ) -> OperationResult<(Self, DenseBuildProfile)> {
        let mut profile = DenseBuildProfile::default();
        let cursor = Self::from_contiguous_pending_batched_impl(
            point_count,
            pending,
            exact_refine_batch,
            plan,
            quantized_scores,
            Some(&mut profile),
        )?;
        Ok((cursor, profile))
    }

    fn from_contiguous_pending_batched_impl(
        point_count: usize,
        pending: Vec<PendingBound>,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
        mut profile: Option<&mut DenseBuildProfile>,
    ) -> OperationResult<Self> {
        if exact_refine_batch == 0 {
            return Err(OperationError::validation_error(
                "exact Dense exact-refine batch must be positive",
            ));
        }
        if point_count > PointOffsetType::MAX as usize + 1 {
            return Err(OperationError::inconsistent_storage(
                "contiguous exact Dense universe exceeds PointOffsetType",
            ));
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
        if pending.len() != point_count
            || pending.iter().enumerate().any(|(ordinal, bound)| {
                bound.id as usize != ordinal
                    || !bound.lower.is_finite()
                    || !bound.value.is_finite()
                    || bound.lower > bound.value
            })
        {
            return Err(OperationError::inconsistent_storage(
                "exact Dense Pending bounds do not match the contiguous eligible universe",
            ));
        }
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.bound_validation_ns = started.elapsed().as_nanos();
            profile.bound_count = pending.len();
            profile.pending_reserved_bytes = pending
                .capacity()
                .saturating_mul(std::mem::size_of::<PendingBound>());
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
        let bounds = BinaryHeap::from(pending);
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.heapify_ns = started.elapsed().as_nanos();
            profile.initial_heap_items = bounds.len();
        }

        let profile_refinement = profile.is_some();
        Ok(Self {
            inner: DenseRankStateInner::Certificate(CertificateState {
                exact_refine_batch,
                bounds,
                exact: BinaryHeap::new(),
                exact_scores: HashMap::new(),
                profile_refinement,
            }),
            eligible: EligibleUniverse::contiguous(point_count),
            telemetry: DenseExecutionTelemetry {
                plan: Some(plan),
                eligible_points: point_count,
                quantized_scores,
                ..Default::default()
            },
        })
    }
}
