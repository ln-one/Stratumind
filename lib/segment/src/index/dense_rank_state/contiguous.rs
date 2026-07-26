use super::*;

impl DenseRankState {
    pub(crate) fn from_contiguous_pending_batched(
        point_count: usize,
        pending: Vec<PendingBound>,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
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

        let bounds = BinaryHeap::from(pending);

        Ok(Self {
            inner: DenseRankStateInner::Certificate(CertificateState {
                exact_refine_batch,
                bounds,
                exact: BinaryHeap::new(),
                exact_scores: HashMap::new(),
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
