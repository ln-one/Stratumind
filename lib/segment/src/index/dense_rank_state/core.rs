// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Reader-independent exact Dense ranking state.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use ordered_float::OrderedFloat;

use super::types::{DenseExecutionPolicy, DenseExecutionTelemetry, DensePhysicalPlan};
use crate::common::check_stopped;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::vectors::{QueryVector, VectorElementType, VectorInternal};
use crate::types::{Distance, VectorStorageDatatype};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead};

#[path = "contiguous.rs"]
mod contiguous;
#[path = "plans.rs"]
mod plans;
use self::plans::{
    build_exact_scan_state, build_per_vector_scalar_certificate_state,
    build_scalar_certificate_state, select_dense_plan,
};

const DENSE_SCORE_CHUNK_SIZE: usize = 4_096;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DenseBuildProfile {
    pub eligible_validation_ns: u128,
    pub bound_validation_ns: u128,
    pub pending_construction_ns: u128,
    pub heapify_ns: u128,
    pub bound_count: usize,
    pub initial_heap_items: usize,
    pub bound_id_reserved_bytes: usize,
    pub pending_reserved_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PendingBound {
    pub(crate) id: PointOffsetType,
    pub(crate) lower: f64,
    pub(crate) value: f64,
}

impl Eq for PendingBound {}

impl Ord for PendingBound {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.value)
            .cmp(&OrderedFloat(other.value))
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for PendingBound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingExact(ScoredPointOffset);

impl Eq for PendingExact {}

impl Ord for PendingExact {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.idx.cmp(&self.0.idx))
    }
}

impl PartialOrd for PendingExact {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct CertificateState {
    exact_refine_batch: usize,
    bounds: BinaryHeap<PendingBound>,
    exact: BinaryHeap<PendingExact>,
    /// Query-lifetime canonical cache. Ordered continuation never scores the
    /// same original vector twice.
    exact_scores: HashMap<PointOffsetType, f32>,
    profile_refinement: bool,
}

struct ExactScanState {
    points: Option<Vec<ScoredPointOffset>>,
    next: usize,
}

enum DenseRankStateInner {
    Certificate(CertificateState),
    Scan(ExactScanState),
}

enum EligibleUniverse {
    Explicit(Vec<PointOffsetType>),
    Contiguous { count: usize },
}

impl EligibleUniverse {
    fn explicit(points: Vec<PointOffsetType>) -> Self {
        Self::Explicit(points)
    }

    fn contiguous(count: usize) -> Self {
        Self::Contiguous { count }
    }
}

/// Reader-independent state of one exact Dense ranking session.
///
/// The state owns every certificate heap and exact-score cache. A caller may
/// move it between Qdrant search workers and provide a short-lived exact scorer
/// only while advancing a batch under a Segment read view.
pub struct DenseRankState {
    inner: DenseRankStateInner,
    eligible: EligibleUniverse,
    telemetry: DenseExecutionTelemetry,
}

impl DenseRankState {
    pub(crate) fn from_certificate_bounds_batched(
        eligible: Vec<PointOffsetType>,
        bounds: Vec<(PointOffsetType, f64, f64)>,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
    ) -> OperationResult<Self> {
        Self::from_certificate_bounds_batched_impl(
            eligible,
            bounds,
            exact_refine_batch,
            plan,
            quantized_scores,
            None,
        )
    }

    pub(crate) fn from_certificate_bounds_batched_profiled(
        eligible: Vec<PointOffsetType>,
        bounds: Vec<(PointOffsetType, f64, f64)>,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
    ) -> OperationResult<(Self, DenseBuildProfile)> {
        let mut profile = DenseBuildProfile::default();
        let cursor = Self::from_certificate_bounds_batched_impl(
            eligible,
            bounds,
            exact_refine_batch,
            plan,
            quantized_scores,
            Some(&mut profile),
        )?;
        Ok((cursor, profile))
    }

    fn from_certificate_bounds_batched_impl(
        mut eligible: Vec<PointOffsetType>,
        bounds: Vec<(PointOffsetType, f64, f64)>,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
        mut profile: Option<&mut DenseBuildProfile>,
    ) -> OperationResult<Self> {
        let profile_refinement = profile.is_some();
        if exact_refine_batch == 0 {
            return Err(OperationError::validation_error(
                "exact Dense exact-refine batch must be positive",
            ));
        }
        let phase_started = profile.as_ref().map(|_| Instant::now());
        eligible.sort_unstable();
        if eligible.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(OperationError::inconsistent_storage(
                "exact Dense session received duplicate eligible point offsets",
            ));
        }
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.eligible_validation_ns = started.elapsed().as_nanos();
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
        let mut bound_ids = bounds.iter().map(|(id, _, _)| *id).collect::<Vec<_>>();
        bound_ids.sort_unstable();
        if bound_ids != eligible
            || bounds
                .iter()
                .any(|(_, lower, upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err(OperationError::inconsistent_storage(
                "exact Dense session bounds do not match its eligible universe",
            ));
        }
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.bound_validation_ns = started.elapsed().as_nanos();
            profile.bound_count = bounds.len();
            profile.bound_id_reserved_bytes = bound_ids
                .capacity()
                .saturating_mul(std::mem::size_of::<PointOffsetType>());
        }

        let phase_started = profile.as_ref().map(|_| Instant::now());
        let pending: Vec<_> = bounds
            .into_iter()
            .map(|(id, lower, value)| PendingBound { id, lower, value })
            .collect();
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), phase_started) {
            profile.pending_construction_ns = started.elapsed().as_nanos();
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

        let eligible_points = eligible.len();
        Ok(Self {
            inner: DenseRankStateInner::Certificate(CertificateState {
                exact_refine_batch,
                bounds,
                exact: BinaryHeap::new(),
                exact_scores: HashMap::new(),
                profile_refinement,
            }),
            eligible: EligibleUniverse::explicit(eligible),
            telemetry: DenseExecutionTelemetry {
                plan: Some(plan),
                eligible_points,
                quantized_scores,
                ..Default::default()
            },
        })
    }

    pub fn new(
        vector_storage: &VectorStorageEnum,
        quantized: Option<&QuantizedVectors>,
        eligible: Vec<PointOffsetType>,
        query: &[f32],
        policy: DenseExecutionPolicy,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        Self::new_with_exact_refine_batch_impl(
            vector_storage,
            quantized,
            eligible,
            query,
            policy,
            hardware_counter,
            stopped,
            1,
        )
    }

    /// Research-only matched baseline. Production callers keep the historical
    /// single-point refinement behavior through [`Self::new`].
    #[cfg(feature = "stratumind-research")]
    #[expect(clippy::too_many_arguments)]
    pub fn new_with_exact_refine_batch(
        vector_storage: &VectorStorageEnum,
        quantized: Option<&QuantizedVectors>,
        eligible: Vec<PointOffsetType>,
        query: &[f32],
        policy: DenseExecutionPolicy,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
        exact_refine_batch: usize,
    ) -> OperationResult<Self> {
        Self::new_with_exact_refine_batch_impl(
            vector_storage,
            quantized,
            eligible,
            query,
            policy,
            hardware_counter,
            stopped,
            exact_refine_batch,
        )
    }

    #[expect(clippy::too_many_arguments)]
    fn new_with_exact_refine_batch_impl(
        vector_storage: &VectorStorageEnum,
        quantized: Option<&QuantizedVectors>,
        eligible: Vec<PointOffsetType>,
        query: &[f32],
        policy: DenseExecutionPolicy,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
        exact_refine_batch: usize,
    ) -> OperationResult<Self> {
        check_stopped(stopped)?;
        if exact_refine_batch == 0 {
            return Err(OperationError::validation_error(
                "exact Dense exact-refine batch must be positive",
            ));
        }
        if query.is_empty() || query.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(OperationError::validation_error(
                "exact Dense stream requires a non-empty finite Query",
            ));
        }
        let mut eligible = eligible;
        eligible.sort_unstable();
        if eligible.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(OperationError::inconsistent_storage(
                "exact Dense stream received duplicate eligible point offsets",
            ));
        }
        match select_dense_plan(vector_storage, quantized, &eligible, policy) {
            DensePhysicalPlan::ScalarCertificate => build_scalar_certificate_state(
                vector_storage,
                quantized.expect("Scalar plan requires quantized vectors"),
                eligible,
                query,
                hardware_counter,
                stopped,
                exact_refine_batch,
            ),
            DensePhysicalPlan::PerVectorScalarCertificate => {
                build_per_vector_scalar_certificate_state(
                    vector_storage,
                    quantized.expect("PerVectorScalar plan requires quantized vectors"),
                    eligible,
                    query,
                    hardware_counter,
                    stopped,
                    exact_refine_batch,
                )
            }
            DensePhysicalPlan::ExactScan => build_exact_scan_state(eligible),
        }
    }

    /// Advance up to `max_results` certified ranks with a scorer borrowed only
    /// for this call.
    pub fn next_batch_with(
        &mut self,
        max_results: usize,
        mut exact_scorer: impl FnMut(&[PointOffsetType], &mut [f32]),
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if max_results == 0 {
            return Err(OperationError::validation_error(
                "exact Dense rank batch must be positive",
            ));
        }
        if let DenseRankStateInner::Scan(scan) = &mut self.inner
            && scan.points.is_none()
        {
            let ids = match &self.eligible {
                EligibleUniverse::Explicit(ids) => ids.clone(),
                EligibleUniverse::Contiguous { count } => (0..*count)
                    .map(|id| PointOffsetType::try_from(id).expect("validated point count"))
                    .collect(),
            };
            let mut scores = vec![0.0; ids.len()];
            exact_scorer(&ids, &mut scores);
            if scores.iter().any(|score| !score.is_finite()) {
                return Err(OperationError::inconsistent_storage(
                    "exact Dense scorer returned a non-finite score",
                ));
            }
            let mut ranked = ids
                .into_iter()
                .zip(scores)
                .map(|(idx, score)| ScoredPointOffset { idx, score })
                .collect::<Vec<_>>();
            ranked.sort_unstable_by(|left, right| {
                OrderedFloat(right.score)
                    .cmp(&OrderedFloat(left.score))
                    .then_with(|| left.idx.cmp(&right.idx))
            });
            self.telemetry.exact_scores = ranked.len();
            scan.points = Some(ranked);
        }
        let mut points = Vec::with_capacity(max_results);
        while points.len() < max_results {
            let point = match &mut self.inner {
                DenseRankStateInner::Scan(cursor) => {
                    let point = cursor
                        .points
                        .as_ref()
                        .expect("exact scan initialized")
                        .get(cursor.next)
                        .copied();
                    cursor.next += usize::from(point.is_some());
                    point
                }
                DenseRankStateInner::Certificate(state) => loop {
                    let fixed = state.exact.peek().is_some_and(|exact| {
                        state.bounds.peek().is_none_or(|bound| {
                            f64::from(exact.0.score) > bound.value
                                || (f64::from(exact.0.score) == bound.value
                                    && exact.0.idx < bound.id)
                        })
                    });
                    if fixed {
                        break state.exact.pop().map(|point| point.0);
                    }
                    let Some(bound) = state.bounds.pop() else {
                        break state.exact.pop().map(|point| point.0);
                    };
                    let mut refine = Vec::with_capacity(state.exact_refine_batch);
                    refine.push(bound);
                    while refine.len() < state.exact_refine_batch {
                        let Some(bound) = state.bounds.pop() else {
                            break;
                        };
                        refine.push(bound);
                    }
                    refine_bounds_with(state, &mut self.telemetry, refine, &mut exact_scorer)?;
                },
            };
            let Some(point) = point else {
                break;
            };
            self.telemetry.points_emitted += 1;
            points.push(point);
        }
        Ok(points)
    }

    pub fn telemetry(&self) -> DenseExecutionTelemetry {
        self.telemetry
    }

    pub fn eligible_len(&self) -> usize {
        match &self.eligible {
            EligibleUniverse::Explicit(points) => points.len(),
            EligibleUniverse::Contiguous { count, .. } => *count,
        }
    }
}

fn refine_bounds_with(
    cursor: &mut CertificateState,
    telemetry: &mut DenseExecutionTelemetry,
    bounds: Vec<PendingBound>,
    exact_scorer: &mut dyn FnMut(&[PointOffsetType], &mut [f32]),
) -> OperationResult<()> {
    let mut unresolved = Vec::with_capacity(bounds.len());
    for bound in &bounds {
        if !cursor.exact_scores.contains_key(&bound.id) {
            unresolved.push(bound.id);
        }
    }
    if !unresolved.is_empty() {
        let mut scores = vec![0.0; unresolved.len()];
        let started = cursor.profile_refinement.then(Instant::now);
        exact_scorer(&unresolved, &mut scores);
        if let Some(started) = started {
            telemetry.exact_refine_ns = telemetry
                .exact_refine_ns
                .saturating_add(started.elapsed().as_nanos());
            telemetry.exact_refine_batches += 1;
        }
        for (id, score) in unresolved.into_iter().zip(scores) {
            if !score.is_finite() {
                return Err(OperationError::inconsistent_storage(format!(
                    "exact Dense exact scorer returned a non-finite score for point {id}",
                )));
            }
            cursor.exact_scores.insert(id, score);
            telemetry.exact_scores += 1;
        }
    }
    for bound in bounds {
        let score = cursor.exact_scores[&bound.id];
        if f64::from(score) < bound.lower || f64::from(score) > bound.value {
            return Err(OperationError::inconsistent_storage(format!(
                "exact Dense certificate violated for point {}: score {score}, interval [{}, {}]",
                bound.id, bound.lower, bound.value,
            )));
        }
        cursor.exact.push(PendingExact(ScoredPointOffset {
            idx: bound.id,
            score,
        }));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
