// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Segment-owned exact Dense stream with a safe physical-plan router.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
#[cfg(feature = "stratumind-research")]
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use ordered_float::OrderedFloat;

use crate::common::check_stopped;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::vectors::{QueryVector, VectorElementType, VectorInternal};
use crate::types::{Distance, VectorStorageDatatype};
use crate::vector_storage::quantized::quantized_vectors::{
    CompactDenseVectorMetadata, DEFAULT_COMPACT_CERTIFICATE_MAX_POINTS, QuantizedVectors,
};
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead, new_raw_scorer};

mod contiguous;
mod plans;

use self::plans::{
    build_compact_certificate_cursor, build_exact_scan_cursor,
    build_per_vector_scalar_certificate_cursor, build_scalar_certificate_cursor, select_dense_plan,
};

pub const DEFAULT_DENSE_SCALAR_MIN_POINTS: usize = 4_096;
pub const DEFAULT_DENSE_COMPACT_MAX_POINTS: usize = DEFAULT_COMPACT_CERTIFICATE_MAX_POINTS;
pub const DEFAULT_DENSE_EXACT_PREFIX_MAX_DIMENSION: usize = 192;
const NATIVE_DENSE_SCORE_CHUNK_SIZE: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensePhysicalPlan {
    ExactPrefix,
    CompactCertificate,
    ScalarCertificate,
    PerVectorScalarCertificate,
    ExactScan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseExecutionPolicy {
    pub scalar_min_points: usize,
    pub compact_max_points: usize,
    pub exact_prefix_max_dimension: usize,
    pub force_exact_scan: bool,
    /// Internal baseline/rollback switch. Production Auto keeps this false.
    pub disable_per_vector_scalar_certificate: bool,
    /// Benchmark/profile switch. Production Auto keeps this false.
    pub disable_compact_certificate: bool,
}

impl Default for DenseExecutionPolicy {
    fn default() -> Self {
        Self {
            scalar_min_points: DEFAULT_DENSE_SCALAR_MIN_POINTS,
            compact_max_points: DEFAULT_DENSE_COMPACT_MAX_POINTS,
            exact_prefix_max_dimension: DEFAULT_DENSE_EXACT_PREFIX_MAX_DIMENSION,
            force_exact_scan: false,
            disable_per_vector_scalar_certificate: false,
            // Production prefers frozen PVS V1 when its derived index is
            // available. Compact remains an explicit research plan.
            disable_compact_certificate: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseExecutionTelemetry {
    pub plan: Option<DensePhysicalPlan>,
    pub eligible_points: usize,
    pub accepted_prefix_points: usize,
    pub exact_prefix_fallbacks: usize,
    pub quantized_scores: usize,
    pub exact_scores: usize,
    /// Exact scores imported from an earlier research-only candidate phase.
    /// They are validated and reused without another original-vector read.
    pub seeded_exact_scores: usize,
    pub points_emitted: usize,
    pub exact_refine_ns: u128,
    pub exact_refine_batches: usize,
}

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
pub struct DenseExactRankProbe {
    pub id: PointOffsetType,
    pub score: f32,
    /// Zero-based exact rank within the cursor's frozen eligible universe.
    pub rank: usize,
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

type ExactBatchScorer<'a> = dyn FnMut(&[PointOffsetType], &mut [f32]) + 'a;

struct CertificateState {
    exact_refine_batch: usize,
    bounds: BinaryHeap<PendingBound>,
    exact: BinaryHeap<PendingExact>,
    /// Query-lifetime canonical cache. Ordered continuation and arbitrary
    /// ExactRank probes must never score the same original vector twice.
    exact_scores: HashMap<PointOffsetType, f32>,
    profile_refinement: bool,
}

struct CertificateCursor<'a> {
    exact_scorer: Box<ExactBatchScorer<'a>>,
    state: CertificateState,
}

impl std::ops::Deref for CertificateCursor<'_> {
    type Target = CertificateState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl std::ops::DerefMut for CertificateCursor<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

struct ExactScanCursor {
    points: Vec<ScoredPointOffset>,
    next: usize,
}

enum ExactDenseCursorInner<'a> {
    Certificate(CertificateCursor<'a>),
    Scan(ExactScanCursor),
}

enum DenseRankStateInner {
    Certificate(CertificateState),
    Scan(ExactScanCursor),
}

enum EligibleUniverse {
    Explicit(Vec<PointOffsetType>),
    Contiguous {
        count: usize,
        #[cfg(feature = "stratumind-research")]
        materialized: OnceLock<Vec<PointOffsetType>>,
    },
}

impl EligibleUniverse {
    fn explicit(points: Vec<PointOffsetType>) -> Self {
        Self::Explicit(points)
    }

    fn contiguous(count: usize) -> Self {
        Self::Contiguous {
            count,
            #[cfg(feature = "stratumind-research")]
            materialized: OnceLock::new(),
        }
    }

    fn contains(&self, id: PointOffsetType) -> bool {
        match self {
            Self::Explicit(points) => points.binary_search(&id).is_ok(),
            Self::Contiguous { count, .. } => (id as usize) < *count,
        }
    }

    #[cfg(feature = "stratumind-research")]
    fn as_slice(&self) -> &[PointOffsetType] {
        match self {
            Self::Explicit(points) => points,
            Self::Contiguous {
                count,
                materialized,
            } => materialized.get_or_init(|| {
                (0..*count)
                    .map(|id| PointOffsetType::try_from(id).expect("validated point count"))
                    .collect()
            }),
        }
    }
}

pub struct ExactDenseCursor<'a> {
    inner: ExactDenseCursorInner<'a>,
    /// Sorted once so probe validation does not require another bitmap or a
    /// second copy of the underlying vector storage.
    eligible: EligibleUniverse,
    telemetry: DenseExecutionTelemetry,
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

impl<'a> ExactDenseCursor<'a> {
    #[cfg(feature = "stratumind-research")]
    pub fn eligible_points(&self) -> &[PointOffsetType] {
        self.eligible.as_slice()
    }

    #[cfg(feature = "stratumind-research")]
    pub(crate) fn from_certificate_bounds(
        eligible: Vec<PointOffsetType>,
        bounds: Vec<(PointOffsetType, f64, f64)>,
        exact_scorer: impl FnMut(PointOffsetType) -> f32 + 'a,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
    ) -> OperationResult<Self> {
        let mut exact_scorer = exact_scorer;
        Self::from_certificate_bounds_batched(
            eligible,
            bounds,
            move |ids, scores| {
                for (&id, score) in ids.iter().zip(scores) {
                    *score = exact_scorer(id);
                }
            },
            1,
            plan,
            quantized_scores,
        )
    }

    pub(crate) fn from_certificate_bounds_batched(
        eligible: Vec<PointOffsetType>,
        bounds: Vec<(PointOffsetType, f64, f64)>,
        exact_scorer: impl FnMut(&[PointOffsetType], &mut [f32]) + 'a,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
    ) -> OperationResult<Self> {
        Self::from_certificate_bounds_batched_impl(
            eligible,
            bounds,
            exact_scorer,
            exact_refine_batch,
            plan,
            quantized_scores,
            None,
        )
    }

    pub(crate) fn from_certificate_bounds_batched_profiled(
        eligible: Vec<PointOffsetType>,
        bounds: Vec<(PointOffsetType, f64, f64)>,
        exact_scorer: impl FnMut(&[PointOffsetType], &mut [f32]) + 'a,
        exact_refine_batch: usize,
        plan: DensePhysicalPlan,
        quantized_scores: usize,
    ) -> OperationResult<(Self, DenseBuildProfile)> {
        let mut profile = DenseBuildProfile::default();
        let cursor = Self::from_certificate_bounds_batched_impl(
            eligible,
            bounds,
            exact_scorer,
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
        exact_scorer: impl FnMut(&[PointOffsetType], &mut [f32]) + 'a,
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
            inner: ExactDenseCursorInner::Certificate(CertificateCursor {
                exact_scorer: Box::new(exact_scorer),
                state: CertificateState {
                    exact_refine_batch,
                    bounds,
                    exact: BinaryHeap::new(),
                    exact_scores: HashMap::new(),
                    profile_refinement,
                },
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
        vector_storage: &'a VectorStorageEnum,
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
        vector_storage: &'a VectorStorageEnum,
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
        vector_storage: &'a VectorStorageEnum,
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
        let query_vector: QueryVector = VectorInternal::Dense(query.to_vec()).into();
        match select_dense_plan(vector_storage, quantized, &eligible, query, policy) {
            DensePhysicalPlan::CompactCertificate => build_compact_certificate_cursor(
                vector_storage,
                quantized.expect("compact plan requires quantized vectors"),
                eligible,
                query,
                query_vector,
                hardware_counter,
                stopped,
                exact_refine_batch,
            ),
            DensePhysicalPlan::ScalarCertificate => build_scalar_certificate_cursor(
                vector_storage,
                quantized.expect("Scalar plan requires quantized vectors"),
                eligible,
                query,
                query_vector,
                hardware_counter,
                stopped,
                exact_refine_batch,
            ),
            DensePhysicalPlan::PerVectorScalarCertificate => {
                build_per_vector_scalar_certificate_cursor(
                    vector_storage,
                    quantized.expect("PerVectorScalar plan requires quantized vectors"),
                    eligible,
                    query,
                    hardware_counter,
                    stopped,
                    exact_refine_batch,
                )
            }
            DensePhysicalPlan::ExactScan => build_exact_scan_cursor(
                vector_storage,
                eligible,
                query_vector,
                hardware_counter,
                stopped,
            ),
            DensePhysicalPlan::ExactPrefix => unreachable!("prefix is owned by SegmentReadView"),
        }
    }

    pub fn next_result(&mut self) -> OperationResult<Option<ScoredPointOffset>> {
        let point = match &mut self.inner {
            ExactDenseCursorInner::Scan(cursor) => {
                let point = cursor.points.get(cursor.next).copied();
                cursor.next += usize::from(point.is_some());
                point
            }
            ExactDenseCursorInner::Certificate(cursor) => loop {
                let fixed = cursor.exact.peek().is_some_and(|exact| {
                    cursor.bounds.peek().is_none_or(|bound| {
                        f64::from(exact.0.score) > bound.value
                            || (f64::from(exact.0.score) == bound.value && exact.0.idx < bound.id)
                    })
                });
                if fixed {
                    break cursor.exact.pop().map(|point| point.0);
                }
                let Some(bound) = cursor.bounds.pop() else {
                    break cursor.exact.pop().map(|point| point.0);
                };
                let mut refine = Vec::with_capacity(cursor.exact_refine_batch);
                refine.push(bound);
                while refine.len() < cursor.exact_refine_batch {
                    let Some(bound) = cursor.bounds.pop() else {
                        break;
                    };
                    refine.push(bound);
                }
                refine_bounds(cursor, &mut self.telemetry, refine)?;
            },
        };
        self.telemetry.points_emitted += usize::from(point.is_some());
        Ok(point)
    }

    pub fn into_rank_state(self) -> DenseRankState {
        DenseRankState {
            inner: match self.inner {
                ExactDenseCursorInner::Certificate(cursor) => {
                    DenseRankStateInner::Certificate(cursor.state)
                }
                ExactDenseCursorInner::Scan(cursor) => DenseRankStateInner::Scan(cursor),
            },
            eligible: self.eligible,
            telemetry: self.telemetry,
        }
    }

    pub fn take_rank_state(&mut self) -> DenseRankState {
        let inner = std::mem::replace(
            &mut self.inner,
            ExactDenseCursorInner::Scan(ExactScanCursor {
                points: Vec::new(),
                next: 0,
            }),
        );
        let eligible =
            std::mem::replace(&mut self.eligible, EligibleUniverse::explicit(Vec::new()));
        DenseRankState {
            inner: match inner {
                ExactDenseCursorInner::Certificate(cursor) => {
                    DenseRankStateInner::Certificate(cursor.state)
                }
                ExactDenseCursorInner::Scan(cursor) => DenseRankStateInner::Scan(cursor),
            },
            eligible,
            telemetry: self.telemetry,
        }
    }

    /// Imports exact scores computed earlier in the same frozen query view.
    ///
    /// This research-only bridge lets an HNSW/Sparse candidate phase hand its
    /// original-vector work to the canonical Native Dense rank session. Every
    /// seed is checked against the eligible universe and the cursor's safe
    /// quantized interval before it enters the exact-score cache.
    #[cfg(feature = "stratumind-research")]
    pub fn seed_exact_scores(
        &mut self,
        seeds: &[(PointOffsetType, f32)],
    ) -> OperationResult<usize> {
        let mut unique = std::collections::HashSet::with_capacity(seeds.len());
        for &(id, score) in seeds {
            if !unique.insert(id) {
                return Err(OperationError::validation_error(format!(
                    "exact Dense seed batch contains duplicate point {id}",
                )));
            }
            if !score.is_finite() {
                return Err(OperationError::validation_error(format!(
                    "exact Dense seed for point {id} is non-finite",
                )));
            }
            if !self.eligible.contains(id) {
                return Err(OperationError::validation_error(format!(
                    "exact Dense seed point {id} is outside the frozen eligible universe",
                )));
            }
        }

        let mut inserted = 0usize;
        match &mut self.inner {
            ExactDenseCursorInner::Scan(cursor) => {
                for &(id, score) in seeds {
                    let expected = cursor
                        .points
                        .iter()
                        .find(|point| point.idx == id)
                        .ok_or_else(|| {
                            OperationError::inconsistent_storage(format!(
                                "exact Dense exact scan lost seed point {id}",
                            ))
                        })?
                        .score;
                    if expected.to_bits() != score.to_bits() {
                        return Err(OperationError::inconsistent_storage(format!(
                            "exact Dense seed score changed for point {id}: {score} != {expected}",
                        )));
                    }
                }
            }
            ExactDenseCursorInner::Certificate(cursor) => {
                for &(id, score) in seeds {
                    if let Some(&current) = cursor.exact_scores.get(&id) {
                        if current.to_bits() != score.to_bits() {
                            return Err(OperationError::inconsistent_storage(format!(
                                "exact Dense seed score changed for point {id}: {score} != {current}",
                            )));
                        }
                        continue;
                    }
                    let bound = cursor
                        .bounds
                        .iter()
                        .find(|bound| bound.id == id)
                        .ok_or_else(|| {
                            OperationError::inconsistent_storage(format!(
                                "exact Dense seed point {id} has no unresolved certificate bound",
                            ))
                        })?;
                    if f64::from(score) < bound.lower || f64::from(score) > bound.value {
                        return Err(OperationError::inconsistent_storage(format!(
                            "exact Dense seed violated point {id} interval [{}, {}] with {score}",
                            bound.lower, bound.value,
                        )));
                    }
                    cursor.exact_scores.insert(id, score);
                    inserted += 1;
                }
            }
        }
        self.telemetry.seeded_exact_scores =
            self.telemetry.seeded_exact_scores.saturating_add(inserted);
        Ok(inserted)
    }

    /// Resolve exact Dense ranks for externally discovered points without
    /// starting another Scalar scan or rebuilding the ordered-stream heap.
    ///
    /// Scalar bounds were materialized once in `new`. A probe only exact-scores
    /// unresolved bounds that can still outrank at least one target. Every
    /// exact score is cached and immediately reusable by `next_result` and all
    /// later probe batches.
    pub fn probe_exact_ranks(
        &mut self,
        ids: &[PointOffsetType],
    ) -> OperationResult<Vec<DenseExactRankProbe>> {
        let mut unique = std::collections::HashSet::with_capacity(ids.len());
        for &id in ids {
            if !unique.insert(id) {
                return Err(OperationError::validation_error(format!(
                    "exact Dense ExactRank probe contains duplicate point {id}",
                )));
            }
            if !self.eligible.contains(id) {
                return Err(OperationError::validation_error(format!(
                    "exact Dense ExactRank probe point {id} is outside the frozen eligible universe",
                )));
            }
        }
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        match &mut self.inner {
            ExactDenseCursorInner::Scan(cursor) => ids
                .iter()
                .map(|&id| {
                    let rank = cursor
                        .points
                        .iter()
                        .position(|point| point.idx == id)
                        .ok_or_else(|| {
                            OperationError::inconsistent_storage(format!(
                                "exact Dense exact scan lost eligible point {id}",
                            ))
                        })?;
                    Ok(DenseExactRankProbe {
                        id,
                        score: cursor.points[rank].score,
                        rank,
                    })
                })
                .collect(),
            ExactDenseCursorInner::Certificate(cursor) => {
                let targets: Vec<_> = ids
                    .iter()
                    .map(|&id| {
                        let score = exact_score_cached(cursor, &mut self.telemetry, id);
                        (id, score)
                    })
                    .collect();

                let ambiguous = cursor
                    .bounds
                    .iter()
                    .filter(|bound| !cursor.exact_scores.contains_key(&bound.id))
                    .filter(|bound| {
                        targets.iter().any(|&(target_id, target_score)| {
                            let target_score = f64::from(target_score);
                            let definitely_outranks = bound.lower > target_score
                                || (bound.lower == target_score && bound.id < target_id);
                            let possibly_outranks = bound.value > target_score
                                || (bound.value == target_score && bound.id < target_id);
                            possibly_outranks && !definitely_outranks
                        })
                    })
                    .map(|bound| (bound.id, bound.lower, bound.value))
                    .collect::<Vec<_>>();
                for (id, lower, upper) in ambiguous {
                    let score = exact_score_cached(cursor, &mut self.telemetry, id);
                    if f64::from(score) < lower || f64::from(score) > upper {
                        return Err(OperationError::inconsistent_storage(format!(
                            "exact Dense certificate violated for point {id}: score {score}, interval [{lower}, {upper}]",
                        )));
                    }
                }

                Ok(targets
                    .into_iter()
                    .map(|(id, score)| DenseExactRankProbe {
                        id,
                        score,
                        rank: cursor
                            .exact_scores
                            .iter()
                            .filter(|&(&other_id, &other_score)| {
                                other_score > score || (other_score == score && other_id < id)
                            })
                            .count()
                            + cursor
                                .bounds
                                .iter()
                                .filter(|bound| !cursor.exact_scores.contains_key(&bound.id))
                                .filter(|bound| {
                                    bound.lower > f64::from(score)
                                        || (bound.lower == f64::from(score) && bound.id < id)
                                })
                                .count(),
                    })
                    .collect())
            }
        }
    }

    /// Research-only ExactRank probe with an authoritative tie key.
    ///
    /// Segment internal offsets are physical identities and need not follow
    /// the external point-identity order required by the WRRF contract.
    #[cfg(feature = "stratumind-research")]
    pub fn probe_exact_ranks_by_key<K: Ord + Copy>(
        &mut self,
        ids: &[PointOffsetType],
        tie_key: impl Fn(PointOffsetType) -> K,
    ) -> OperationResult<Vec<DenseExactRankProbe>> {
        let mut unique = std::collections::HashSet::with_capacity(ids.len());
        for &id in ids {
            if !unique.insert(id) {
                return Err(OperationError::validation_error(format!(
                    "exact Dense ExactRank probe contains duplicate point {id}",
                )));
            }
            if !self.eligible.contains(id) {
                return Err(OperationError::validation_error(format!(
                    "exact Dense ExactRank probe point {id} is outside the frozen eligible universe",
                )));
            }
        }
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        match &mut self.inner {
            ExactDenseCursorInner::Scan(cursor) => ids
                .iter()
                .map(|&id| {
                    let target = cursor
                        .points
                        .iter()
                        .find(|point| point.idx == id)
                        .ok_or_else(|| {
                            OperationError::inconsistent_storage(format!(
                                "exact Dense exact scan lost eligible point {id}",
                            ))
                        })?;
                    let target_key = tie_key(id);
                    let rank = cursor
                        .points
                        .iter()
                        .filter(|point| {
                            point.score > target.score
                                || (point.score == target.score && tie_key(point.idx) < target_key)
                        })
                        .count();
                    Ok(DenseExactRankProbe {
                        id,
                        score: target.score,
                        rank,
                    })
                })
                .collect(),
            ExactDenseCursorInner::Certificate(cursor) => {
                let targets: Vec<_> = ids
                    .iter()
                    .map(|&id| {
                        let score = exact_score_cached(cursor, &mut self.telemetry, id);
                        (id, score)
                    })
                    .collect();
                let ambiguous = cursor
                    .bounds
                    .iter()
                    .filter(|bound| !cursor.exact_scores.contains_key(&bound.id))
                    .filter(|bound| {
                        targets.iter().any(|&(target_id, target_score)| {
                            let target_score = f64::from(target_score);
                            let bound_key = tie_key(bound.id);
                            let target_key = tie_key(target_id);
                            let definitely_outranks = bound.lower > target_score
                                || (bound.lower == target_score && bound_key < target_key);
                            let possibly_outranks = bound.value > target_score
                                || (bound.value == target_score && bound_key < target_key);
                            possibly_outranks && !definitely_outranks
                        })
                    })
                    .map(|bound| (bound.id, bound.lower, bound.value))
                    .collect::<Vec<_>>();
                for (id, lower, upper) in ambiguous {
                    let score = exact_score_cached(cursor, &mut self.telemetry, id);
                    if f64::from(score) < lower || f64::from(score) > upper {
                        return Err(OperationError::inconsistent_storage(format!(
                            "exact Dense certificate violated for point {id}: score {score}, interval [{lower}, {upper}]",
                        )));
                    }
                }

                Ok(targets
                    .into_iter()
                    .map(|(id, score)| {
                        let target_key = tie_key(id);
                        let rank = cursor
                            .exact_scores
                            .iter()
                            .filter(|&(&other_id, &other_score)| {
                                other_score > score
                                    || (other_score == score && tie_key(other_id) < target_key)
                            })
                            .count()
                            + cursor
                                .bounds
                                .iter()
                                .filter(|bound| !cursor.exact_scores.contains_key(&bound.id))
                                .filter(|bound| {
                                    bound.lower > f64::from(score)
                                        || (bound.lower == f64::from(score)
                                            && tie_key(bound.id) < target_key)
                                })
                                .count();
                        DenseExactRankProbe { id, score, rank }
                    })
                    .collect())
            }
        }
    }

    pub fn telemetry(&self) -> DenseExecutionTelemetry {
        self.telemetry
    }
}

fn exact_score_cached(
    cursor: &mut CertificateCursor<'_>,
    telemetry: &mut DenseExecutionTelemetry,
    id: PointOffsetType,
) -> f32 {
    if let Some(&score) = cursor.exact_scores.get(&id) {
        return score;
    }
    let mut scores = [0.0];
    let started = cursor.profile_refinement.then(Instant::now);
    (cursor.exact_scorer)(&[id], &mut scores);
    if let Some(started) = started {
        telemetry.exact_refine_ns = telemetry
            .exact_refine_ns
            .saturating_add(started.elapsed().as_nanos());
        telemetry.exact_refine_batches += 1;
    }
    let score = scores[0];
    cursor.exact_scores.insert(id, score);
    telemetry.exact_scores += 1;
    score
}

impl DenseRankState {
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
        let mut points = Vec::with_capacity(max_results);
        while points.len() < max_results {
            let point = match &mut self.inner {
                DenseRankStateInner::Scan(cursor) => {
                    let point = cursor.points.get(cursor.next).copied();
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

fn refine_bounds(
    cursor: &mut CertificateCursor<'_>,
    telemetry: &mut DenseExecutionTelemetry,
    bounds: Vec<PendingBound>,
) -> OperationResult<()> {
    let CertificateCursor {
        exact_scorer,
        state,
    } = cursor;
    refine_bounds_with(state, telemetry, bounds, exact_scorer)
}

fn refine_bounds_with(
    cursor: &mut CertificateState,
    telemetry: &mut DenseExecutionTelemetry,
    bounds: Vec<PendingBound>,
    exact_scorer: &mut ExactBatchScorer<'_>,
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
#[path = "exact_dense_stream/tests.rs"]
mod tests;
