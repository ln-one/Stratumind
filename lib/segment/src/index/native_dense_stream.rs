// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Segment-owned exact Dense stream with a safe physical-plan router.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::AtomicBool;

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

mod plans;

use self::plans::{
    build_compact_certificate_cursor, build_exact_scan_cursor,
    build_per_vector_scalar_certificate_cursor, build_scalar_certificate_cursor, select_dense_plan,
};

pub const DEFAULT_DENSE_SCALAR_MIN_POINTS: usize = 4_096;
pub const DEFAULT_DENSE_COMPACT_MAX_POINTS: usize = DEFAULT_COMPACT_CERTIFICATE_MAX_POINTS;
const NATIVE_DENSE_SCORE_CHUNK_SIZE: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDensePlan {
    CompactCertificate,
    PerVectorScalarCertificate,
    ScalarCertificate,
    ExactScan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDensePolicy {
    pub scalar_min_points: usize,
    pub compact_max_points: usize,
    pub force_exact_scan: bool,
    /// Internal rollback/baseline switch. Production Auto keeps this false.
    pub disable_per_vector_scalar_certificate: bool,
    /// Internal benchmark switch. Production keeps this true.
    pub disable_compact_certificate: bool,
}

impl Default for NativeDensePolicy {
    fn default() -> Self {
        Self {
            scalar_min_points: DEFAULT_DENSE_SCALAR_MIN_POINTS,
            compact_max_points: DEFAULT_DENSE_COMPACT_MAX_POINTS,
            force_exact_scan: false,
            disable_per_vector_scalar_certificate: false,
            // Production defaults to Qdrant's native Scalar quantization
            // scorer. Compact remains benchmark-only until it can reuse the
            // same native storage and SIMD dispatch.
            disable_compact_certificate: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeDenseTelemetry {
    pub plan: Option<NativeDensePlan>,
    pub eligible_points: usize,
    pub native_quantized_scores: usize,
    pub exact_scores: usize,
    pub points_emitted: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeDenseExactRankProbe {
    pub id: PointOffsetType,
    pub score: f32,
    /// Zero-based exact rank within the cursor's frozen eligible universe.
    pub rank: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingBound {
    id: PointOffsetType,
    lower: f64,
    value: f64,
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

struct CertificateCursor<'a> {
    exact_scorer: Box<dyn FnMut(PointOffsetType) -> f32 + 'a>,
    bounds: BinaryHeap<PendingBound>,
    exact: BinaryHeap<PendingExact>,
    /// Query-lifetime canonical cache. Ordered continuation and arbitrary
    /// ExactRank probes must never score the same original vector twice.
    exact_scores: HashMap<PointOffsetType, f32>,
}

struct ExactScanCursor {
    points: Vec<ScoredPointOffset>,
    next: usize,
}

enum NativeDenseCursorInner<'a> {
    Certificate(CertificateCursor<'a>),
    Scan(ExactScanCursor),
}

pub struct NativeDenseIndexCursor<'a> {
    inner: NativeDenseCursorInner<'a>,
    /// Sorted once so probe validation does not require another bitmap or a
    /// second copy of the underlying vector storage.
    eligible: Vec<PointOffsetType>,
    telemetry: NativeDenseTelemetry,
}

impl<'a> NativeDenseIndexCursor<'a> {
    pub(crate) fn from_certificate_bounds(
        mut eligible: Vec<PointOffsetType>,
        bounds: Vec<(PointOffsetType, f64, f64)>,
        exact_scorer: impl FnMut(PointOffsetType) -> f32 + 'a,
        plan: NativeDensePlan,
        native_quantized_scores: usize,
    ) -> OperationResult<Self> {
        eligible.sort_unstable();
        if eligible.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(OperationError::inconsistent_storage(
                "native Dense session received duplicate eligible point offsets",
            ));
        }
        let mut bound_ids = bounds.iter().map(|(id, _, _)| *id).collect::<Vec<_>>();
        bound_ids.sort_unstable();
        if bound_ids != eligible
            || bounds
                .iter()
                .any(|(_, lower, upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err(OperationError::inconsistent_storage(
                "native Dense session bounds do not match its eligible universe",
            ));
        }
        let pending: Vec<_> = bounds
            .into_iter()
            .map(|(id, lower, value)| PendingBound { id, lower, value })
            .collect();
        let eligible_points = eligible.len();
        Ok(Self {
            inner: NativeDenseCursorInner::Certificate(CertificateCursor {
                exact_scorer: Box::new(exact_scorer),
                bounds: BinaryHeap::from(pending),
                exact: BinaryHeap::new(),
                exact_scores: HashMap::new(),
            }),
            eligible,
            telemetry: NativeDenseTelemetry {
                plan: Some(plan),
                eligible_points,
                native_quantized_scores,
                ..Default::default()
            },
        })
    }

    pub fn new(
        vector_storage: &'a VectorStorageEnum,
        quantized: Option<&QuantizedVectors>,
        eligible: Vec<PointOffsetType>,
        query: &[f32],
        policy: NativeDensePolicy,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        check_stopped(stopped)?;
        if query.is_empty() || query.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(OperationError::validation_error(
                "native Dense stream requires a non-empty finite Query",
            ));
        }
        let mut eligible = eligible;
        eligible.sort_unstable();
        if eligible.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(OperationError::inconsistent_storage(
                "native Dense stream received duplicate eligible point offsets",
            ));
        }
        let query_vector: QueryVector = VectorInternal::Dense(query.to_vec()).into();
        match select_dense_plan(vector_storage, quantized, &eligible, query, policy) {
            NativeDensePlan::PerVectorScalarCertificate => {
                build_per_vector_scalar_certificate_cursor(
                    vector_storage,
                    quantized.expect("PVS plan requires quantized vectors"),
                    eligible,
                    query,
                    hardware_counter,
                    stopped,
                )
            }
            NativeDensePlan::CompactCertificate => build_compact_certificate_cursor(
                vector_storage,
                quantized.expect("compact plan requires quantized vectors"),
                eligible,
                query,
                query_vector,
                hardware_counter,
                stopped,
            ),
            NativeDensePlan::ScalarCertificate => build_scalar_certificate_cursor(
                vector_storage,
                quantized.expect("Scalar plan requires quantized vectors"),
                eligible,
                query,
                query_vector,
                hardware_counter,
                stopped,
            ),
            NativeDensePlan::ExactScan => build_exact_scan_cursor(
                vector_storage,
                eligible,
                query_vector,
                hardware_counter,
                stopped,
            ),
        }
    }

    pub fn next_result(&mut self) -> OperationResult<Option<ScoredPointOffset>> {
        let point = match &mut self.inner {
            NativeDenseCursorInner::Scan(cursor) => {
                let point = cursor.points.get(cursor.next).copied();
                cursor.next += usize::from(point.is_some());
                point
            }
            NativeDenseCursorInner::Certificate(cursor) => loop {
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
                let score = exact_score_cached(cursor, &mut self.telemetry, bound.id);
                if f64::from(score) > bound.value {
                    return Err(OperationError::inconsistent_storage(format!(
                        "native Dense Scalar certificate violated for point {}: score {score}, bound {}",
                        bound.id, bound.value,
                    )));
                }
                cursor.exact.push(PendingExact(ScoredPointOffset {
                    idx: bound.id,
                    score,
                }));
            },
        };
        self.telemetry.points_emitted += usize::from(point.is_some());
        Ok(point)
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
    ) -> OperationResult<Vec<NativeDenseExactRankProbe>> {
        let mut unique = std::collections::HashSet::with_capacity(ids.len());
        for &id in ids {
            if !unique.insert(id) {
                return Err(OperationError::validation_error(format!(
                    "native Dense ExactRank probe contains duplicate point {id}",
                )));
            }
            if self.eligible.binary_search(&id).is_err() {
                return Err(OperationError::validation_error(format!(
                    "native Dense ExactRank probe point {id} is outside the frozen eligible universe",
                )));
            }
        }
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        match &mut self.inner {
            NativeDenseCursorInner::Scan(cursor) => ids
                .iter()
                .map(|&id| {
                    let rank = cursor
                        .points
                        .iter()
                        .position(|point| point.idx == id)
                        .ok_or_else(|| {
                            OperationError::inconsistent_storage(format!(
                                "native Dense exact scan lost eligible point {id}",
                            ))
                        })?;
                    Ok(NativeDenseExactRankProbe {
                        id,
                        score: cursor.points[rank].score,
                        rank,
                    })
                })
                .collect(),
            NativeDenseCursorInner::Certificate(cursor) => {
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
                            "native Dense Scalar certificate violated for point {id}: score {score}, interval [{lower}, {upper}]",
                        )));
                    }
                }

                Ok(targets
                    .into_iter()
                    .map(|(id, score)| NativeDenseExactRankProbe {
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

    pub fn telemetry(&self) -> NativeDenseTelemetry {
        self.telemetry
    }
}

fn exact_score_cached(
    cursor: &mut CertificateCursor<'_>,
    telemetry: &mut NativeDenseTelemetry,
    id: PointOffsetType,
) -> f32 {
    if let Some(&score) = cursor.exact_scores.get(&id) {
        return score;
    }
    let score = (cursor.exact_scorer)(id);
    cursor.exact_scores.insert(id, score);
    telemetry.exact_scores += 1;
    score
}

#[cfg(test)]
#[path = "native_dense_stream/tests.rs"]
mod tests;
