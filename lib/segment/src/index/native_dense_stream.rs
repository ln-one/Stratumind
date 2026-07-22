// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Segment-owned exact Dense stream with a safe physical-plan router.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use ordered_float::OrderedFloat;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::vectors::{QueryVector, VectorElementType, VectorInternal};
use crate::types::{Distance, VectorStorageDatatype};
use crate::vector_storage::quantized::quantized_vectors::{
    CompactDenseVectorMetadata, DEFAULT_COMPACT_CERTIFICATE_MAX_POINTS, QuantizedVectors,
};
use crate::vector_storage::{RawScorer, VectorStorageEnum, VectorStorageRead, new_raw_scorer};

pub const DEFAULT_DENSE_SCALAR_MIN_POINTS: usize = 4_096;
pub const DEFAULT_DENSE_COMPACT_MAX_POINTS: usize = DEFAULT_COMPACT_CERTIFICATE_MAX_POINTS;
pub const DEFAULT_DENSE_EXACT_PREFIX_MAX_DIMENSION: usize = 192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDensePlan {
    ExactPrefix,
    CompactCertificate,
    ScalarCertificate,
    ExactScan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDensePolicy {
    pub scalar_min_points: usize,
    pub compact_max_points: usize,
    pub exact_prefix_max_dimension: usize,
    pub force_exact_scan: bool,
    /// Benchmark/profile switch. Production Auto keeps this false.
    pub disable_compact_certificate: bool,
}

impl Default for NativeDensePolicy {
    fn default() -> Self {
        Self {
            scalar_min_points: DEFAULT_DENSE_SCALAR_MIN_POINTS,
            compact_max_points: DEFAULT_DENSE_COMPACT_MAX_POINTS,
            exact_prefix_max_dimension: DEFAULT_DENSE_EXACT_PREFIX_MAX_DIMENSION,
            force_exact_scan: false,
            disable_compact_certificate: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeDenseTelemetry {
    pub plan: Option<NativeDensePlan>,
    pub eligible_points: usize,
    pub accepted_prefix_points: usize,
    pub exact_prefix_fallbacks: usize,
    pub native_quantized_scores: usize,
    pub exact_scores: usize,
    pub points_emitted: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingBound {
    id: PointOffsetType,
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
    exact_scorer: Box<dyn RawScorer + 'a>,
    bounds: BinaryHeap<PendingBound>,
    exact: BinaryHeap<PendingExact>,
}

enum NativeDenseCursorInner<'a> {
    Certificate(CertificateCursor<'a>),
    Scan(VecDeque<ScoredPointOffset>),
}

pub struct NativeDenseIndexCursor<'a> {
    inner: NativeDenseCursorInner<'a>,
    telemetry: NativeDenseTelemetry,
}

impl<'a> NativeDenseIndexCursor<'a> {
    pub fn new(
        vector_storage: &'a VectorStorageEnum,
        quantized: Option<&QuantizedVectors>,
        eligible: Vec<PointOffsetType>,
        query: &[f32],
        policy: NativeDensePolicy,
        hardware_counter: &HardwareCounterCell,
    ) -> OperationResult<Self> {
        if query.is_empty() || query.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(OperationError::validation_error(
                "native Dense stream requires a non-empty finite Query",
            ));
        }
        let query_vector: QueryVector = VectorInternal::Dense(query.to_vec()).into();
        let compact_available = !policy.force_exact_scan
            && !policy.disable_compact_certificate
            && eligible.len() >= policy.scalar_min_points
            && eligible.len() <= policy.compact_max_points
            && matches!(vector_storage.distance(), Distance::Dot | Distance::Cosine)
            && vector_storage.datatype() == VectorStorageDatatype::Float32
            && quantized.is_some_and(|quantized| {
                quantized
                    .compact_dense_certificate()
                    .is_some_and(|compact| {
                        compact.dimension() == query.len()
                            && eligible.iter().all(|id| compact.row(*id).is_some())
                    })
            });
        let scalar_available = !policy.force_exact_scan
            && eligible.len() >= policy.scalar_min_points
            && matches!(vector_storage.distance(), Distance::Dot | Distance::Cosine)
            && quantized.is_some_and(|quantized| {
                matches!(quantized.distance(), Distance::Dot | Distance::Cosine)
                    && quantized.datatype() == VectorStorageDatatype::Float32
                    && quantized
                        .scalar_reconstruction_table()
                        .is_some_and(|table| eligible.iter().all(|id| (*id as usize) < table.len()))
            });

        if compact_available {
            let compact = quantized
                .and_then(QuantizedVectors::compact_dense_certificate)
                .expect("availability checked");
            let processed_query = vector_storage
                .distance()
                .preprocess_vector::<VectorElementType>(query.to_vec());
            let (query_codes, query_metadata) = compact_encode(&processed_query);
            let dimension = query.len() as f64;
            let gamma = f64::from(f32::EPSILON) * (8.0 * dimension + 64.0);
            let mut bounds = Vec::with_capacity(eligible.len());
            for &id in &eligible {
                let (document_codes, document) = compact
                    .row(id)
                    .expect("compact certificate availability checked");
                let approximate = integer_dot(&query_codes, document_codes) as f64
                    * query_metadata.scale
                    * document.scale;
                let error = query_metadata.residual_norm * document.original_norm
                    + query_metadata.reconstructed_norm * document.residual_norm;
                let scale = approximate.abs()
                    + error
                    + query_metadata.original_norm * document.original_norm;
                let floating_guard = if gamma < 1.0 {
                    scale * gamma / (1.0 - gamma)
                } else {
                    f64::INFINITY
                };
                bounds.push(PendingBound {
                    id,
                    value: (approximate + error + floating_guard).next_up(),
                });
            }
            let exact_scorer =
                new_raw_scorer(query_vector, vector_storage, hardware_counter.fork())?;
            return Ok(Self {
                inner: NativeDenseCursorInner::Certificate(CertificateCursor {
                    exact_scorer,
                    bounds: BinaryHeap::from(bounds),
                    exact: BinaryHeap::new(),
                }),
                telemetry: NativeDenseTelemetry {
                    plan: Some(NativeDensePlan::CompactCertificate),
                    eligible_points: eligible.len(),
                    native_quantized_scores: eligible.len(),
                    ..Default::default()
                },
            });
        }

        if scalar_available {
            let quantized = quantized.expect("availability checked");
            let reconstruction = quantized
                .scalar_reconstruction_table()
                .expect("availability checked");
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
            let approximate_scorer =
                quantized.raw_scorer(query_vector.clone(), hardware_counter.fork())?;
            let mut approximate = vec![0.0; eligible.len()];
            approximate_scorer.score_points(&eligible, &mut approximate);

            let dimension = query.len() as f64;
            let gamma = f64::from(f32::EPSILON) * (8.0 * dimension + 64.0);
            let mut bounds = Vec::with_capacity(eligible.len());
            for (&id, approximate) in eligible.iter().zip(approximate) {
                let document = reconstruction[id as usize];
                let error = query_stats.residual_norm * document.original_norm
                    + query_stats.reconstructed_norm * document.residual_norm;
                let scale = f64::from(approximate).abs()
                    + error
                    + query_stats.original_norm * document.original_norm;
                let floating_guard = if gamma < 1.0 {
                    scale * gamma / (1.0 - gamma)
                } else {
                    f64::INFINITY
                };
                bounds.push(PendingBound {
                    id,
                    value: (f64::from(approximate) + error + floating_guard).next_up(),
                });
            }
            drop(approximate_scorer);
            let exact_scorer =
                new_raw_scorer(query_vector, vector_storage, hardware_counter.fork())?;
            return Ok(Self {
                inner: NativeDenseCursorInner::Certificate(CertificateCursor {
                    exact_scorer,
                    bounds: BinaryHeap::from(bounds),
                    exact: BinaryHeap::new(),
                }),
                telemetry: NativeDenseTelemetry {
                    plan: Some(NativeDensePlan::ScalarCertificate),
                    eligible_points: eligible.len(),
                    native_quantized_scores: eligible.len(),
                    ..Default::default()
                },
            });
        }

        let scorer = new_raw_scorer(query_vector, vector_storage, hardware_counter.fork())?;
        let mut scores = vec![0.0; eligible.len()];
        scorer.score_points(&eligible, &mut scores);
        let mut points: Vec<_> = eligible
            .into_iter()
            .zip(scores)
            .map(|(idx, score)| ScoredPointOffset { idx, score })
            .collect();
        points.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.idx.cmp(&right.idx))
        });
        let point_count = points.len();
        Ok(Self {
            inner: NativeDenseCursorInner::Scan(VecDeque::from(points)),
            telemetry: NativeDenseTelemetry {
                plan: Some(NativeDensePlan::ExactScan),
                eligible_points: point_count,
                exact_scores: point_count,
                ..Default::default()
            },
        })
    }

    pub fn next_result(&mut self) -> OperationResult<Option<ScoredPointOffset>> {
        let point = match &mut self.inner {
            NativeDenseCursorInner::Scan(points) => points.pop_front(),
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
                let score = cursor.exact_scorer.score_point(bound.id);
                if f64::from(score) > bound.value {
                    return Err(OperationError::inconsistent_storage(format!(
                        "native Dense Scalar certificate violated for point {}: score {score}, bound {}",
                        bound.id, bound.value,
                    )));
                }
                self.telemetry.exact_scores += 1;
                cursor.exact.push(PendingExact(ScoredPointOffset {
                    idx: bound.id,
                    score,
                }));
            },
        };
        self.telemetry.points_emitted += usize::from(point.is_some());
        Ok(point)
    }

    pub fn telemetry(&self) -> NativeDenseTelemetry {
        self.telemetry
    }
}

fn compact_encode(vector: &[f32]) -> (Vec<i8>, CompactDenseVectorMetadata) {
    const MAX_CODE: f64 = 127.0;
    let original_norm = vector
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let max_abs = vector
        .iter()
        .map(|value| f64::from(*value).abs())
        .fold(0.0, f64::max);
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / MAX_CODE
    };
    let codes: Vec<_> = vector
        .iter()
        .map(|value| {
            (f64::from(*value) / scale)
                .round()
                .clamp(-MAX_CODE, MAX_CODE) as i8
        })
        .collect();
    let mut reconstructed_squared = 0.0;
    let mut residual_squared = 0.0;
    for (&value, &code) in vector.iter().zip(&codes) {
        let reconstructed = f64::from(code) * scale;
        reconstructed_squared += reconstructed * reconstructed;
        let residual = f64::from(value) - reconstructed;
        residual_squared += residual * residual;
    }
    (
        codes,
        CompactDenseVectorMetadata {
            scale,
            reconstructed_norm: reconstructed_squared.sqrt(),
            residual_norm: residual_squared.sqrt().next_up(),
            original_norm,
        },
    )
}

fn integer_dot(left: &[i8], right: &[i8]) -> i64 {
    debug_assert_eq!(left.len(), right.len());
    const MAX_I32_DOT_DIMENSION: usize = i32::MAX as usize / (127 * 127);
    if left.len() <= MAX_I32_DOT_DIMENSION {
        i64::from(
            left.iter()
                .zip(right)
                .map(|(&left, &right)| i32::from(left) * i32::from(right))
                .sum::<i32>(),
        )
    } else {
        left.iter()
            .zip(right)
            .map(|(&left, &right)| i64::from(left) * i64::from(right))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use common::counter::hardware_counter::HardwareCounterCell;

    use super::*;
    use crate::data_types::query_context::QueryContext;
    use crate::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
    use crate::entry::{ReadSegmentEntry, SegmentEntry};
    use crate::segment_constructor::simple_segment_constructor::build_simple_segment;
    use crate::types::{
        Distance, QuantizationConfig, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
        SearchParams,
    };
    use crate::vector_storage::quantized::quantized_vectors::{
        QUANTIZED_COMPACT_CERTIFICATE_PATH, QuantizedVectors, QuantizedVectorsStorageType,
    };

    #[test]
    fn default_compact_build_limit_keeps_scalar_fallback_without_extra_file() {
        let segment_dir = tempfile::tempdir().unwrap();
        let quantized_dir = tempfile::tempdir().unwrap();
        let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        for id in 0..=DEFAULT_DENSE_COMPACT_MAX_POINTS as u64 {
            segment
                .upsert_point(
                    id,
                    id.into(),
                    only_default_vector(&[id as f32, 1.0]),
                    &hardware_counter,
                )
                .unwrap();
        }
        let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
            scalar: ScalarQuantizationConfig {
                r#type: ScalarType::Int8,
                quantile: None,
                always_ram: Some(true),
            },
        });
        let quantized = QuantizedVectors::create(
            &segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .borrow(),
            &scalar_config,
            QuantizedVectorsStorageType::Immutable,
            quantized_dir.path(),
            1,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(quantized.scalar_reconstruction_table().is_some());
        assert!(quantized.compact_dense_certificate().is_none());
        assert!(
            !quantized_dir
                .path()
                .join(QUANTIZED_COMPACT_CERTIFICATE_PATH)
                .exists()
        );
    }

    #[test]
    fn exact_prefix_matches_full_scan_and_avoids_certificate_startup() {
        let segment_dir = tempfile::tempdir().unwrap();
        let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        for id in 0..8_192u64 {
            segment
                .upsert_point(
                    id,
                    (20_000 - id).into(),
                    only_default_vector(&[id as f32, 1.0]),
                    &hardware_counter,
                )
                .unwrap();
        }
        let query = vec![1.0, 0.0];
        let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
        let expected = segment
            .search(
                DEFAULT_VECTOR_NAME,
                &query_vector,
                &Default::default(),
                &Default::default(),
                None,
                20,
                Some(&SearchParams {
                    exact: true,
                    ..Default::default()
                }),
            )
            .unwrap()
            .into_iter()
            .map(|point| point.id)
            .collect::<Vec<_>>();
        let mut query_context = QueryContext::default();
        segment.fill_query_context(&mut query_context).unwrap();
        let segment_query_context = query_context.get_segment_query_context();
        let (actual, telemetry) = segment
            .with_view(|view| {
                view.with_native_dense_stream(
                    DEFAULT_VECTOR_NAME,
                    &query,
                    None,
                    64,
                    NativeDensePolicy::default(),
                    &segment_query_context,
                    |next| {
                        (0..20)
                            .map(|_| next().map(Option::unwrap))
                            .collect::<OperationResult<Vec<_>>>()
                    },
                )
            })
            .unwrap();
        assert_eq!(
            actual.into_iter().map(|point| point.id).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(telemetry.plan, Some(NativeDensePlan::ExactPrefix));
        assert_eq!(telemetry.accepted_prefix_points, 64);
        assert_eq!(telemetry.exact_prefix_fallbacks, 0);
    }

    #[test]
    fn tied_exact_prefix_falls_back_before_emitting_and_freezes_identity_order() {
        let segment_dir = tempfile::tempdir().unwrap();
        let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let mut identities = Vec::new();
        for id in 0..128u64 {
            let external = (1_000 - id).into();
            identities.push(external);
            segment
                .upsert_point(
                    id,
                    external,
                    only_default_vector(&[1.0, id as f32]),
                    &hardware_counter,
                )
                .unwrap();
        }
        identities.sort_unstable();
        identities.truncate(20);
        let mut query_context = QueryContext::default();
        segment.fill_query_context(&mut query_context).unwrap();
        let segment_query_context = query_context.get_segment_query_context();
        let (actual, telemetry) = segment
            .with_view(|view| {
                view.with_native_dense_stream(
                    DEFAULT_VECTOR_NAME,
                    &[1.0, 0.0],
                    None,
                    16,
                    NativeDensePolicy::default(),
                    &segment_query_context,
                    |next| {
                        (0..20)
                            .map(|_| next().map(Option::unwrap))
                            .collect::<OperationResult<Vec<_>>>()
                    },
                )
            })
            .unwrap();
        assert_eq!(
            actual.into_iter().map(|point| point.id).collect::<Vec<_>>(),
            identities
        );
        assert_eq!(telemetry.plan, Some(NativeDensePlan::ExactScan));
        assert_eq!(telemetry.accepted_prefix_points, 0);
        assert_eq!(telemetry.exact_prefix_fallbacks, 1);
    }

    #[test]
    fn persisted_compact_certificate_matches_segment_exact_top_k() {
        let segment_dir = tempfile::tempdir().unwrap();
        let quantized_dir = tempfile::tempdir().unwrap();
        let mut segment = build_simple_segment(segment_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        for id in 0..8_192u64 {
            segment
                .upsert_point(
                    id,
                    (20_000 - id).into(),
                    only_default_vector(&[id as f32, 1.0]),
                    &hardware_counter,
                )
                .unwrap();
        }

        let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
            scalar: ScalarQuantizationConfig {
                r#type: ScalarType::Int8,
                quantile: None,
                always_ram: Some(true),
            },
        });
        let quantized = QuantizedVectors::create(
            &segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .borrow(),
            &scalar_config,
            QuantizedVectorsStorageType::Immutable,
            quantized_dir.path(),
            1,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(
            quantized.scalar_reconstruction_table().unwrap().len(),
            8_192
        );
        assert_eq!(quantized.compact_dense_certificate().unwrap().len(), 8_192);
        drop(quantized);
        let quantized = QuantizedVectors::load(
            &scalar_config,
            &segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .borrow(),
            quantized_dir.path(),
            &AtomicBool::new(false),
        )
        .unwrap()
        .unwrap();
        *segment.vector_data[DEFAULT_VECTOR_NAME]
            .quantized_vectors
            .borrow_mut() = Some(quantized);

        let query = vec![1.0, 0.0];
        let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
        let mut expected = segment
            .search(
                DEFAULT_VECTOR_NAME,
                &query_vector,
                &Default::default(),
                &Default::default(),
                None,
                20,
                Some(&SearchParams {
                    exact: true,
                    ..Default::default()
                }),
            )
            .unwrap();
        expected.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut query_context = QueryContext::default();
        segment.fill_query_context(&mut query_context).unwrap();
        let segment_query_context = query_context.get_segment_query_context();
        let (actual, telemetry) = segment
            .with_view(|view| {
                view.with_native_dense_stream(
                    DEFAULT_VECTOR_NAME,
                    &query,
                    None,
                    0,
                    NativeDensePolicy {
                        scalar_min_points: 0,
                        compact_max_points: usize::MAX,
                        ..NativeDensePolicy::default()
                    },
                    &segment_query_context,
                    |next| {
                        (0..20)
                            .map(|_| next().map(Option::unwrap))
                            .collect::<OperationResult<Vec<_>>>()
                    },
                )
            })
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(telemetry.plan, Some(NativeDensePlan::CompactCertificate));
        assert_eq!(telemetry.native_quantized_scores, 8_192);
        assert!(telemetry.exact_scores < telemetry.eligible_points);
    }

    #[test]
    fn persisted_compact_certificate_is_exact_for_mixed_sign_dot() {
        check_mixed_sign_persisted_certificate(Distance::Dot);
    }

    #[test]
    fn persisted_compact_certificate_is_exact_for_mixed_sign_cosine() {
        check_mixed_sign_persisted_certificate(Distance::Cosine);
    }

    fn check_mixed_sign_persisted_certificate(distance: Distance) {
        const DIMENSION: usize = 16;
        const POINTS: usize = 1_024;
        const QUERIES: usize = 12;
        const TOP_K: usize = 32;

        let segment_dir = tempfile::tempdir().unwrap();
        let quantized_dir = tempfile::tempdir().unwrap();
        let mut segment = build_simple_segment(segment_dir.path(), DIMENSION, distance).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        for id in 0..POINTS {
            let vector = (0..DIMENSION)
                .map(|coordinate| {
                    let mixed = (id * 131 + coordinate * 47 + id * coordinate * 3) % 257;
                    (mixed as f32 - 128.0) / 37.0
                        + id as f32 * 0.000_001
                        + coordinate as f32 * 0.000_01
                })
                .collect::<Vec<_>>();
            segment
                .upsert_point(
                    id as u64,
                    (50_000 - id as u64).into(),
                    only_default_vector(&vector),
                    &hardware_counter,
                )
                .unwrap();
        }

        let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
            scalar: ScalarQuantizationConfig {
                r#type: ScalarType::Int8,
                quantile: None,
                always_ram: Some(true),
            },
        });
        let quantized = QuantizedVectors::create(
            &segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .borrow(),
            &scalar_config,
            QuantizedVectorsStorageType::Immutable,
            quantized_dir.path(),
            1,
            &AtomicBool::new(false),
        )
        .unwrap();
        drop(quantized);
        let quantized = QuantizedVectors::load(
            &scalar_config,
            &segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .borrow(),
            quantized_dir.path(),
            &AtomicBool::new(false),
        )
        .unwrap()
        .unwrap();
        *segment.vector_data[DEFAULT_VECTOR_NAME]
            .quantized_vectors
            .borrow_mut() = Some(quantized);

        let mut query_context = QueryContext::default();
        segment.fill_query_context(&mut query_context).unwrap();
        let segment_query_context = query_context.get_segment_query_context();
        for query_id in 0..QUERIES {
            let query = (0..DIMENSION)
                .map(|coordinate| {
                    let mixed =
                        (query_id * 73 + coordinate * 29 + query_id * coordinate * 11) % 193;
                    (mixed as f32 - 96.0) / 31.0 + coordinate as f32 * 0.000_03
                })
                .collect::<Vec<_>>();
            let query_vector: QueryVector = VectorInternal::Dense(query.clone()).into();
            let mut expected = segment
                .search(
                    DEFAULT_VECTOR_NAME,
                    &query_vector,
                    &Default::default(),
                    &Default::default(),
                    None,
                    TOP_K,
                    Some(&SearchParams {
                        exact: true,
                        ..Default::default()
                    }),
                )
                .unwrap();
            expected.sort_unstable_by(|left, right| {
                OrderedFloat(right.score)
                    .cmp(&OrderedFloat(left.score))
                    .then_with(|| left.id.cmp(&right.id))
            });

            let (actual, telemetry) = segment
                .with_view(|view| {
                    view.with_native_dense_stream(
                        DEFAULT_VECTOR_NAME,
                        &query,
                        None,
                        0,
                        NativeDensePolicy {
                            scalar_min_points: 0,
                            compact_max_points: usize::MAX,
                            ..NativeDensePolicy::default()
                        },
                        &segment_query_context,
                        |next| {
                            (0..TOP_K)
                                .map(|_| next().map(Option::unwrap))
                                .collect::<OperationResult<Vec<_>>>()
                        },
                    )
                })
                .unwrap();

            assert_eq!(actual, expected, "distance={distance:?}, query={query_id}");
            assert_eq!(telemetry.plan, Some(NativeDensePlan::CompactCertificate));
        }
    }
}
