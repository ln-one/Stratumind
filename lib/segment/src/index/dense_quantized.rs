// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact Dense ordered streams proposed by low-bit scores with residual certificates.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use common::types::PointOffsetType;
use ordered_float::OrderedFloat;

use super::dense_ball::{DenseDocument, DenseScoredPoint};

const MAX_CODE: f64 = 127.0;
const MAX_I32_DOT_DIMENSION: usize = i32::MAX as usize / (127 * 127);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseQuantizedError {
    EmptyDocuments,
    DuplicateDocument(PointOffsetType),
    InvalidDocument(PointOffsetType),
    InvalidQuery,
}

impl Display for DenseQuantizedError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDocuments => {
                formatter.write_str("Dense Quantized requires at least one document")
            }
            Self::DuplicateDocument(id) => {
                write!(formatter, "Dense Quantized document {id} occurs more than once")
            }
            Self::InvalidDocument(id) => write!(
                formatter,
                "Dense Quantized document {id} has a mismatched dimension or non-finite coordinate",
            ),
            Self::InvalidQuery => formatter.write_str(
                "Dense Quantized query must match the index dimension and contain finite coordinates",
            ),
        }
    }
}

impl Error for DenseQuantizedError {}

#[derive(Clone, Debug)]
struct QuantizedVectorMetadata {
    scale: f64,
    reconstructed_norm: f64,
    residual_norm: f64,
    original_norm: f64,
}

#[derive(Clone, Debug)]
pub struct DenseQuantizedIndex {
    documents: Vec<DenseDocument>,
    codes: Vec<i8>,
    quantized: Vec<QuantizedVectorMetadata>,
    dimension: usize,
}

impl DenseQuantizedIndex {
    pub fn build(mut documents: Vec<DenseDocument>) -> Result<Self, DenseQuantizedError> {
        let Some(first) = documents.first() else {
            return Err(DenseQuantizedError::EmptyDocuments);
        };
        let dimension = first.vector.len();
        if dimension == 0 {
            return Err(DenseQuantizedError::InvalidDocument(first.id));
        }
        let mut identities = HashSet::with_capacity(documents.len());
        for document in &documents {
            if !identities.insert(document.id) {
                return Err(DenseQuantizedError::DuplicateDocument(document.id));
            }
            if document.vector.len() != dimension
                || document.vector.iter().any(|value| !value.is_finite())
            {
                return Err(DenseQuantizedError::InvalidDocument(document.id));
            }
        }
        documents.sort_unstable_by_key(|document| document.id);
        let mut codes = Vec::with_capacity(documents.len() * dimension);
        let quantized = documents
            .iter()
            .map(|document| {
                let vector = quantize(&document.vector);
                codes.extend_from_slice(&vector.codes);
                vector.metadata
            })
            .collect();
        Ok(Self {
            documents,
            codes,
            quantized,
            dimension,
        })
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn encoded_bytes(&self) -> usize {
        self.codes.len() + self.quantized.len() * 4 * size_of::<f64>()
    }

    pub fn stream(&self, query: &[f32]) -> Result<DenseQuantizedStream<'_>, DenseQuantizedError> {
        DenseQuantizedStream::new(self, query)
    }

    /// Builds independent exact streams while traversing each document code row
    /// once for the whole query batch. Dot-product count is unchanged; physical
    /// document-major traversal and cache residency are shared across channels.
    pub fn streams_document_major<'a>(
        &'a self,
        queries: &[&[f32]],
    ) -> Result<Vec<DenseQuantizedStream<'a>>, DenseQuantizedError> {
        let prepared: Vec<_> = queries
            .iter()
            .map(|query| PreparedQuery::new(self, query))
            .collect::<Result<_, _>>()?;
        let mut bounds: Vec<Vec<PendingBound>> = queries
            .iter()
            .map(|_| Vec::with_capacity(self.document_count()))
            .collect();

        for (ordinal, (document, quantized_document)) in
            self.documents.iter().zip(&self.quantized).enumerate()
        {
            let start = ordinal * self.dimension;
            let document_codes = &self.codes[start..start + self.dimension];
            for (query, query_bounds) in prepared.iter().zip(&mut bounds) {
                query_bounds.push(PendingBound {
                    ordinal,
                    id: document.id,
                    upper_bound: query.upper_bound(self, quantized_document, document_codes),
                });
            }
        }

        Ok(prepared
            .into_iter()
            .zip(bounds)
            .map(|(query, pending_bounds)| DenseQuantizedStream {
                index: self,
                query: query.exact,
                pending_bounds: BinaryHeap::from(pending_bounds),
                pending_exact: BinaryHeap::new(),
                telemetry: DenseQuantizedTelemetry {
                    quantized_scores: self.document_count(),
                    ..Default::default()
                },
            })
            .collect())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseQuantizedTelemetry {
    pub quantized_scores: usize,
    pub exact_scores: usize,
    pub points_emitted: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingBound {
    ordinal: usize,
    id: PointOffsetType,
    upper_bound: f64,
}

impl Eq for PendingBound {}

impl Ord for PendingBound {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.upper_bound)
            .cmp(&OrderedFloat(other.upper_bound))
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for PendingBound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingExact(DenseScoredPoint);

impl Eq for PendingExact {}

impl Ord for PendingExact {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.id.cmp(&self.0.id))
    }
}

impl PartialOrd for PendingExact {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct DenseQuantizedStream<'a> {
    index: &'a DenseQuantizedIndex,
    query: Vec<f64>,
    pending_bounds: BinaryHeap<PendingBound>,
    pending_exact: BinaryHeap<PendingExact>,
    telemetry: DenseQuantizedTelemetry,
}

struct PreparedQuery {
    exact: Vec<f64>,
    quantized: QuantizedVector,
}

impl PreparedQuery {
    fn new(index: &DenseQuantizedIndex, query: &[f32]) -> Result<Self, DenseQuantizedError> {
        if query.len() != index.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(DenseQuantizedError::InvalidQuery);
        }
        Ok(Self {
            exact: query.iter().map(|value| f64::from(*value)).collect(),
            quantized: quantize(query),
        })
    }

    fn upper_bound(
        &self,
        index: &DenseQuantizedIndex,
        document: &QuantizedVectorMetadata,
        document_codes: &[i8],
    ) -> f64 {
        let query = &self.quantized.metadata;
        let integer_dot = integer_dot(&self.quantized.codes, document_codes);
        let approximate = integer_dot as f64 * query.scale * document.scale;
        let error = query.residual_norm * document.original_norm
            + query.reconstructed_norm * document.residual_norm;
        let floating_guard =
            (approximate.abs() + error + query.original_norm * document.original_norm)
                * f64::EPSILON
                * (8.0 * index.dimension as f64 + 32.0);
        (approximate + error + floating_guard).next_up()
    }
}

impl<'a> DenseQuantizedStream<'a> {
    fn new(index: &'a DenseQuantizedIndex, query: &[f32]) -> Result<Self, DenseQuantizedError> {
        let prepared = PreparedQuery::new(index, query)?;
        let mut pending_bounds = Vec::with_capacity(index.document_count());
        for (ordinal, (document, quantized_document)) in
            index.documents.iter().zip(&index.quantized).enumerate()
        {
            let start = ordinal * index.dimension;
            let document_codes = &index.codes[start..start + index.dimension];
            pending_bounds.push(PendingBound {
                ordinal,
                id: document.id,
                upper_bound: prepared.upper_bound(index, quantized_document, document_codes),
            });
        }
        Ok(Self {
            index,
            query: prepared.exact,
            pending_bounds: BinaryHeap::from(pending_bounds),
            pending_exact: BinaryHeap::new(),
            telemetry: DenseQuantizedTelemetry {
                quantized_scores: index.document_count(),
                ..Default::default()
            },
        })
    }

    pub fn telemetry(&self) -> DenseQuantizedTelemetry {
        self.telemetry
    }

    fn next_point_is_fixed(&self) -> bool {
        let Some(point) = self.pending_exact.peek() else {
            return false;
        };
        let Some(bound) = self.pending_bounds.peek() else {
            return true;
        };
        point.0.score > bound.upper_bound
            || (point.0.score == bound.upper_bound && point.0.id < bound.id)
    }

    fn rescore_next_bound(&mut self) {
        let Some(bound) = self.pending_bounds.pop() else {
            return;
        };
        let document = &self.index.documents[bound.ordinal];
        let score = document
            .vector
            .iter()
            .zip(&self.query)
            .map(|(left, right)| f64::from(*left) * *right)
            .sum::<f64>();
        assert!(
            score <= bound.upper_bound,
            "Dense quantized certificate violated for point {}: score {score}, bound {}",
            document.id,
            bound.upper_bound,
        );
        self.telemetry.exact_scores += 1;
        self.pending_exact.push(PendingExact(DenseScoredPoint {
            id: document.id,
            score,
        }));
    }
}

impl Iterator for DenseQuantizedStream<'_> {
    type Item = DenseScoredPoint;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_point_is_fixed() {
                self.telemetry.points_emitted += 1;
                return self.pending_exact.pop().map(|point| point.0);
            }
            if self.pending_bounds.is_empty() {
                let point = self.pending_exact.pop()?.0;
                self.telemetry.points_emitted += 1;
                return Some(point);
            }
            self.rescore_next_bound();
        }
    }
}

fn integer_dot(left: &[i8], right: &[i8]) -> i64 {
    debug_assert_eq!(left.len(), right.len());
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

struct QuantizedVector {
    codes: Vec<i8>,
    metadata: QuantizedVectorMetadata,
}

fn quantize(vector: &[f32]) -> QuantizedVector {
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
    QuantizedVector {
        codes,
        metadata: QuantizedVectorMetadata {
            scale,
            reconstructed_norm: reconstructed_squared.sqrt(),
            residual_norm: residual_squared.sqrt().next_up(),
            original_norm,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exhaustive(documents: &[DenseDocument], query: &[f32]) -> Vec<DenseScoredPoint> {
        let mut points: Vec<_> = documents
            .iter()
            .map(|document| DenseScoredPoint {
                id: document.id,
                score: document
                    .vector
                    .iter()
                    .zip(query)
                    .map(|(left, right)| f64::from(*left) * f64::from(*right))
                    .sum(),
            })
            .collect();
        points.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.id.cmp(&right.id))
        });
        points
    }

    #[test]
    fn exact_stream_matches_exhaustive_for_mixed_sign_vectors() {
        for dimension in [1, 3, 17, 64] {
            let documents: Vec<_> = (0..97)
                .map(|id| DenseDocument {
                    id,
                    vector: (0..dimension)
                        .map(|coordinate| {
                            let value = ((id as usize * 31 + coordinate * 17) % 101) as f32;
                            (value - 50.0) / 37.0
                        })
                        .collect(),
                })
                .collect();
            let query: Vec<_> = (0..dimension)
                .map(|coordinate| ((coordinate * 23 % 47) as f32 - 21.0) / 19.0)
                .collect();
            let index = DenseQuantizedIndex::build(documents.clone()).unwrap();

            let actual: Vec<_> = index.stream(&query).unwrap().collect();

            assert_eq!(actual, exhaustive(&documents, &query));
        }
    }

    #[test]
    fn only_rescores_overlapping_intervals_for_separated_vectors() {
        let documents: Vec<_> = (0..1_000)
            .map(|id| DenseDocument {
                id,
                vector: vec![id as f32, 1.0],
            })
            .collect();
        let index = DenseQuantizedIndex::build(documents).unwrap();
        let mut stream = index.stream(&[1.0, 0.0]).unwrap();

        let top: Vec<_> = stream.by_ref().take(20).collect();

        assert_eq!(top[0].id, 999);
        assert!(stream.telemetry().exact_scores < index.document_count());
    }

    #[test]
    fn preserves_identity_order_for_ties() {
        let documents = vec![
            DenseDocument {
                id: 7,
                vector: vec![1.0, 2.0],
            },
            DenseDocument {
                id: 3,
                vector: vec![1.0, 2.0],
            },
        ];
        let index = DenseQuantizedIndex::build(documents).unwrap();
        let points: Vec<_> = index.stream(&[1.0, 1.0]).unwrap().collect();

        assert_eq!(points[0].id, 3);
        assert_eq!(points[1].id, 7);
    }

    #[test]
    fn document_major_batch_matches_independent_streams() {
        let documents: Vec<_> = (0..257)
            .map(|id| DenseDocument {
                id,
                vector: (0..31)
                    .map(|coordinate| {
                        ((id as usize * 19 + coordinate * 43) % 127) as f32 / 63.0 - 1.0
                    })
                    .collect(),
            })
            .collect();
        let index = DenseQuantizedIndex::build(documents).unwrap();
        let queries: Vec<Vec<f32>> = (0..4)
            .map(|query| {
                (0..31)
                    .map(|coordinate| ((query * 29 + coordinate * 17) % 101) as f32 / 50.0 - 1.0)
                    .collect()
            })
            .collect();
        let references: Vec<_> = queries.iter().map(Vec::as_slice).collect();

        let actual: Vec<Vec<_>> = index
            .streams_document_major(&references)
            .unwrap()
            .into_iter()
            .map(Iterator::collect)
            .collect();
        let expected: Vec<Vec<_>> = queries
            .iter()
            .map(|query| index.stream(query).unwrap().collect())
            .collect();

        assert_eq!(actual, expected);
    }
}
