// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact Dense stream backed by Qdrant's native Scalar quantized scorer.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use ordered_float::OrderedFloat;
use quantization::ScalarReconstructionStats;

use super::dense_ball::{DenseDocument, DenseScoredPoint};
use super::dense_quantized::DenseQuantizedError;
use crate::common::operation_error::OperationResult;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;

#[derive(Clone, Debug)]
pub struct DenseScalarCertifiedIndex {
    documents: Vec<DenseDocument>,
    reconstruction: Vec<ScalarReconstructionStats>,
    dimension: usize,
}

impl DenseScalarCertifiedIndex {
    pub fn build(
        quantized: &QuantizedVectors,
        documents: Vec<DenseDocument>,
    ) -> Result<Self, DenseQuantizedError> {
        let Some(first) = documents.first() else {
            return Err(DenseQuantizedError::EmptyDocuments);
        };
        let dimension = first.vector.len();
        let mut reconstruction = Vec::with_capacity(documents.len());
        for (ordinal, document) in documents.iter().enumerate() {
            if document.id as usize != ordinal
                || document.vector.len() != dimension
                || document.vector.iter().any(|value| !value.is_finite())
            {
                return Err(DenseQuantizedError::InvalidDocument(document.id));
            }
            reconstruction.push(
                quantized
                    .scalar_reconstruction_stats(document.id, &document.vector)
                    .ok_or(DenseQuantizedError::InvalidDocument(document.id))?,
            );
        }
        Ok(Self {
            documents,
            reconstruction,
            dimension,
        })
    }

    pub fn stream<'a>(
        &'a self,
        quantized: &'a QuantizedVectors,
        query: &[f32],
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<DenseScalarCertifiedStream<'a>> {
        DenseScalarCertifiedStream::new(self, quantized, query, hardware_counter)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseScalarCertifiedTelemetry {
    pub native_quantized_scores: usize,
    pub exact_scores: usize,
    pub points_emitted: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bound {
    ordinal: usize,
    id: PointOffsetType,
    value: f64,
}

impl Eq for Bound {}
impl Ord for Bound {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.value)
            .cmp(&OrderedFloat(other.value))
            .then_with(|| other.id.cmp(&self.id))
    }
}
impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Exact(DenseScoredPoint);

impl Eq for Exact {}
impl Ord for Exact {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.id.cmp(&self.0.id))
    }
}
impl PartialOrd for Exact {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct DenseScalarCertifiedStream<'a> {
    index: &'a DenseScalarCertifiedIndex,
    query: Vec<f64>,
    bounds: BinaryHeap<Bound>,
    exact: BinaryHeap<Exact>,
    telemetry: DenseScalarCertifiedTelemetry,
}

impl<'a> DenseScalarCertifiedStream<'a> {
    fn new(
        index: &'a DenseScalarCertifiedIndex,
        quantized: &'a QuantizedVectors,
        query: &[f32],
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self> {
        if query.len() != index.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(
                crate::common::operation_error::OperationError::validation_error(
                    "Dense Scalar certificate received an invalid Query",
                ),
            );
        }
        let query_stats = quantized
            .scalar_query_reconstruction_stats(query)
            .ok_or_else(|| {
                crate::common::operation_error::OperationError::validation_error(
                    "Dense Scalar certificate requires Scalar quantization",
                )
            })?;
        let ids: Vec<_> = index.documents.iter().map(|document| document.id).collect();
        let mut approximate = vec![0.0f32; ids.len()];
        quantized
            .raw_scorer(query.to_vec().into(), hardware_counter)?
            .score_points(&ids, &mut approximate);
        let mut bounds = BinaryHeap::with_capacity(ids.len());
        for (ordinal, ((document, document_stats), approximate)) in index
            .documents
            .iter()
            .zip(&index.reconstruction)
            .zip(approximate)
            .enumerate()
        {
            let error = query_stats.residual_norm * document_stats.original_norm
                + query_stats.reconstructed_norm * document_stats.residual_norm;
            let scale = f64::from(approximate).abs()
                + error
                + query_stats.original_norm * document_stats.original_norm;
            let gamma = f64::from(f32::EPSILON) * (8.0 * index.dimension as f64 + 64.0);
            let floating_guard = scale * gamma / (1.0 - gamma);
            bounds.push(Bound {
                ordinal,
                id: document.id,
                value: (f64::from(approximate) + error + floating_guard).next_up(),
            });
        }
        Ok(Self {
            index,
            query: query.iter().map(|value| f64::from(*value)).collect(),
            bounds,
            exact: BinaryHeap::new(),
            telemetry: DenseScalarCertifiedTelemetry {
                native_quantized_scores: ids.len(),
                ..Default::default()
            },
        })
    }

    pub fn telemetry(&self) -> DenseScalarCertifiedTelemetry {
        self.telemetry
    }

    fn fixed(&self) -> bool {
        let Some(exact) = self.exact.peek() else {
            return false;
        };
        let Some(bound) = self.bounds.peek() else {
            return true;
        };
        exact.0.score > bound.value || (exact.0.score == bound.value && exact.0.id < bound.id)
    }

    fn rescore(&mut self) {
        let bound = self.bounds.pop().expect("rescore requires a bound");
        let document = &self.index.documents[bound.ordinal];
        let score = document
            .vector
            .iter()
            .zip(&self.query)
            .map(|(left, right)| f64::from(*left) * *right)
            .sum::<f64>();
        assert!(
            score <= bound.value,
            "Scalar reconstruction certificate violated"
        );
        self.telemetry.exact_scores += 1;
        self.exact.push(Exact(DenseScoredPoint {
            id: document.id,
            score,
        }));
    }
}

impl Iterator for DenseScalarCertifiedStream<'_> {
    type Item = DenseScoredPoint;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.fixed() {
                self.telemetry.points_emitted += 1;
                return self.exact.pop().map(|point| point.0);
            }
            if self.bounds.is_empty() {
                let point = self.exact.pop()?.0;
                self.telemetry.points_emitted += 1;
                return Some(point);
            }
            self.rescore();
        }
    }
}
