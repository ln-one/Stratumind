// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::cell::Cell;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::block_max::{
    BlockMaxError, BlockMaxIndex, BlockMaxStream, BlockMaxTelemetry, SparseDocument,
};

use super::dense_ball::{
    DenseBallError, DenseBallIndex, DenseBallStream, DenseBallTelemetry, DenseDocument,
};
use crate::common::operation_error::OperationError;
use crate::common::reciprocal_rank_fusion::{
    DynamicRrfExecution, ExactRrfStream, execute_dynamic_rrf,
};
use crate::types::ExtendedPointId;

#[derive(Debug)]
pub enum HybridExactError {
    IdentityUniverseMismatch,
    Sparse(BlockMaxError),
    Dense(DenseBallError),
    Fusion(OperationError),
}

impl Display for HybridExactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdentityUniverseMismatch => formatter.write_str(
                "Hybrid Exact Sparse and Dense documents must have the same point identity universe",
            ),
            Self::Sparse(error) => write!(formatter, "Hybrid Exact Sparse index failed: {error}"),
            Self::Dense(error) => write!(formatter, "Hybrid Exact Dense index failed: {error}"),
            Self::Fusion(error) => write!(formatter, "Hybrid Exact fusion failed: {error}"),
        }
    }
}

impl Error for HybridExactError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IdentityUniverseMismatch => None,
            Self::Sparse(error) => Some(error),
            Self::Dense(error) => Some(error),
            Self::Fusion(error) => Some(error),
        }
    }
}

impl From<BlockMaxError> for HybridExactError {
    fn from(error: BlockMaxError) -> Self {
        Self::Sparse(error)
    }
}

impl From<DenseBallError> for HybridExactError {
    fn from(error: DenseBallError) -> Self {
        Self::Dense(error)
    }
}

impl From<OperationError> for HybridExactError {
    fn from(error: OperationError) -> Self {
        Self::Fusion(error)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HybridExactTelemetry {
    pub sparse: BlockMaxTelemetry,
    pub dense: DenseBallTelemetry,
}

#[derive(Debug)]
pub struct HybridExactSearchResult {
    pub execution: DynamicRrfExecution,
    pub telemetry: HybridExactTelemetry,
}

#[derive(Clone, Debug)]
pub struct HybridExactIndex {
    sparse: BlockMaxIndex,
    dense: DenseBallIndex,
}

impl HybridExactIndex {
    pub fn build(
        sparse_documents: Vec<SparseDocument>,
        dense_documents: Vec<DenseDocument>,
        sparse_block_size: usize,
        dense_block_size: usize,
    ) -> Result<Self, HybridExactError> {
        let mut sparse_ids: Vec<_> = sparse_documents
            .iter()
            .map(|document| document.id)
            .collect();
        let mut dense_ids: Vec<_> = dense_documents.iter().map(|document| document.id).collect();
        sparse_ids.sort_unstable();
        dense_ids.sort_unstable();
        if sparse_ids != dense_ids {
            return Err(HybridExactError::IdentityUniverseMismatch);
        }

        Ok(Self {
            sparse: BlockMaxIndex::build(sparse_documents, sparse_block_size)?,
            dense: DenseBallIndex::build(dense_documents, dense_block_size)?,
        })
    }

    pub fn build_dense_clustered(
        sparse_documents: Vec<SparseDocument>,
        dense_documents: Vec<DenseDocument>,
        sparse_block_size: usize,
        dense_block_size: usize,
        dense_clustering_iterations: usize,
    ) -> Result<Self, HybridExactError> {
        let mut sparse_ids: Vec<_> = sparse_documents
            .iter()
            .map(|document| document.id)
            .collect();
        let mut dense_ids: Vec<_> = dense_documents.iter().map(|document| document.id).collect();
        sparse_ids.sort_unstable();
        dense_ids.sort_unstable();
        if sparse_ids != dense_ids {
            return Err(HybridExactError::IdentityUniverseMismatch);
        }

        Ok(Self {
            sparse: BlockMaxIndex::build(sparse_documents, sparse_block_size)?,
            dense: DenseBallIndex::build_clustered(
                dense_documents,
                dense_block_size,
                dense_clustering_iterations,
            )?,
        })
    }

    pub fn build_dense_hierarchical(
        sparse_documents: Vec<SparseDocument>,
        dense_documents: Vec<DenseDocument>,
        sparse_block_size: usize,
        dense_leaf_size: usize,
        dense_branch_factor: usize,
        dense_clustering_iterations: usize,
    ) -> Result<Self, HybridExactError> {
        let mut sparse_ids: Vec<_> = sparse_documents
            .iter()
            .map(|document| document.id)
            .collect();
        let mut dense_ids: Vec<_> = dense_documents.iter().map(|document| document.id).collect();
        sparse_ids.sort_unstable();
        dense_ids.sort_unstable();
        if sparse_ids != dense_ids {
            return Err(HybridExactError::IdentityUniverseMismatch);
        }

        Ok(Self {
            sparse: BlockMaxIndex::build(sparse_documents, sparse_block_size)?,
            dense: DenseBallIndex::build_hierarchical_clustered(
                dense_documents,
                dense_leaf_size,
                dense_branch_factor,
                dense_clustering_iterations,
            )?,
        })
    }

    pub fn document_count(&self) -> usize {
        self.sparse.document_count()
    }

    pub fn sparse_index(&self) -> &BlockMaxIndex {
        &self.sparse
    }

    pub fn dense_index(&self) -> &DenseBallIndex {
        &self.dense
    }

    pub fn search(
        &self,
        sparse_query: RemappedSparseVector,
        dense_query: &[f32],
        top_k: usize,
        rrf_k: usize,
        weights: Option<&[f32]>,
    ) -> Result<HybridExactSearchResult, HybridExactError> {
        let sparse_telemetry = Rc::new(Cell::new(BlockMaxTelemetry::default()));
        let dense_telemetry = Rc::new(Cell::new(DenseBallTelemetry::default()));
        let sources: Vec<ExactRrfStream<'_>> = vec![
            Box::new(SparseIdentityStream {
                inner: self.sparse.stream(sparse_query)?,
                telemetry: Rc::clone(&sparse_telemetry),
            }),
            Box::new(DenseIdentityStream {
                inner: self.dense.stream(dense_query)?,
                telemetry: Rc::clone(&dense_telemetry),
            }),
        ];
        let execution = execute_dynamic_rrf(sources, top_k, rrf_k, weights)?;

        Ok(HybridExactSearchResult {
            execution,
            telemetry: HybridExactTelemetry {
                sparse: sparse_telemetry.get(),
                dense: dense_telemetry.get(),
            },
        })
    }
}

struct SparseIdentityStream<'a> {
    inner: BlockMaxStream<'a>,
    telemetry: Rc<Cell<BlockMaxTelemetry>>,
}

impl Iterator for SparseIdentityStream<'_> {
    type Item = ExtendedPointId;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry.set(self.inner.telemetry());
        point.map(|point| ExtendedPointId::from(u64::from(point.idx)))
    }
}

struct DenseIdentityStream<'a> {
    inner: DenseBallStream<'a>,
    telemetry: Rc<Cell<DenseBallTelemetry>>,
}

impl Iterator for DenseIdentityStream<'_> {
    type Item = ExtendedPointId;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry.set(self.inner.telemetry());
        point.map(|point| ExtendedPointId::from(u64::from(point.id)))
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;
    use crate::common::reciprocal_rank_fusion::{DEFAULT_RRF_K, exact_rrf_scoring};
    use crate::types::ScoredPoint;

    fn documents() -> (Vec<SparseDocument>, Vec<DenseDocument>) {
        let sparse = (0..64)
            .map(|id| SparseDocument {
                id,
                vector: RemappedSparseVector {
                    indices: vec![0, 1],
                    values: vec![1.0, (id % 9) as f32],
                },
            })
            .collect();
        let dense = (0..64)
            .map(|id| DenseDocument {
                id,
                vector: vec![(id % 7) as f32, (63 - id) as f32],
            })
            .collect();
        (sparse, dense)
    }

    fn scored_point(id: ExtendedPointId) -> ScoredPoint {
        ScoredPoint {
            id,
            version: 0,
            score: 0.0,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    #[test]
    fn rejects_mismatched_identity_universes() {
        let (sparse, mut dense) = documents();
        dense.pop();

        let result = HybridExactIndex::build(sparse, dense, 8, 8);

        assert!(matches!(
            result,
            Err(HybridExactError::IdentityUniverseMismatch)
        ));
    }

    #[test]
    fn single_request_matches_exhaustive_hybrid_rrf() {
        let (sparse_documents, dense_documents) = documents();
        let index =
            HybridExactIndex::build(sparse_documents.clone(), dense_documents.clone(), 8, 8)
                .unwrap();
        let sparse_query = RemappedSparseVector {
            indices: vec![0, 1],
            values: vec![1.0, 2.0],
        };
        let dense_query = vec![1.0, -0.25];

        let mut sparse_ranking: Vec<_> = sparse_documents
            .iter()
            .filter_map(|document| {
                document
                    .vector
                    .score(&sparse_query)
                    .filter(|score| *score != 0.0)
                    .map(|score| (document.id, f64::from(score)))
            })
            .collect();
        sparse_ranking.sort_unstable_by(|left, right| {
            OrderedFloat(right.1)
                .cmp(&OrderedFloat(left.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut dense_ranking: Vec<_> = dense_documents
            .iter()
            .map(|document| {
                let score = document
                    .vector
                    .iter()
                    .zip(&dense_query)
                    .map(|(coordinate, query)| f64::from(*coordinate) * f64::from(*query))
                    .sum::<f64>();
                (document.id, score)
            })
            .collect();
        dense_ranking.sort_unstable_by(|left, right| {
            OrderedFloat(right.1)
                .cmp(&OrderedFloat(left.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        let responses = vec![sparse_ranking, dense_ranking]
            .into_iter()
            .map(|ranking| {
                ranking
                    .into_iter()
                    .map(|(id, _)| scored_point(ExtendedPointId::from(u64::from(id))))
                    .collect()
            })
            .collect();
        let expected: Vec<_> = exact_rrf_scoring(responses, DEFAULT_RRF_K, None)
            .unwrap()
            .into_iter()
            .take(10)
            .map(|point| point.id)
            .collect();

        let actual = index
            .search(sparse_query, &dense_query, 10, DEFAULT_RRF_K, None)
            .unwrap();

        assert_eq!(actual.execution.point_ids, expected);
        assert_eq!(actual.execution.point_ids.len(), 10);
        assert!(actual.telemetry.sparse.documents_evaluated > 0);
        assert!(actual.telemetry.dense.documents_evaluated > 0);
    }
}
