// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Recoverable exact Dense stream seeded by an approximate identity proposal.
//!
//! Proposal identities affect only the initial exact lower bound. Unopened work
//! remains guarded by [`DenseBallIndex`] bounds, so a weak or empty proposal
//! safely falls back to the same exhaustive order.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use common::types::PointOffsetType;
use ordered_float::OrderedFloat;

use super::dense_ball::{
    DenseBallError, DenseBallIndex, DenseBallNodeContents, DenseDocument, DenseScoredPoint,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseProposalCertifiedError {
    DenseBall(DenseBallError),
    ZeroBatchSize,
    UnknownProposal(PointOffsetType),
}

impl Display for DenseProposalCertifiedError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DenseBall(error) => Display::fmt(error, formatter),
            Self::ZeroBatchSize => {
                formatter.write_str("Dense proposal certificate batch size must be positive")
            }
            Self::UnknownProposal(id) => {
                write!(
                    formatter,
                    "Dense proposal identity {id} is outside the index"
                )
            }
        }
    }
}

impl Error for DenseProposalCertifiedError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::DenseBall(error) => Some(error),
            Self::ZeroBatchSize | Self::UnknownProposal(_) => None,
        }
    }
}

impl From<DenseBallError> for DenseProposalCertifiedError {
    fn from(error: DenseBallError) -> Self {
        Self::DenseBall(error)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseProposalCertifiedTelemetry {
    pub proposal_identities_requested: usize,
    pub proposal_identities_scored: usize,
    pub proposal_identities_reused: usize,
    pub bound_evaluations: usize,
    pub internal_nodes_expanded: usize,
    pub leaf_nodes_expanded: usize,
    pub exact_scores: usize,
    pub points_emitted: usize,
    pub certified_batches: usize,
    pub max_pending_nodes: usize,
    pub max_pending_candidates: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingNode {
    node_index: usize,
    min_id: PointOffsetType,
    upper_bound: OrderedFloat<f64>,
}

impl Ord for PendingNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound
            .cmp(&other.upper_bound)
            .then_with(|| other.min_id.cmp(&self.min_id))
            .then_with(|| other.node_index.cmp(&self.node_index))
    }
}

impl PartialOrd for PendingNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ExactCandidate(DenseScoredPoint);

impl Eq for ExactCandidate {}

impl Ord for ExactCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.id.cmp(&self.0.id))
    }
}

impl PartialOrd for ExactCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct DenseProposalCertifiedStream<'a> {
    index: &'a DenseBallIndex,
    query: Vec<f64>,
    query_norm: f64,
    batch_size: usize,
    proposed: HashSet<PointOffsetType>,
    pending_nodes: BinaryHeap<PendingNode>,
    pending_candidates: BinaryHeap<ExactCandidate>,
    certified: VecDeque<DenseScoredPoint>,
    telemetry: DenseProposalCertifiedTelemetry,
}

impl<'a> DenseProposalCertifiedStream<'a> {
    pub fn new(
        index: &'a DenseBallIndex,
        query: &[f32],
        proposal_ids: &[PointOffsetType],
        batch_size: usize,
    ) -> Result<Self, DenseProposalCertifiedError> {
        if batch_size == 0 {
            return Err(DenseProposalCertifiedError::ZeroBatchSize);
        }
        let (query, query_norm) = index.query_components(query)?;
        let mut proposed = HashSet::with_capacity(proposal_ids.len());
        let mut pending_candidates = BinaryHeap::with_capacity(proposal_ids.len());
        for &id in proposal_ids {
            if !proposed.insert(id) {
                continue;
            }
            let document = index
                .document(id)
                .ok_or(DenseProposalCertifiedError::UnknownProposal(id))?;
            pending_candidates.push(ExactCandidate(score(document, &query)));
        }

        let mut pending_nodes = BinaryHeap::with_capacity(index.root_indices().len());
        for &node_index in index.root_indices() {
            let bound = index.node_bound(node_index, &query, query_norm);
            pending_nodes.push(PendingNode {
                node_index: bound.node_index,
                min_id: bound.min_id,
                upper_bound: OrderedFloat(bound.upper_bound),
            });
        }
        let telemetry = DenseProposalCertifiedTelemetry {
            proposal_identities_requested: proposal_ids.len(),
            proposal_identities_scored: proposed.len(),
            bound_evaluations: pending_nodes.len(),
            exact_scores: proposed.len(),
            max_pending_nodes: pending_nodes.len(),
            max_pending_candidates: pending_candidates.len(),
            ..Default::default()
        };

        Ok(Self {
            index,
            query,
            query_norm,
            batch_size,
            proposed,
            pending_nodes,
            pending_candidates,
            certified: VecDeque::with_capacity(batch_size),
            telemetry,
        })
    }

    pub fn telemetry(&self) -> DenseProposalCertifiedTelemetry {
        self.telemetry
    }

    fn expand_next_node(&mut self) {
        let pending = self
            .pending_nodes
            .pop()
            .expect("node expansion requires work");
        match self.index.node_contents(pending.node_index) {
            DenseBallNodeContents::Internal(children) => {
                self.telemetry.internal_nodes_expanded += 1;
                self.telemetry.bound_evaluations += children.len();
                for &child_index in children {
                    let bound = self
                        .index
                        .node_bound(child_index, &self.query, self.query_norm);
                    self.pending_nodes.push(PendingNode {
                        node_index: bound.node_index,
                        min_id: bound.min_id,
                        upper_bound: OrderedFloat(bound.upper_bound),
                    });
                }
                self.telemetry.max_pending_nodes = self
                    .telemetry
                    .max_pending_nodes
                    .max(self.pending_nodes.len());
            }
            DenseBallNodeContents::Leaf(documents) => {
                self.telemetry.leaf_nodes_expanded += 1;
                for document in documents {
                    if self.proposed.contains(&document.id) {
                        self.telemetry.proposal_identities_reused += 1;
                        continue;
                    }
                    let candidate = score(document, &self.query);
                    assert!(
                        candidate.score <= pending.upper_bound.0,
                        "Dense proposal certificate violated its leaf bound"
                    );
                    self.pending_candidates.push(ExactCandidate(candidate));
                    self.telemetry.exact_scores += 1;
                }
                self.telemetry.max_pending_candidates = self
                    .telemetry
                    .max_pending_candidates
                    .max(self.pending_candidates.len());
            }
        }
    }

    fn certify_next_batch(&mut self) -> bool {
        loop {
            let count = self.batch_size.min(self.pending_candidates.len());
            if count > 0 && (count == self.batch_size || self.pending_nodes.is_empty()) {
                let mut leading = Vec::with_capacity(count);
                for _ in 0..count {
                    leading.push(
                        self.pending_candidates
                            .pop()
                            .expect("candidate count was checked"),
                    );
                }
                let fixed = self.pending_nodes.peek().is_none_or(|node| {
                    let boundary = leading.last().expect("leading batch is non-empty").0;
                    boundary.score > node.upper_bound.0
                        || (boundary.score == node.upper_bound.0 && boundary.id < node.min_id)
                });
                if fixed {
                    self.certified
                        .extend(leading.into_iter().map(|entry| entry.0));
                    self.telemetry.certified_batches += 1;
                    return true;
                }
                self.pending_candidates.extend(leading);
            }

            if self.pending_nodes.is_empty() {
                return false;
            }
            self.expand_next_node();
        }
    }
}

impl Iterator for DenseProposalCertifiedStream<'_> {
    type Item = DenseScoredPoint;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(point) = self.certified.pop_front() {
            self.telemetry.points_emitted += 1;
            return Some(point);
        }
        if !self.certify_next_batch() {
            return None;
        }
        let point = self.certified.pop_front();
        self.telemetry.points_emitted += usize::from(point.is_some());
        point
    }
}

fn score(document: &DenseDocument, query: &[f64]) -> DenseScoredPoint {
    DenseScoredPoint {
        id: document.id,
        score: document
            .vector
            .iter()
            .zip(query)
            .map(|(coordinate, query)| f64::from(*coordinate) * *query)
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    fn documents() -> Vec<DenseDocument> {
        vec![
            DenseDocument {
                id: 7,
                vector: vec![1.0, 0.0],
            },
            DenseDocument {
                id: 3,
                vector: vec![0.9, 0.1],
            },
            DenseDocument {
                id: 11,
                vector: vec![0.0, 1.0],
            },
            DenseDocument {
                id: 5,
                vector: vec![-1.0, 0.0],
            },
            DenseDocument {
                id: 13,
                vector: vec![0.5, 0.5],
            },
        ]
    }

    fn exhaustive(mut documents: Vec<DenseDocument>, query: &[f32]) -> Vec<DenseScoredPoint> {
        let query: Vec<_> = query.iter().map(|value| f64::from(*value)).collect();
        let mut scored: Vec<_> = documents
            .drain(..)
            .map(|document| score(&document, &query))
            .collect();
        scored.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.id.cmp(&right.id))
        });
        scored
    }

    #[test]
    fn empty_and_imperfect_proposals_preserve_exhaustive_order() {
        let documents = documents();
        let index =
            DenseBallIndex::build_hierarchical_clustered(documents.clone(), 2, 2, 3).unwrap();
        let query = [1.0, 0.25];
        let expected = exhaustive(documents, &query);

        for proposals in [&[][..], &[11, 5][..], &[7, 3, 13][..]] {
            let actual: Vec<_> = DenseProposalCertifiedStream::new(&index, &query, proposals, 2)
                .unwrap()
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn duplicate_proposals_are_scored_once_and_reused_at_leaf_expansion() {
        let index = DenseBallIndex::build(documents(), 2).unwrap();
        let mut stream =
            DenseProposalCertifiedStream::new(&index, &[1.0, 0.25], &[7, 7], 2).unwrap();
        let actual: Vec<_> = stream.by_ref().collect();
        assert_eq!(actual.len(), 5);
        let telemetry = stream.telemetry();
        assert_eq!(telemetry.proposal_identities_requested, 2);
        assert_eq!(telemetry.proposal_identities_scored, 1);
        assert_eq!(telemetry.proposal_identities_reused, 1);
        assert_eq!(telemetry.exact_scores, 5);
    }

    #[test]
    fn rejects_unknown_proposal_and_zero_batch() {
        let index = DenseBallIndex::build(documents(), 2).unwrap();
        assert_eq!(
            DenseProposalCertifiedStream::new(&index, &[1.0, 0.25], &[99], 2)
                .err()
                .unwrap(),
            DenseProposalCertifiedError::UnknownProposal(99)
        );
        assert_eq!(
            DenseProposalCertifiedStream::new(&index, &[1.0, 0.25], &[], 0)
                .err()
                .unwrap(),
            DenseProposalCertifiedError::ZeroBatchSize
        );
    }

    #[test]
    fn randomized_proposals_and_batches_match_exhaustive_order() {
        let mut rng = StdRng::seed_from_u64(0x5052_4f50_4f53_414c);
        for case in 0..200 {
            let dimension = rng.random_range(2usize..=10);
            let document_count = rng.random_range(4usize..=48);
            let documents: Vec<_> = (0..document_count)
                .map(|id| DenseDocument {
                    id: id as PointOffsetType,
                    vector: (0..dimension)
                        .map(|_| rng.random_range(-1.0f32..=1.0))
                        .collect(),
                })
                .collect();
            let mut query: Vec<_> = (0..dimension)
                .map(|_| rng.random_range(-1.0f32..=1.0))
                .collect();
            if query.iter().all(|value| *value == 0.0) {
                query[0] = 1.0;
            }
            let proposal_count = rng.random_range(0..=document_count);
            let proposals: Vec<_> = (0..proposal_count)
                .map(|_| rng.random_range(0..document_count) as PointOffsetType)
                .collect();
            let batch_size = rng.random_range(1..=10);
            let expected = exhaustive(documents.clone(), &query);
            let index = DenseBallIndex::build_hierarchical_clustered(
                documents,
                rng.random_range(1..=8),
                rng.random_range(2..=4),
                2,
            )
            .unwrap();
            let actual: Vec<_> =
                DenseProposalCertifiedStream::new(&index, &query, &proposals, batch_size)
                    .unwrap()
                    .collect();
            assert_eq!(actual, expected, "case {case}");
        }
    }
}
