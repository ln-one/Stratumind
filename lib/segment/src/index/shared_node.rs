// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Query-independent physical Nodes shared by any number of exact channel streams.
//!
//! A channel supplies one admissible upper bound per Node and an exact scorer for
//! materialized identities. The catalog owns identity, layout, and materialization
//! state once; channel values and queues remain logically independent.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::rc::Rc;

use common::types::PointOffsetType;
use ordered_float::OrderedFloat;

use crate::common::operation_error::OperationError;
use crate::common::reciprocal_rank_fusion::{
    DynamicRrfExecution, ExactRrfStream, execute_dynamic_rrf,
};
use crate::types::ExtendedPointId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SharedNode {
    start: usize,
    end: usize,
    min_id: PointOffsetType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SharedNodeError {
    EmptyIdentityUniverse,
    ZeroBlockSize,
    DuplicateIdentity(PointOffsetType),
    EmptyChannels,
    BoundCountMismatch { expected: usize, actual: usize },
    InvalidBound { node: usize },
    InvalidScore { point: PointOffsetType },
    BoundViolation { node: usize, point: PointOffsetType },
    Fusion(String),
}

impl Display for SharedNodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyIdentityUniverse => {
                formatter.write_str("Shared Node catalog requires at least one identity")
            }
            Self::ZeroBlockSize => formatter.write_str("Shared Node block size must be positive"),
            Self::DuplicateIdentity(id) => {
                write!(formatter, "Shared Node identity {id} occurs more than once")
            }
            Self::EmptyChannels => {
                formatter.write_str("Shared Node execution requires at least one channel")
            }
            Self::BoundCountMismatch { expected, actual } => write!(
                formatter,
                "Shared Node channel supplied {actual} bounds; expected {expected}",
            ),
            Self::InvalidBound { node } => {
                write!(formatter, "Shared Node {node} has a non-finite upper bound")
            }
            Self::InvalidScore { point } => {
                write!(
                    formatter,
                    "Shared Node point {point} has a non-finite score"
                )
            }
            Self::BoundViolation { node, point } => write!(
                formatter,
                "Shared Node {node} upper bound is below exact point {point} score",
            ),
            Self::Fusion(message) => write!(formatter, "Shared Node fusion failed: {message}"),
        }
    }
}

impl Error for SharedNodeError {}

impl From<OperationError> for SharedNodeError {
    fn from(error: OperationError) -> Self {
        Self::Fusion(error.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct SharedNodeCatalog {
    identities: Vec<PointOffsetType>,
    nodes: Vec<SharedNode>,
    block_size: usize,
}

impl SharedNodeCatalog {
    pub fn build(
        identities: Vec<PointOffsetType>,
        block_size: usize,
    ) -> Result<Self, SharedNodeError> {
        if identities.is_empty() {
            return Err(SharedNodeError::EmptyIdentityUniverse);
        }
        if block_size == 0 {
            return Err(SharedNodeError::ZeroBlockSize);
        }
        let mut seen = HashSet::with_capacity(identities.len());
        for &identity in &identities {
            if !seen.insert(identity) {
                return Err(SharedNodeError::DuplicateIdentity(identity));
            }
        }
        let nodes = identities
            .chunks(block_size)
            .enumerate()
            .map(|(node, chunk)| SharedNode {
                start: node * block_size,
                end: node * block_size + chunk.len(),
                min_id: *chunk.iter().min().expect("a Node chunk is non-empty"),
            })
            .collect();
        Ok(Self {
            identities,
            nodes,
            block_size,
        })
    }

    pub fn document_count(&self) -> usize {
        self.identities.len()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn identities(&self) -> &[PointOffsetType] {
        &self.identities
    }

    pub fn node_range(&self, node: usize) -> Option<Range<usize>> {
        self.nodes.get(node).map(|node| node.start..node.end)
    }
}

pub type SharedExactScorer<'a> =
    Box<dyn Fn(usize, PointOffsetType) -> Result<Option<f64>, SharedNodeError> + 'a>;

pub struct SharedChannelPlan<'a> {
    pub node_upper_bounds: Vec<Option<f64>>,
    pub score: SharedExactScorer<'a>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedChannelTelemetry {
    pub nodes_expanded: usize,
    pub documents_scored: usize,
    pub points_emitted: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedPhysicalTelemetry {
    pub node_open_requests: usize,
    pub unique_node_materializations: usize,
    pub reused_node_materializations: usize,
    pub unique_identity_materializations: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedNChannelExecution {
    pub fusion: DynamicRrfExecution,
    pub channels: Vec<SharedChannelTelemetry>,
    pub physical: SharedPhysicalTelemetry,
}

#[derive(Debug)]
struct SharedAccessState {
    materialized: Vec<bool>,
    telemetry: SharedPhysicalTelemetry,
}

impl SharedAccessState {
    fn new(node_count: usize) -> Self {
        Self {
            materialized: vec![false; node_count],
            telemetry: SharedPhysicalTelemetry::default(),
        }
    }

    fn open(&mut self, node: usize, identity_count: usize) {
        self.telemetry.node_open_requests += 1;
        if self.materialized[node] {
            self.telemetry.reused_node_materializations += 1;
        } else {
            self.materialized[node] = true;
            self.telemetry.unique_node_materializations += 1;
            self.telemetry.unique_identity_materializations += identity_count;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingNode {
    node: usize,
    upper_bound: f64,
    min_id: PointOffsetType,
}

impl Eq for PendingNode {}

impl Ord for PendingNode {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.upper_bound)
            .cmp(&OrderedFloat(other.upper_bound))
            .then_with(|| other.min_id.cmp(&self.min_id))
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for PendingNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingPoint {
    id: PointOffsetType,
    score: f64,
}

impl Eq for PendingPoint {}

impl Ord for PendingPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.score)
            .cmp(&OrderedFloat(other.score))
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for PendingPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct SharedExactStream<'a> {
    catalog: &'a SharedNodeCatalog,
    pending_nodes: BinaryHeap<PendingNode>,
    pending_points: BinaryHeap<PendingPoint>,
    score: SharedExactScorer<'a>,
    access: Rc<RefCell<SharedAccessState>>,
    telemetry: Rc<RefCell<SharedChannelTelemetry>>,
    failed: Rc<RefCell<Option<SharedNodeError>>>,
}

impl<'a> SharedExactStream<'a> {
    fn new(
        catalog: &'a SharedNodeCatalog,
        plan: SharedChannelPlan<'a>,
        access: Rc<RefCell<SharedAccessState>>,
        telemetry: Rc<RefCell<SharedChannelTelemetry>>,
        failed: Rc<RefCell<Option<SharedNodeError>>>,
    ) -> Result<Self, SharedNodeError> {
        if plan.node_upper_bounds.len() != catalog.nodes.len() {
            return Err(SharedNodeError::BoundCountMismatch {
                expected: catalog.nodes.len(),
                actual: plan.node_upper_bounds.len(),
            });
        }
        let mut pending_nodes = BinaryHeap::new();
        for (node, upper_bound) in plan.node_upper_bounds.into_iter().enumerate() {
            let Some(upper_bound) = upper_bound else {
                continue;
            };
            if !upper_bound.is_finite() {
                return Err(SharedNodeError::InvalidBound { node });
            }
            pending_nodes.push(PendingNode {
                node,
                upper_bound,
                min_id: catalog.nodes[node].min_id,
            });
        }
        Ok(Self {
            catalog,
            pending_nodes,
            pending_points: BinaryHeap::new(),
            score: plan.score,
            access,
            telemetry,
            failed,
        })
    }

    fn next_point_is_fixed(&self) -> bool {
        let Some(point) = self.pending_points.peek() else {
            return false;
        };
        let Some(node) = self.pending_nodes.peek() else {
            return true;
        };
        point.score > node.upper_bound
            || (point.score == node.upper_bound && point.id < node.min_id)
    }

    fn expand_next_node(&mut self) {
        let Some(pending) = self.pending_nodes.pop() else {
            return;
        };
        let node = self.catalog.nodes[pending.node];
        self.access
            .borrow_mut()
            .open(pending.node, node.end - node.start);
        self.telemetry.borrow_mut().nodes_expanded += 1;
        for ordinal in node.start..node.end {
            let id = self.catalog.identities[ordinal];
            match (self.score)(ordinal, id) {
                Ok(Some(score)) if score.is_finite() && score <= pending.upper_bound => {
                    self.telemetry.borrow_mut().documents_scored += 1;
                    self.pending_points.push(PendingPoint { id, score });
                }
                Ok(Some(score)) if !score.is_finite() => {
                    *self.failed.borrow_mut() = Some(SharedNodeError::InvalidScore { point: id });
                    self.pending_nodes.clear();
                    self.pending_points.clear();
                    return;
                }
                Ok(Some(_)) => {
                    *self.failed.borrow_mut() = Some(SharedNodeError::BoundViolation {
                        node: pending.node,
                        point: id,
                    });
                    self.pending_nodes.clear();
                    self.pending_points.clear();
                    return;
                }
                Ok(None) => {}
                Err(error) => {
                    *self.failed.borrow_mut() = Some(error);
                    self.pending_nodes.clear();
                    self.pending_points.clear();
                    return;
                }
            }
        }
    }
}

impl Iterator for SharedExactStream<'_> {
    type Item = ExtendedPointId;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.failed.borrow().is_some() {
                return None;
            }
            if self.next_point_is_fixed() {
                self.telemetry.borrow_mut().points_emitted += 1;
                return self
                    .pending_points
                    .pop()
                    .map(|point| ExtendedPointId::from(u64::from(point.id)));
            }
            if self.pending_nodes.is_empty() {
                let point = self.pending_points.pop()?;
                self.telemetry.borrow_mut().points_emitted += 1;
                return Some(ExtendedPointId::from(u64::from(point.id)));
            }
            self.expand_next_node();
        }
    }
}

pub fn execute_shared_dynamic_rrf<'a>(
    catalog: &'a SharedNodeCatalog,
    plans: Vec<SharedChannelPlan<'a>>,
    top_k: usize,
    rrf_k: usize,
    weights: Option<&[f32]>,
) -> Result<SharedNChannelExecution, SharedNodeError> {
    if plans.is_empty() {
        return Err(SharedNodeError::EmptyChannels);
    }
    let access = Rc::new(RefCell::new(SharedAccessState::new(catalog.node_count())));
    let failed = Rc::new(RefCell::new(None));
    let channel_telemetry: Vec<_> = (0..plans.len())
        .map(|_| Rc::new(RefCell::new(SharedChannelTelemetry::default())))
        .collect();
    let mut sources = Vec::<ExactRrfStream<'a>>::with_capacity(plans.len());
    for (plan, telemetry) in plans.into_iter().zip(&channel_telemetry) {
        sources.push(Box::new(SharedExactStream::new(
            catalog,
            plan,
            Rc::clone(&access),
            Rc::clone(telemetry),
            Rc::clone(&failed),
        )?));
    }
    let fusion = execute_dynamic_rrf(sources, top_k, rrf_k, weights)?;
    if let Some(error) = failed.borrow_mut().take() {
        return Err(error);
    }
    let channels = channel_telemetry
        .iter()
        .map(|telemetry| *telemetry.borrow())
        .collect();
    let physical = access.borrow().telemetry;
    Ok(SharedNChannelExecution {
        fusion,
        channels,
        physical,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::reciprocal_rank_fusion::{DEFAULT_RRF_K, exact_rrf_scoring};
    use crate::types::ScoredPoint;

    fn scored_point(id: PointOffsetType) -> ScoredPoint {
        ScoredPoint {
            id: ExtendedPointId::from(u64::from(id)),
            version: 0,
            score: 0.0,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    fn plan<'a>(catalog: &SharedNodeCatalog, scores: &'a [f64]) -> SharedChannelPlan<'a> {
        let node_upper_bounds = catalog
            .nodes
            .iter()
            .map(|node| {
                Some(
                    scores[node.start..node.end]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                )
            })
            .collect();
        SharedChannelPlan {
            node_upper_bounds,
            score: Box::new(move |ordinal, _| Ok(Some(scores[ordinal]))),
        }
    }

    fn exhaustive(scores: &[Vec<f64>], top_k: usize) -> Vec<ExtendedPointId> {
        let rankings = scores
            .iter()
            .map(|channel| {
                let mut ranking: Vec<_> = channel
                    .iter()
                    .enumerate()
                    .map(|(id, score)| (id as PointOffsetType, *score))
                    .collect();
                ranking.sort_unstable_by(|left, right| {
                    OrderedFloat(right.1)
                        .cmp(&OrderedFloat(left.1))
                        .then_with(|| left.0.cmp(&right.0))
                });
                ranking
                    .into_iter()
                    .map(|(id, _)| scored_point(id))
                    .collect()
            })
            .collect();
        exact_rrf_scoring(rankings, DEFAULT_RRF_K, None)
            .unwrap()
            .into_iter()
            .take(top_k)
            .map(|point| point.id)
            .collect()
    }

    #[test]
    fn n_channel_streams_match_exhaustive_rrf() {
        for channel_count in 1..=8 {
            let identities = (0..97).collect();
            let catalog = SharedNodeCatalog::build(identities, 11).unwrap();
            let scores: Vec<Vec<f64>> = (0..channel_count)
                .map(|channel| {
                    (0..catalog.document_count())
                        .map(|id| {
                            let mixed = id * (channel * 14 + 17) + channel * 31;
                            ((mixed % 101) as f64) + (id as f64 / 10_000.0)
                        })
                        .collect()
                })
                .collect();
            let plans = scores
                .iter()
                .map(|channel| plan(&catalog, channel))
                .collect();

            let actual =
                execute_shared_dynamic_rrf(&catalog, plans, 20, DEFAULT_RRF_K, None).unwrap();

            assert_eq!(actual.fusion.point_ids, exhaustive(&scores, 20));
        }
    }

    #[test]
    fn physical_node_materialization_is_shared_across_channels() {
        let catalog = SharedNodeCatalog::build((0..32).collect(), 8).unwrap();
        let scores = [
            (0..32).map(f64::from).collect::<Vec<_>>(),
            (0..32).map(|id| f64::from(31 - id)).collect::<Vec<_>>(),
            (0..32)
                .map(|id| f64::from((id * 7) % 32))
                .collect::<Vec<_>>(),
        ];
        let plans = scores
            .iter()
            .map(|channel| plan(&catalog, channel))
            .collect();

        let execution =
            execute_shared_dynamic_rrf(&catalog, plans, 32, DEFAULT_RRF_K, None).unwrap();

        assert_eq!(execution.physical.unique_node_materializations, 4);
        assert_eq!(execution.physical.node_open_requests, 12);
        assert_eq!(execution.physical.reused_node_materializations, 8);
        assert_eq!(execution.physical.unique_identity_materializations, 32);
    }

    #[test]
    fn rejects_a_non_admissible_node_value() {
        let catalog = SharedNodeCatalog::build(vec![0, 1], 2).unwrap();
        let scores = [2.0, 1.0];
        let result = execute_shared_dynamic_rrf(
            &catalog,
            vec![SharedChannelPlan {
                node_upper_bounds: vec![Some(1.5)],
                score: Box::new(|ordinal, _| Ok(Some(scores[ordinal]))),
            }],
            1,
            DEFAULT_RRF_K,
            None,
        );

        assert!(matches!(
            result,
            Err(SharedNodeError::BoundViolation { .. })
        ));
    }
}
