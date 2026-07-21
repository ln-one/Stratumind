use std::env;

use segment::index::exact_composition::{ExactCompositionNode, ExactCompositionPlan};
use segment::types::ExtendedPointId;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IsolatedExecutionMode {
    Dynamic,
    Exhaustive,
}

impl IsolatedExecutionMode {
    pub fn from_env() -> Option<Self> {
        match env::var("SPECTRA_ISOLATED_EXECUTION").as_deref() {
            Ok("dynamic") => Some(Self::Dynamic),
            Ok("exhaustive") => Some(Self::Exhaustive),
            Err(_) => None,
            Ok(value) => {
                panic!("SPECTRA_ISOLATED_EXECUTION must be dynamic or exhaustive; got {value}",)
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Dynamic => "dynamic",
            Self::Exhaustive => "exhaustive",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct IsolatedExecutionResult {
    pub schema_version: usize,
    pub experiment: &'static str,
    pub execution_mode: &'static str,
    pub operations: usize,
    pub point_ids_returned: usize,
    pub leaf_identities_consumed: usize,
    pub intermediate_identities: usize,
    pub elapsed_ns: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Topology {
    Flat,
    ChainOpen,
    BalancedOpen,
    BalancedFinite,
    DiamondShared,
    DiamondUnfolded,
}

impl Topology {
    pub fn parse(value: &str) -> Self {
        match value {
            "flat" => Self::Flat,
            "chain_open" => Self::ChainOpen,
            "balanced_open" => Self::BalancedOpen,
            "balanced_finite" => Self::BalancedFinite,
            "diamond_shared" => Self::DiamondShared,
            "diamond_unfolded" => Self::DiamondUnfolded,
            _ => panic!(
                "topology must be flat, chain_open, balanced_open, balanced_finite, diamond_shared, or diamond_unfolded; got {value}",
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::ChainOpen => "chain_open",
            Self::BalancedOpen => "balanced_open",
            Self::BalancedFinite => "balanced_finite",
            Self::DiamondShared => "diamond_shared",
            Self::DiamondUnfolded => "diamond_unfolded",
        }
    }
}

pub struct BuiltPlan {
    pub plan: ExactCompositionPlan,
    /// For each physical leaf node, the source order index it represents.
    /// Shared diamonds reference one leaf node twice; unfolded diamonds contain
    /// two physical leaf nodes mapped to the same source order.
    pub physical_leaf_sources: Vec<usize>,
    pub fusion_nodes: usize,
}

pub fn build_plan(
    topology: Topology,
    orders: &[Vec<ExtendedPointId>],
    top_k: usize,
    rrf_k: usize,
    internal_limit: usize,
) -> BuiltPlan {
    assert!(orders.len() >= 2);
    build_plan_with_leaf_factory(
        topology,
        orders.len(),
        top_k,
        rrf_k,
        internal_limit,
        |source| ExactCompositionNode::Leaf {
            order: orders[source].clone(),
        },
    )
}

pub fn build_external_plan(
    topology: Topology,
    leaves: usize,
    top_k: usize,
    rrf_k: usize,
    internal_limit: usize,
) -> BuiltPlan {
    build_plan_with_leaf_factory(topology, leaves, top_k, rrf_k, internal_limit, |_| {
        ExactCompositionNode::ExternalLeaf
    })
}

fn build_plan_with_leaf_factory(
    topology: Topology,
    leaves: usize,
    top_k: usize,
    rrf_k: usize,
    internal_limit: usize,
    mut leaf: impl FnMut(usize) -> ExactCompositionNode,
) -> BuiltPlan {
    assert!(leaves >= 2);
    if matches!(
        topology,
        Topology::DiamondShared | Topology::DiamondUnfolded
    ) {
        assert!(leaves >= 4, "diamond experiments require four leaves");
    }
    let mut nodes: Vec<_> = (0..leaves).map(&mut leaf).collect();
    let mut physical_leaf_sources: Vec<_> = (0..leaves).collect();
    let original_leaves = nodes.len();
    let root = match topology {
        Topology::Flat => push_fusion(
            &mut nodes,
            (0..original_leaves).collect(),
            rrf_k,
            Some(top_k),
        ),
        Topology::ChainOpen => {
            let mut current = push_fusion(&mut nodes, vec![0, 1], rrf_k, None);
            for input in 2..original_leaves {
                let limit = (input + 1 == original_leaves).then_some(top_k);
                current = push_fusion(&mut nodes, vec![current, input], rrf_k, limit);
            }
            if original_leaves == 2
                && let ExactCompositionNode::Fusion { output_limit, .. } = &mut nodes[current]
            {
                *output_limit = Some(top_k);
            }
            current
        }
        Topology::BalancedOpen | Topology::BalancedFinite => {
            let finite = topology == Topology::BalancedFinite;
            let mut level: Vec<_> = (0..original_leaves).collect();
            while level.len() > 1 {
                let mut next = Vec::with_capacity(level.len().div_ceil(2));
                for pair in level.chunks(2) {
                    if pair.len() == 1 {
                        next.push(pair[0]);
                    } else {
                        next.push(push_fusion(
                            &mut nodes,
                            pair.to_vec(),
                            rrf_k,
                            finite.then_some(internal_limit),
                        ));
                    }
                }
                level = next;
            }
            let root = level[0];
            if let ExactCompositionNode::Fusion { output_limit, .. } = &mut nodes[root] {
                *output_limit = Some(top_k);
            }
            root
        }
        Topology::DiamondShared => {
            let left = push_fusion(&mut nodes, vec![0, 1], rrf_k, None);
            let right = push_fusion(&mut nodes, vec![0, 2], rrf_k, None);
            let mut inputs = vec![left, right];
            inputs.extend(3..original_leaves);
            push_fusion(&mut nodes, inputs, rrf_k, Some(top_k))
        }
        Topology::DiamondUnfolded => {
            let duplicate = nodes.len();
            nodes.push(leaf(0));
            physical_leaf_sources.push(0);
            let left = push_fusion(&mut nodes, vec![0, 1], rrf_k, None);
            let right = push_fusion(&mut nodes, vec![duplicate, 2], rrf_k, None);
            let mut inputs = vec![left, right];
            inputs.extend(3..original_leaves);
            push_fusion(&mut nodes, inputs, rrf_k, Some(top_k))
        }
    };
    let fusion_nodes = nodes.len() - physical_leaf_sources.len();
    BuiltPlan {
        plan: ExactCompositionPlan::new(nodes, root).unwrap(),
        physical_leaf_sources,
        fusion_nodes,
    }
}

fn push_fusion(
    nodes: &mut Vec<ExactCompositionNode>,
    inputs: Vec<usize>,
    rrf_k: usize,
    output_limit: Option<usize>,
) -> usize {
    let node = nodes.len();
    nodes.push(ExactCompositionNode::Fusion {
        weights: vec![1.0; inputs.len()],
        inputs,
        rrf_k,
        output_limit,
    });
    node
}
