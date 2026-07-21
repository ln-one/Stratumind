// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Demand-driven composition of exact rank streams over a finite DAG.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::reciprocal_rank_fusion::{
    DynamicRrfPolicy, DynamicRrfSession, ExactRrfStream, exact_rrf_scoring,
};
use crate::index::rank_stream::{ReplayBufferTracker, SharedRankStream};
use crate::types::{ExtendedPointId, ScoredPoint};

pub type ExactCompositionNodeId = usize;
pub type FallibleExactCompositionStream<'a> =
    Box<dyn Iterator<Item = Result<ExtendedPointId, ExactCompositionStreamFailure>> + 'a>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExactCompositionStreamFailure {
    Cancelled,
    SnapshotChanged,
    Source(String),
}

impl Display for ExactCompositionStreamFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("cancelled"),
            Self::SnapshotChanged => formatter.write_str("snapshot changed"),
            Self::Source(message) => formatter.write_str(message),
        }
    }
}

impl Error for ExactCompositionStreamFailure {}

#[derive(Clone, Debug, PartialEq)]
pub enum ExactCompositionNode {
    /// Materialized leaf used by deterministic tests and exhaustive oracles.
    Leaf { order: Vec<ExtendedPointId> },
    /// Production leaf whose exact stream must be supplied at execution time.
    /// Keeping it separate prevents logical plans from retaining an exhaustive
    /// identity order solely for research validation.
    ExternalLeaf,
    Fusion {
        inputs: Vec<ExactCompositionNodeId>,
        weights: Vec<f32>,
        rrf_k: usize,
        /// `Some(k)` is one complete V4 output. `None` is the resumable family
        /// of exact prefixes used when an internal operator must remain open.
        output_limit: Option<usize>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExactCompositionError {
    EmptyNetwork,
    InvalidRoot(ExactCompositionNodeId),
    InvalidInput {
        node: ExactCompositionNodeId,
        input: ExactCompositionNodeId,
    },
    Cycle,
    DuplicateLeafIdentity {
        node: ExactCompositionNodeId,
        identity: ExtendedPointId,
    },
    EmptyFusion(ExactCompositionNodeId),
    InvalidWeights(ExactCompositionNodeId),
    InvalidRrfK(ExactCompositionNodeId),
    InvalidOutputLimit(ExactCompositionNodeId),
    InvalidRequestedTopK,
    InvalidLeafStreams,
    MissingLeafStream(ExactCompositionNodeId),
    Leaf {
        node: ExactCompositionNodeId,
        failure: ExactCompositionStreamFailure,
    },
    Fusion {
        node: ExactCompositionNodeId,
        message: String,
    },
}

impl Display for ExactCompositionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyNetwork => formatter.write_str("an exact composition network is empty"),
            Self::InvalidRoot(root) => write!(formatter, "composition root {root} is out of range"),
            Self::InvalidInput { node, input } => {
                write!(
                    formatter,
                    "composition node {node} references missing input {input}"
                )
            }
            Self::Cycle => formatter.write_str("an exact composition network must be acyclic"),
            Self::DuplicateLeafIdentity { node, identity } => write!(
                formatter,
                "composition leaf {node} contains duplicate identity {identity}",
            ),
            Self::EmptyFusion(node) => {
                write!(formatter, "composition fusion node {node} has no inputs")
            }
            Self::InvalidWeights(node) => write!(
                formatter,
                "composition fusion node {node} has mismatched, negative, or non-finite weights",
            ),
            Self::InvalidRrfK(node) => {
                write!(formatter, "composition fusion node {node} has rrf_k=0")
            }
            Self::InvalidOutputLimit(node) => write!(
                formatter,
                "composition fusion node {node} has a zero finite output limit",
            ),
            Self::InvalidRequestedTopK => {
                formatter.write_str("composition root Top-K must be positive")
            }
            Self::InvalidLeafStreams => {
                formatter.write_str("external leaf stream slots must match the composition plan")
            }
            Self::MissingLeafStream(node) => {
                write!(formatter, "composition external leaf {node} has no stream")
            }
            Self::Leaf { node, failure } => {
                write!(formatter, "composition leaf {node} failed: {failure}")
            }
            Self::Fusion { node, message } => {
                write!(
                    formatter,
                    "composition fusion node {node} failed: {message}"
                )
            }
        }
    }
}

impl Error for ExactCompositionError {}

#[derive(Clone, Debug)]
pub struct ExactCompositionPlan {
    nodes: Vec<ExactCompositionNode>,
    root: ExactCompositionNodeId,
    topological_order: Vec<ExactCompositionNodeId>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExactCompositionExecutionPolicy {
    pub fusion_policy: DynamicRrfPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactCompositionNodeKind {
    Leaf,
    Fusion,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExactCompositionNodeTelemetry {
    pub node: ExactCompositionNodeId,
    pub kind: Option<ExactCompositionNodeKind>,
    pub output_limit: Option<usize>,
    pub logical_input_pulls: Vec<usize>,
    pub physical_output_pulls: usize,
    pub outputs_produced: usize,
    pub certification_checks: usize,
    pub peak_replay_buffered_identities: usize,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExactCompositionExecution {
    pub point_ids: Vec<ExtendedPointId>,
    pub nodes: Vec<ExactCompositionNodeTelemetry>,
    pub leaf_physical_pulls: usize,
    pub intermediate_outputs_produced: usize,
    /// Exact peak across all replay buffers in this execution.
    pub peak_replay_buffered_identities: usize,
    /// Sum of each node's individual peak. This is an upper bound, not a
    /// simultaneous-memory measurement.
    pub sum_node_peak_replay_buffered_identities: usize,
    pub cancelled_nodes: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExactCompositionExhaustiveExecution {
    pub point_ids: Vec<ExtendedPointId>,
    pub node_output_identities: Vec<usize>,
    pub leaf_identities_materialized: usize,
    pub intermediate_identities_materialized: usize,
    pub total_identities_materialized: usize,
}

#[derive(Clone, Debug, Default)]
struct FusionProgress {
    logical_input_pulls: Vec<usize>,
    outputs_produced: usize,
    certification_checks: usize,
}

struct LazyFusionStream<'a> {
    node: ExactCompositionNodeId,
    session: Option<DynamicRrfSession<'a>>,
    output_limit: Option<usize>,
    emitted: usize,
    progress: Rc<RefCell<FusionProgress>>,
    failure: Rc<RefCell<Option<ExactCompositionError>>>,
}

struct FallibleLeafAdapter<'a> {
    node: ExactCompositionNodeId,
    inner: Option<FallibleExactCompositionStream<'a>>,
    failure: Rc<RefCell<Option<ExactCompositionError>>>,
    seen: HashSet<ExtendedPointId>,
}

impl Iterator for FallibleLeafAdapter<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.as_mut()?.next() {
            Some(Ok(identity)) if self.seen.insert(identity) => Some(Ok(identity)),
            Some(Ok(identity)) => {
                let mut failure = self.failure.borrow_mut();
                if failure.is_none() {
                    *failure = Some(ExactCompositionError::DuplicateLeafIdentity {
                        node: self.node,
                        identity,
                    });
                }
                self.inner = None;
                Some(Err(OperationError::validation_error(format!(
                    "composition leaf {} contains duplicate identity {identity}",
                    self.node
                ))))
            }
            Some(Err(source_failure)) => {
                let mut failure = self.failure.borrow_mut();
                if failure.is_none() {
                    *failure = Some(ExactCompositionError::Leaf {
                        node: self.node,
                        failure: source_failure.clone(),
                    });
                }
                self.inner = None;
                Some(Err(match source_failure {
                    ExactCompositionStreamFailure::Cancelled => OperationError::cancelled(format!(
                        "composition leaf {} was cancelled",
                        self.node
                    )),
                    ExactCompositionStreamFailure::SnapshotChanged => {
                        OperationError::service_error_light(format!(
                            "composition leaf {} snapshot changed",
                            self.node
                        ))
                    }
                    ExactCompositionStreamFailure::Source(message) => {
                        OperationError::service_error_light(format!(
                            "composition leaf {} failed: {message}",
                            self.node
                        ))
                    }
                }))
            }
            None => {
                self.inner = None;
                None
            }
        }
    }
}

impl Iterator for LazyFusionStream<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        if self
            .output_limit
            .is_some_and(|output_limit| self.emitted >= output_limit)
        {
            self.session = None;
            return None;
        }
        let target = self.emitted + 1;
        let result = self
            .session
            .as_mut()
            .expect("an unfinished lazy fusion has a session")
            .run_to_prefix(target);
        let execution = match result {
            Ok(execution) => execution,
            Err(error) => {
                let mut failure = self.failure.borrow_mut();
                if failure.is_none() {
                    *failure = Some(ExactCompositionError::Fusion {
                        node: self.node,
                        message: error.to_string(),
                    });
                }
                self.session = None;
                return Some(Err(error));
            }
        };
        {
            let mut progress = self.progress.borrow_mut();
            progress.logical_input_pulls = execution.source_pulls.clone();
            progress.certification_checks = execution.certification_checks;
        }
        let Some(identity) = execution.point_ids.get(self.emitted).copied() else {
            self.session = None;
            return None;
        };
        self.emitted += 1;
        self.progress.borrow_mut().outputs_produced = self.emitted;
        if self.output_limit == Some(self.emitted) {
            self.session = None;
        }
        Some(Ok(identity))
    }
}

struct RuntimeNode<'a> {
    shared: SharedRankStream<'a>,
    progress: Option<Rc<RefCell<FusionProgress>>>,
}

impl ExactCompositionPlan {
    pub fn new(
        nodes: Vec<ExactCompositionNode>,
        root: ExactCompositionNodeId,
    ) -> Result<Self, ExactCompositionError> {
        if nodes.is_empty() {
            return Err(ExactCompositionError::EmptyNetwork);
        }
        if root >= nodes.len() {
            return Err(ExactCompositionError::InvalidRoot(root));
        }
        for (node, definition) in nodes.iter().enumerate() {
            match definition {
                ExactCompositionNode::Leaf { order } => {
                    let mut identities = HashSet::with_capacity(order.len());
                    for &identity in order {
                        if !identities.insert(identity) {
                            return Err(ExactCompositionError::DuplicateLeafIdentity {
                                node,
                                identity,
                            });
                        }
                    }
                }
                ExactCompositionNode::ExternalLeaf => {}
                ExactCompositionNode::Fusion {
                    inputs,
                    weights,
                    rrf_k,
                    output_limit,
                } => {
                    if inputs.is_empty() {
                        return Err(ExactCompositionError::EmptyFusion(node));
                    }
                    for &input in inputs {
                        if input >= nodes.len() {
                            return Err(ExactCompositionError::InvalidInput { node, input });
                        }
                    }
                    if weights.len() != inputs.len()
                        || weights
                            .iter()
                            .any(|weight| !weight.is_finite() || *weight < 0.0)
                        || !weights.iter().any(|weight| *weight > 0.0)
                    {
                        return Err(ExactCompositionError::InvalidWeights(node));
                    }
                    if *rrf_k == 0 {
                        return Err(ExactCompositionError::InvalidRrfK(node));
                    }
                    if *output_limit == Some(0) {
                        return Err(ExactCompositionError::InvalidOutputLimit(node));
                    }
                }
            }
        }
        let topological_order = topological_order(&nodes)?;
        Ok(Self {
            nodes,
            root,
            topological_order,
        })
    }

    pub fn execute(
        &self,
        root_top_k: usize,
    ) -> Result<ExactCompositionExecution, ExactCompositionError> {
        self.execute_with_policy(root_top_k, ExactCompositionExecutionPolicy::default())
    }

    pub fn execute_with_policy(
        &self,
        root_top_k: usize,
        policy: ExactCompositionExecutionPolicy,
    ) -> Result<ExactCompositionExecution, ExactCompositionError> {
        let leaf_streams = std::iter::repeat_with(|| None)
            .take(self.nodes.len())
            .collect();
        self.execute_with_leaf_streams_and_policy(root_top_k, leaf_streams, policy)
    }

    /// Executes the logical plan with externally supplied exact leaf streams.
    /// A `None` leaf uses the plan's materialized order; fusion slots must stay
    /// `None`. This preserves an explicit oracle while allowing any physical
    /// Dense, Sparse, remote, or nested exact stream at a leaf boundary.
    pub fn execute_with_leaf_streams<'a>(
        &self,
        root_top_k: usize,
        leaf_streams: Vec<Option<ExactRrfStream<'a>>>,
    ) -> Result<ExactCompositionExecution, ExactCompositionError> {
        self.execute_with_leaf_streams_and_policy(
            root_top_k,
            leaf_streams,
            ExactCompositionExecutionPolicy::default(),
        )
    }

    pub fn execute_with_leaf_streams_and_policy<'a>(
        &self,
        root_top_k: usize,
        leaf_streams: Vec<Option<ExactRrfStream<'a>>>,
        policy: ExactCompositionExecutionPolicy,
    ) -> Result<ExactCompositionExecution, ExactCompositionError> {
        let leaf_streams = leaf_streams
            .into_iter()
            .map(|stream| {
                stream.map(|stream| {
                    Box::new(stream.map(|item| {
                        item.map_err(|error| {
                            ExactCompositionStreamFailure::Source(error.to_string())
                        })
                    })) as FallibleExactCompositionStream<'a>
                })
            })
            .collect();
        self.execute_with_fallible_leaf_streams_and_policy(root_top_k, leaf_streams, policy)
    }

    pub fn execute_with_fallible_leaf_streams<'a>(
        &self,
        root_top_k: usize,
        leaf_streams: Vec<Option<FallibleExactCompositionStream<'a>>>,
    ) -> Result<ExactCompositionExecution, ExactCompositionError> {
        self.execute_with_fallible_leaf_streams_and_policy(
            root_top_k,
            leaf_streams,
            ExactCompositionExecutionPolicy::default(),
        )
    }

    pub fn execute_with_fallible_leaf_streams_and_policy<'a>(
        &self,
        root_top_k: usize,
        mut leaf_streams: Vec<Option<FallibleExactCompositionStream<'a>>>,
        policy: ExactCompositionExecutionPolicy,
    ) -> Result<ExactCompositionExecution, ExactCompositionError> {
        if root_top_k == 0 {
            return Err(ExactCompositionError::InvalidRequestedTopK);
        }
        if leaf_streams.len() != self.nodes.len()
            || self.nodes.iter().enumerate().any(|(node, definition)| {
                matches!(definition, ExactCompositionNode::Fusion { .. })
                    && leaf_streams[node].is_some()
            })
        {
            return Err(ExactCompositionError::InvalidLeafStreams);
        }
        let failure = Rc::new(RefCell::new(None));
        let replay_memory = ReplayBufferTracker::default();
        let mut runtime: Vec<Option<RuntimeNode<'a>>> = std::iter::repeat_with(|| None)
            .take(self.nodes.len())
            .collect();

        for &node in &self.topological_order {
            let (producer, progress): (ExactRrfStream<'a>, _) = match &self.nodes[node] {
                ExactCompositionNode::Leaf { order } => {
                    let inner = leaf_streams[node].take().unwrap_or_else(|| {
                        Box::new(order.clone().into_iter().map(Ok))
                            as FallibleExactCompositionStream<'a>
                    });
                    (
                        Box::new(FallibleLeafAdapter {
                            node,
                            inner: Some(inner),
                            failure: Rc::clone(&failure),
                            seen: HashSet::new(),
                        }),
                        None,
                    )
                }
                ExactCompositionNode::ExternalLeaf => {
                    let inner = leaf_streams[node]
                        .take()
                        .ok_or(ExactCompositionError::MissingLeafStream(node))?;
                    (
                        Box::new(FallibleLeafAdapter {
                            node,
                            inner: Some(inner),
                            failure: Rc::clone(&failure),
                            seen: HashSet::new(),
                        }),
                        None,
                    )
                }
                ExactCompositionNode::Fusion {
                    inputs,
                    weights,
                    rrf_k,
                    output_limit,
                } => {
                    let sources = inputs
                        .iter()
                        .map(|&input| {
                            runtime[input]
                                .as_ref()
                                .expect("topological input is built")
                                .shared
                                .subscribe()
                        })
                        .collect();
                    let session = DynamicRrfSession::new(
                        sources,
                        1,
                        *rrf_k,
                        Some(weights),
                        policy.fusion_policy,
                    )
                    .map_err(|error| ExactCompositionError::Fusion {
                        node,
                        message: error.to_string(),
                    })?;
                    let progress = Rc::new(RefCell::new(FusionProgress::default()));
                    (
                        Box::new(LazyFusionStream {
                            node,
                            session: Some(session),
                            output_limit: *output_limit,
                            emitted: 0,
                            progress: Rc::clone(&progress),
                            failure: Rc::clone(&failure),
                        }),
                        Some(progress),
                    )
                }
            };
            runtime[node] = Some(RuntimeNode {
                shared: SharedRankStream::new_tracked(producer, replay_memory.clone()),
                progress,
            });
        }

        let mut root = runtime[self.root]
            .as_ref()
            .expect("the root runtime is built")
            .shared
            .subscribe();
        let point_ids = root
            .by_ref()
            .take(root_top_k)
            .collect::<OperationResult<Vec<_>>>();
        drop(root);
        if let Some(error) = failure.borrow_mut().take() {
            return Err(error);
        }
        let point_ids = point_ids.map_err(|error| ExactCompositionError::Fusion {
            node: self.root,
            message: error.to_string(),
        })?;

        let mut telemetry = Vec::with_capacity(runtime.len());
        let mut leaf_physical_pulls = 0;
        let mut intermediate_outputs_produced = 0;
        let mut sum_node_peak_replay_buffered_identities = 0;
        let mut cancelled_nodes = 0;
        for (node, runtime_node) in runtime.iter().enumerate() {
            let runtime_node = runtime_node.as_ref().expect("every runtime node is built");
            let shared = runtime_node.shared.telemetry();
            let (kind, output_limit, logical_input_pulls, outputs_produced, checks) =
                match (&self.nodes[node], &runtime_node.progress) {
                    (
                        ExactCompositionNode::Leaf { .. } | ExactCompositionNode::ExternalLeaf,
                        None,
                    ) => {
                        leaf_physical_pulls += shared.physical_pulls;
                        (ExactCompositionNodeKind::Leaf, None, Vec::new(), 0, 0)
                    }
                    (ExactCompositionNode::Fusion { output_limit, .. }, Some(progress)) => {
                        let progress = progress.borrow();
                        intermediate_outputs_produced += progress.outputs_produced;
                        (
                            ExactCompositionNodeKind::Fusion,
                            *output_limit,
                            progress.logical_input_pulls.clone(),
                            progress.outputs_produced,
                            progress.certification_checks,
                        )
                    }
                    _ => unreachable!("runtime progress matches node kind"),
                };
            sum_node_peak_replay_buffered_identities += shared.peak_buffered_identities;
            cancelled_nodes += usize::from(shared.cancelled);
            telemetry.push(ExactCompositionNodeTelemetry {
                node,
                kind: Some(kind),
                output_limit,
                logical_input_pulls,
                physical_output_pulls: shared.physical_pulls,
                outputs_produced,
                certification_checks: checks,
                peak_replay_buffered_identities: shared.peak_buffered_identities,
                cancelled: shared.cancelled,
            });
        }

        Ok(ExactCompositionExecution {
            point_ids,
            nodes: telemetry,
            leaf_physical_pulls,
            intermediate_outputs_produced,
            peak_replay_buffered_identities: replay_memory.peak_identities(),
            sum_node_peak_replay_buffered_identities,
            cancelled_nodes,
        })
    }

    pub fn exhaustive_root(
        &self,
        root_top_k: usize,
    ) -> Result<Vec<ExtendedPointId>, ExactCompositionError> {
        Ok(self.exhaustive_execute(root_top_k)?.point_ids)
    }

    pub fn exhaustive_execute(
        &self,
        root_top_k: usize,
    ) -> Result<ExactCompositionExhaustiveExecution, ExactCompositionError> {
        let leaf_streams = std::iter::repeat_with(|| None)
            .take(self.nodes.len())
            .collect();
        self.exhaustive_with_leaf_streams(root_top_k, leaf_streams)
    }

    /// Fully materializes every leaf and fusion node in the same logical
    /// network. This is the correctness oracle and the explicit full-work
    /// baseline for composition experiments.
    pub fn exhaustive_with_leaf_streams<'a>(
        &self,
        root_top_k: usize,
        leaf_streams: Vec<Option<ExactRrfStream<'a>>>,
    ) -> Result<ExactCompositionExhaustiveExecution, ExactCompositionError> {
        let leaf_streams = leaf_streams
            .into_iter()
            .map(|stream| {
                stream.map(|stream| {
                    Box::new(stream.map(|item| {
                        item.map_err(|error| {
                            ExactCompositionStreamFailure::Source(error.to_string())
                        })
                    })) as FallibleExactCompositionStream<'a>
                })
            })
            .collect();
        self.exhaustive_with_fallible_leaf_streams(root_top_k, leaf_streams)
    }

    pub fn exhaustive_with_fallible_leaf_streams<'a>(
        &self,
        root_top_k: usize,
        mut leaf_streams: Vec<Option<FallibleExactCompositionStream<'a>>>,
    ) -> Result<ExactCompositionExhaustiveExecution, ExactCompositionError> {
        if root_top_k == 0 {
            return Err(ExactCompositionError::InvalidRequestedTopK);
        }
        if leaf_streams.len() != self.nodes.len()
            || self.nodes.iter().enumerate().any(|(node, definition)| {
                matches!(definition, ExactCompositionNode::Fusion { .. })
                    && leaf_streams[node].is_some()
            })
        {
            return Err(ExactCompositionError::InvalidLeafStreams);
        }
        let mut materialized: Vec<Option<Vec<ExtendedPointId>>> = vec![None; self.nodes.len()];
        let mut node_output_identities = vec![0; self.nodes.len()];
        let mut leaf_identities_materialized = 0;
        let mut intermediate_identities_materialized = 0;
        for &node in &self.topological_order {
            let output =
                match &self.nodes[node] {
                    ExactCompositionNode::Leaf { order } => {
                        let stream = leaf_streams[node].take().unwrap_or_else(|| {
                            Box::new(order.clone().into_iter().map(Ok))
                                as FallibleExactCompositionStream<'a>
                        });
                        let output = collect_leaf_stream(node, stream)?;
                        leaf_identities_materialized += output.len();
                        output
                    }
                    ExactCompositionNode::ExternalLeaf => {
                        let stream = leaf_streams[node]
                            .take()
                            .ok_or(ExactCompositionError::MissingLeafStream(node))?;
                        let output = collect_leaf_stream(node, stream)?;
                        leaf_identities_materialized += output.len();
                        output
                    }
                    ExactCompositionNode::Fusion {
                        inputs,
                        weights,
                        rrf_k,
                        output_limit,
                    } => {
                        let responses = inputs
                            .iter()
                            .map(|&input| {
                                materialized[input]
                                    .as_ref()
                                    .expect("topological input is materialized")
                                    .iter()
                                    .map(|&id| scored_point(id))
                                    .collect()
                            })
                            .collect();
                        let mut output = exact_rrf_scoring(responses, *rrf_k, Some(weights))
                            .map_err(|error| ExactCompositionError::Fusion {
                                node,
                                message: error.to_string(),
                            })?;
                        if let Some(output_limit) = output_limit {
                            output.truncate(*output_limit);
                        }
                        let output: Vec<_> = output.into_iter().map(|point| point.id).collect();
                        intermediate_identities_materialized += output.len();
                        output
                    }
                };
            node_output_identities[node] = output.len();
            materialized[node] = Some(output);
        }
        let mut point_ids = materialized[self.root]
            .as_ref()
            .expect("the root is materialized")
            .clone();
        point_ids.truncate(root_top_k);
        Ok(ExactCompositionExhaustiveExecution {
            point_ids,
            node_output_identities,
            leaf_identities_materialized,
            intermediate_identities_materialized,
            total_identities_materialized: leaf_identities_materialized
                + intermediate_identities_materialized,
        })
    }
}

fn collect_leaf_stream(
    node: ExactCompositionNodeId,
    stream: FallibleExactCompositionStream<'_>,
) -> Result<Vec<ExtendedPointId>, ExactCompositionError> {
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for identity in stream {
        let identity = identity.map_err(|failure| ExactCompositionError::Leaf { node, failure })?;
        if !seen.insert(identity) {
            return Err(ExactCompositionError::DuplicateLeafIdentity { node, identity });
        }
        output.push(identity);
    }
    Ok(output)
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

fn topological_order(
    nodes: &[ExactCompositionNode],
) -> Result<Vec<ExactCompositionNodeId>, ExactCompositionError> {
    let mut dependencies = vec![0usize; nodes.len()];
    let mut consumers = vec![Vec::new(); nodes.len()];
    for (node, definition) in nodes.iter().enumerate() {
        if let ExactCompositionNode::Fusion { inputs, .. } = definition {
            dependencies[node] = inputs.len();
            for &input in inputs {
                consumers[input].push(node);
            }
        }
    }
    let mut ready: BTreeSet<_> = dependencies
        .iter()
        .enumerate()
        .filter_map(|(node, &dependencies)| (dependencies == 0).then_some(node))
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(node) = ready.pop_first() {
        order.push(node);
        for &consumer in &consumers[node] {
            dependencies[consumer] -= 1;
            if dependencies[consumer] == 0 {
                ready.insert(consumer);
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(ExactCompositionError::Cycle);
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use sparse::common::sparse_vector::RemappedSparseVector;
    use sparse::index::block_max::{BlockMaxIndex, SparseDocument};

    use super::*;
    use crate::index::dense_ball::DenseDocument;
    use crate::index::dense_quantized::DenseQuantizedIndex;

    fn ids(values: impl IntoIterator<Item = u64>) -> Vec<ExtendedPointId> {
        values.into_iter().map(ExtendedPointId::from).collect()
    }

    #[test]
    fn adversarial_chain_matches_full_materialization_and_safely_exhausts() {
        let plan = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf { order: ids(0..100) },
                ExactCompositionNode::Leaf {
                    order: ids((0..100).rev()),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(40),
                },
                ExactCompositionNode::Leaf {
                    order: ids((50..150).rev()),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![2, 3],
                    weights: vec![2.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(20),
                },
            ],
            4,
        )
        .unwrap();

        let exhaustive = plan.exhaustive_execute(20).unwrap();
        let actual = plan.execute(20).unwrap();
        let degraded = plan
            .execute_with_policy(
                20,
                ExactCompositionExecutionPolicy {
                    fusion_policy: DynamicRrfPolicy {
                        exhaustive_after_pulls_per_source: Some(16),
                        ..DynamicRrfPolicy::default()
                    },
                },
            )
            .unwrap();

        assert_eq!(actual.point_ids, exhaustive.point_ids);
        assert_eq!(degraded.point_ids, exhaustive.point_ids);
        assert_eq!(actual.leaf_physical_pulls, 300);
        assert_eq!(exhaustive.leaf_identities_materialized, 300);
        assert_eq!(exhaustive.intermediate_identities_materialized, 60);
        assert_eq!(exhaustive.total_identities_materialized, 360);
    }

    #[test]
    fn diamond_shares_one_physical_child_stream_between_two_consumers() {
        let plan = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf {
                    order: ids(0..1_000),
                },
                ExactCompositionNode::Leaf {
                    order: ids(1_000..2_000),
                },
                ExactCompositionNode::Leaf {
                    order: ids(2_000..3_000),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(100),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 2],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(100),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![3, 4],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(20),
                },
            ],
            5,
        )
        .unwrap();

        let expected = plan.exhaustive_root(20).unwrap();
        let actual = plan.execute(20).unwrap();

        assert_eq!(actual.point_ids, expected);
        let shared_leaf = &actual.nodes[0];
        let logical_pulls =
            actual.nodes[3].logical_input_pulls[0] + actual.nodes[4].logical_input_pulls[0];
        assert!(shared_leaf.physical_output_pulls < logical_pulls);
        assert!(shared_leaf.peak_replay_buffered_identities > 0);
        assert!(actual.peak_replay_buffered_identities > 0);
        assert!(
            actual.peak_replay_buffered_identities
                <= actual.sum_node_peak_replay_buffered_identities
        );
    }

    #[test]
    fn real_dense_and_sparse_leaf_streams_compose_without_adapter_semantics() {
        let dense_documents: Vec<_> = (0..257)
            .map(|id| DenseDocument {
                id,
                vector: vec![
                    ((id * 17) % 31) as f32 / 31.0,
                    ((id * 29 + 7) % 37) as f32 / 37.0,
                    ((id * 11 + 3) % 41) as f32 / 41.0,
                ],
            })
            .collect();
        let sparse_documents: Vec<_> = (0..257)
            .map(|id| SparseDocument {
                id,
                vector: RemappedSparseVector {
                    indices: vec![0, 1, 2],
                    values: vec![1.0, (id % 13) as f32, ((id * 7) % 17) as f32],
                },
            })
            .collect();
        let dense_query = [1.0, -0.25, 0.5];
        let sparse_query = RemappedSparseVector {
            indices: vec![0, 2],
            values: vec![1.0, 3.0],
        };
        let dense = DenseQuantizedIndex::build(dense_documents).unwrap();
        let sparse = BlockMaxIndex::build(sparse_documents, 17).unwrap();
        let dense_order: Vec<_> = dense
            .stream(&dense_query)
            .unwrap()
            .map(|point| ExtendedPointId::from(u64::from(point.id)))
            .collect();
        let sparse_order: Vec<_> = sparse
            .stream(sparse_query.clone())
            .unwrap()
            .map(|point| ExtendedPointId::from(u64::from(point.idx)))
            .collect();
        let plan = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf { order: dense_order },
                ExactCompositionNode::Leaf {
                    order: sparse_order,
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(20),
                },
            ],
            2,
        )
        .unwrap();
        let leaf_streams: Vec<Option<ExactRrfStream<'_>>> = vec![
            Some(Box::new(
                dense
                    .stream(&dense_query)
                    .unwrap()
                    .map(|point| Ok(ExtendedPointId::from(u64::from(point.id)))),
            )),
            Some(Box::new(sparse.stream(sparse_query).unwrap().map(
                |point| Ok(ExtendedPointId::from(u64::from(point.idx))),
            ))),
            None,
        ];

        let actual = plan.execute_with_leaf_streams(20, leaf_streams).unwrap();

        assert_eq!(actual.point_ids, plan.exhaustive_root(20).unwrap());
        assert!(actual.leaf_physical_pulls > 0);
    }

    #[test]
    fn finite_child_limit_is_semantics_not_an_execution_hint() {
        let limited = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf { order: ids(0..20) },
                ExactCompositionNode::Leaf {
                    order: ids((0..20).rev()),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: Some(3),
                },
                ExactCompositionNode::Leaf { order: ids(10..30) },
                ExactCompositionNode::Fusion {
                    inputs: vec![2, 3],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: Some(10),
                },
            ],
            4,
        )
        .unwrap();
        let open = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf { order: ids(0..20) },
                ExactCompositionNode::Leaf {
                    order: ids((0..20).rev()),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: None,
                },
                ExactCompositionNode::Leaf { order: ids(10..30) },
                ExactCompositionNode::Fusion {
                    inputs: vec![2, 3],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: Some(10),
                },
            ],
            4,
        )
        .unwrap();

        assert_ne!(
            limited.exhaustive_root(10).unwrap(),
            open.exhaustive_root(10).unwrap()
        );
        assert_eq!(
            limited.execute(10).unwrap().point_ids,
            limited.exhaustive_root(10).unwrap()
        );
        assert_eq!(
            open.execute(10).unwrap().point_ids,
            open.exhaustive_root(10).unwrap()
        );
    }

    #[test]
    fn fusion_topology_is_not_associative() {
        let leaves = [
            ids([0, 1, 2, 3, 4]),
            ids([1, 2, 0, 3, 4]),
            ids([2, 3, 1, 0, 4]),
        ];
        let left_nested = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf {
                    order: leaves[0].clone(),
                },
                ExactCompositionNode::Leaf {
                    order: leaves[1].clone(),
                },
                ExactCompositionNode::Leaf {
                    order: leaves[2].clone(),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: Some(3),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![3, 2],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: Some(5),
                },
            ],
            4,
        )
        .unwrap();
        let right_nested = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf {
                    order: leaves[0].clone(),
                },
                ExactCompositionNode::Leaf {
                    order: leaves[1].clone(),
                },
                ExactCompositionNode::Leaf {
                    order: leaves[2].clone(),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![1, 2],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: Some(3),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 3],
                    weights: vec![1.0, 1.0],
                    rrf_k: 2,
                    output_limit: Some(5),
                },
            ],
            4,
        )
        .unwrap();

        let left = left_nested.execute(5).unwrap().point_ids;
        let right = right_nested.execute(5).unwrap().point_ids;

        assert_eq!(left, left_nested.exhaustive_root(5).unwrap());
        assert_eq!(right, right_nested.exhaustive_root(5).unwrap());
        assert_eq!(left, ids([1, 2, 0, 3, 4]));
        assert_eq!(right, ids([2, 1, 0, 3, 4]));
        assert_ne!(left, right);
    }

    #[test]
    fn deterministic_random_dags_match_full_materialization() {
        for case in 0_u64..250 {
            let leaf = |leaf: u64| {
                let mut order: Vec<_> = (0_u64..32).collect();
                order.sort_unstable_by_key(|identity| {
                    identity
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(case.wrapping_mul(0xBF58_476D_1CE4_E5B9))
                        .wrapping_add(leaf.wrapping_mul(0x94D0_49BB_1331_11EB))
                        .rotate_left(((identity + leaf + case) % 64) as u32)
                });
                ids(order)
            };
            let small_limit = 4 + case as usize % 13;
            let plan = ExactCompositionPlan::new(
                vec![
                    ExactCompositionNode::Leaf { order: leaf(0) },
                    ExactCompositionNode::Leaf { order: leaf(1) },
                    ExactCompositionNode::Leaf { order: leaf(2) },
                    ExactCompositionNode::Fusion {
                        inputs: vec![0, 1],
                        weights: vec![1.0, 0.5 + (case % 3) as f32],
                        rrf_k: 2 + case as usize % 9,
                        output_limit: Some(small_limit),
                    },
                    ExactCompositionNode::Fusion {
                        inputs: vec![0, 2],
                        weights: vec![1.5, 1.0],
                        rrf_k: 3 + case as usize % 7,
                        output_limit: None,
                    },
                    ExactCompositionNode::Fusion {
                        inputs: vec![3, 4, 1],
                        weights: vec![1.0, 2.0, 0.25],
                        rrf_k: 2 + case as usize % 5,
                        output_limit: Some(12),
                    },
                ],
                5,
            )
            .unwrap();

            assert_eq!(
                plan.execute(12).unwrap().point_ids,
                plan.exhaustive_root(12).unwrap(),
                "case={case}",
            );
        }
    }

    #[test]
    fn root_early_stop_cancels_unneeded_descendants() {
        let plan = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf {
                    order: ids(0..10_000),
                },
                ExactCompositionNode::Leaf {
                    order: ids(0..10_000),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(1_000),
                },
            ],
            2,
        )
        .unwrap();

        let execution = plan.execute(5).unwrap();

        assert_eq!(execution.point_ids, ids(0..5));
        assert_eq!(execution.leaf_physical_pulls, 10);
        assert!(execution.cancelled_nodes >= 1);
        assert!(
            execution
                .nodes
                .iter()
                .all(|node| node.physical_output_pulls <= 5)
        );
    }

    #[test]
    fn observed_leaf_failure_is_not_treated_as_exact_exhaustion() {
        let plan = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf { order: ids(0..2) },
                ExactCompositionNode::Leaf { order: ids(0..2) },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(2),
                },
            ],
            2,
        )
        .unwrap();
        let streams: Vec<Option<FallibleExactCompositionStream<'_>>> = vec![
            Some(Box::new(
                vec![Err(ExactCompositionStreamFailure::Cancelled)].into_iter(),
            )),
            Some(Box::new(vec![Ok(0.into()), Ok(1.into())].into_iter())),
            None,
        ];

        let error = plan
            .execute_with_fallible_leaf_streams(2, streams)
            .unwrap_err();

        assert_eq!(
            error,
            ExactCompositionError::Leaf {
                node: 0,
                failure: ExactCompositionStreamFailure::Cancelled,
            }
        );
        let exhaustive_error = plan
            .exhaustive_with_fallible_leaf_streams(
                2,
                vec![
                    Some(Box::new(
                        vec![Err(ExactCompositionStreamFailure::SnapshotChanged)].into_iter(),
                    )),
                    Some(Box::new(vec![Ok(0.into()), Ok(1.into())].into_iter())),
                    None,
                ],
            )
            .unwrap_err();
        assert_eq!(
            exhaustive_error,
            ExactCompositionError::Leaf {
                node: 0,
                failure: ExactCompositionStreamFailure::SnapshotChanged,
            }
        );
    }

    #[test]
    fn unobserved_leaf_tail_failure_does_not_invalidate_a_certified_prefix() {
        let plan = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Leaf { order: ids(0..2) },
                ExactCompositionNode::Leaf { order: ids(0..2) },
                ExactCompositionNode::Fusion {
                    inputs: vec![0, 1],
                    weights: vec![1.0, 1.0],
                    rrf_k: 60,
                    output_limit: Some(1),
                },
            ],
            2,
        )
        .unwrap();
        let source = || {
            Box::new(
                vec![
                    Ok(ExtendedPointId::from(0)),
                    Err(ExactCompositionStreamFailure::SnapshotChanged),
                ]
                .into_iter(),
            ) as FallibleExactCompositionStream<'_>
        };

        let actual = plan
            .execute_with_fallible_leaf_streams(1, vec![Some(source()), Some(source()), None])
            .unwrap();

        assert_eq!(actual.point_ids, ids([0]));
    }

    #[test]
    fn cycle_and_duplicate_leaf_identity_are_rejected() {
        let cycle = ExactCompositionPlan::new(
            vec![
                ExactCompositionNode::Fusion {
                    inputs: vec![1],
                    weights: vec![1.0],
                    rrf_k: 60,
                    output_limit: Some(1),
                },
                ExactCompositionNode::Fusion {
                    inputs: vec![0],
                    weights: vec![1.0],
                    rrf_k: 60,
                    output_limit: Some(1),
                },
            ],
            0,
        );
        assert!(matches!(cycle, Err(ExactCompositionError::Cycle)));

        let duplicate = ExactCompositionPlan::new(
            vec![ExactCompositionNode::Leaf {
                order: ids([1, 2, 1]),
            }],
            0,
        );
        assert!(matches!(
            duplicate,
            Err(ExactCompositionError::DuplicateLeafIdentity { .. })
        ));
    }

    #[test]
    fn external_leaf_requires_a_stream_without_retaining_an_oracle_order() {
        let plan = ExactCompositionPlan::new(vec![ExactCompositionNode::ExternalLeaf], 0).unwrap();

        assert_eq!(
            plan.execute(1).unwrap_err(),
            ExactCompositionError::MissingLeafStream(0)
        );
        let actual = plan
            .execute_with_leaf_streams(2, vec![Some(Box::new(ids([3, 1]).into_iter().map(Ok)))])
            .unwrap();
        let exhaustive = plan
            .exhaustive_with_leaf_streams(2, vec![Some(Box::new(ids([3, 1]).into_iter().map(Ok)))])
            .unwrap();

        assert_eq!(actual.point_ids, ids([3, 1]));
        assert_eq!(actual.point_ids, exhaustive.point_ids);
    }

    #[test]
    fn external_leaf_rejects_duplicate_identities_in_both_executors() {
        let plan = ExactCompositionPlan::new(vec![ExactCompositionNode::ExternalLeaf], 0).unwrap();
        let stream = || {
            Some(Box::new(ids([3, 3]).into_iter().map(Ok)) as FallibleExactCompositionStream<'_>)
        };

        let lazy_error = plan
            .execute_with_fallible_leaf_streams(2, vec![stream()])
            .unwrap_err();
        let exhaustive_error = plan
            .exhaustive_with_fallible_leaf_streams(2, vec![stream()])
            .unwrap_err();

        assert_eq!(
            lazy_error,
            ExactCompositionError::DuplicateLeafIdentity {
                node: 0,
                identity: 3.into(),
            }
        );
        assert_eq!(exhaustive_error, lazy_error);
    }
}
