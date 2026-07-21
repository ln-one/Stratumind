// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use common::types::PointOffsetType;
use ordered_float::OrderedFloat;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseDocument {
    pub id: PointOffsetType,
    pub vector: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseScoredPoint {
    pub id: PointOffsetType,
    pub score: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseBallError {
    ZeroBlockSize,
    ZeroClusteringIterations,
    InvalidBranchFactor,
    EmptyDocuments,
    DuplicateDocument(PointOffsetType),
    InvalidDocument(PointOffsetType),
    InvalidQuery,
}

impl Display for DenseBallError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBlockSize => formatter.write_str("Dense Ball block size must be positive"),
            Self::ZeroClusteringIterations => {
                formatter.write_str("Dense Ball clustering iterations must be positive")
            }
            Self::InvalidBranchFactor => {
                formatter.write_str("Dense Ball tree branch factor must be at least two")
            }
            Self::EmptyDocuments => formatter.write_str("Dense Ball requires at least one document"),
            Self::DuplicateDocument(id) => {
                write!(formatter, "Dense Ball document {id} occurs more than once")
            }
            Self::InvalidDocument(id) => write!(
                formatter,
                "Dense Ball document {id} has a mismatched dimension or non-finite coordinate"
            ),
            Self::InvalidQuery => formatter.write_str(
                "Dense Ball query must match the index dimension, be finite, and have non-zero norm",
            ),
        }
    }
}

impl Error for DenseBallError {}

#[derive(Clone, Debug)]
enum BallNodeKind {
    Leaf {
        document_start: usize,
        document_end: usize,
    },
    Internal {
        children: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
struct BallNode {
    min_id: PointOffsetType,
    center: Vec<f64>,
    radius: f64,
    spherical_cap: SphericalCap,
    kind: BallNodeKind,
}

#[derive(Clone, Debug)]
struct SphericalCap {
    direction: Vec<f64>,
    min_cosine: f64,
    min_norm: f64,
    max_norm: f64,
    contains_zero: bool,
}

#[derive(Clone, Debug)]
pub struct DenseBallIndex {
    documents: Vec<DenseDocument>,
    identity_positions: Vec<(PointOffsetType, usize)>,
    nodes: Vec<BallNode>,
    roots: Vec<usize>,
    leaf_block_count: usize,
    dimension: usize,
    block_size: usize,
}

impl DenseBallIndex {
    pub fn build(
        mut documents: Vec<DenseDocument>,
        block_size: usize,
    ) -> Result<Self, DenseBallError> {
        let dimension = validate_and_sort_documents(&mut documents, block_size)?;
        Ok(Self::build_from_ordered_documents(
            documents, block_size, dimension,
        ))
    }

    /// Builds query-independent, capacity-bounded clusters before computing each ball.
    ///
    /// Clustering changes only physical grouping. Point identity and exact scores remain unchanged.
    pub fn build_clustered(
        mut documents: Vec<DenseDocument>,
        block_size: usize,
        iterations: usize,
    ) -> Result<Self, DenseBallError> {
        if iterations == 0 {
            return Err(DenseBallError::ZeroClusteringIterations);
        }
        let dimension = validate_and_sort_documents(&mut documents, block_size)?;
        let documents = balanced_kmeans_order(documents, block_size, iterations, dimension);
        Ok(Self::build_from_ordered_documents(
            documents, block_size, dimension,
        ))
    }

    /// Builds a recursive, query-independent ball tree.
    ///
    /// Every internal node carries an admissible upper bound and may refine into smaller nodes.
    /// Only leaf expansion evaluates exact document scores.
    pub fn build_hierarchical_clustered(
        mut documents: Vec<DenseDocument>,
        leaf_size: usize,
        branch_factor: usize,
        iterations: usize,
    ) -> Result<Self, DenseBallError> {
        if iterations == 0 {
            return Err(DenseBallError::ZeroClusteringIterations);
        }
        let dimension = validate_and_sort_documents(&mut documents, leaf_size)?;
        if branch_factor < 2 {
            return Err(DenseBallError::InvalidBranchFactor);
        }

        let mut ordered_documents = Vec::with_capacity(documents.len());
        let mut nodes = Vec::new();
        let root = build_cluster_tree(
            documents,
            leaf_size,
            branch_factor,
            iterations,
            dimension,
            &mut ordered_documents,
            &mut nodes,
        );
        let leaf_block_count = nodes
            .iter()
            .filter(|node| matches!(node.kind, BallNodeKind::Leaf { .. }))
            .count();

        let identity_positions = identity_positions(&ordered_documents);
        Ok(Self {
            documents: ordered_documents,
            identity_positions,
            nodes,
            roots: vec![root],
            leaf_block_count,
            dimension,
            block_size: leaf_size,
        })
    }

    fn build_from_ordered_documents(
        documents: Vec<DenseDocument>,
        block_size: usize,
        dimension: usize,
    ) -> Self {
        let nodes: Vec<_> = documents
            .chunks(block_size)
            .enumerate()
            .map(|(block_index, block_documents)| {
                let document_start = block_index * block_size;
                let document_end = document_start + block_documents.len();
                let center = centroid(block_documents, dimension);
                let radius = block_documents
                    .iter()
                    .map(|document| euclidean_distance(&document.vector, &center))
                    .fold(0.0, f64::max);
                let radius = if radius == 0.0 { 0.0 } else { radius.next_up() };
                BallNode {
                    min_id: block_documents
                        .iter()
                        .map(|document| document.id)
                        .min()
                        .unwrap(),
                    center,
                    radius,
                    spherical_cap: spherical_cap(block_documents, dimension),
                    kind: BallNodeKind::Leaf {
                        document_start,
                        document_end,
                    },
                }
            })
            .collect();
        let roots = (0..nodes.len()).collect();
        let leaf_block_count = nodes.len();

        let identity_positions = identity_positions(&documents);
        Self {
            documents,
            identity_positions,
            nodes,
            roots,
            leaf_block_count,
            dimension,
            block_size,
        }
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn block_count(&self) -> usize {
        self.leaf_block_count
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn stream(&self, query: &[f32]) -> Result<DenseBallStream<'_>, DenseBallError> {
        DenseBallStream::new(self, query)
    }

    pub(super) fn query_components(
        &self,
        query: &[f32],
    ) -> Result<(Vec<f64>, f64), DenseBallError> {
        if query.len() != self.dimension || query.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(DenseBallError::InvalidQuery);
        }
        let query: Vec<_> = query
            .iter()
            .map(|coordinate| f64::from(*coordinate))
            .collect();
        let query_norm = dot(&query, &query).sqrt();
        if query_norm == 0.0 {
            return Err(DenseBallError::InvalidQuery);
        }
        Ok((query, query_norm))
    }

    pub(super) fn root_indices(&self) -> &[usize] {
        &self.roots
    }

    pub(super) fn node_bound(
        &self,
        node_index: usize,
        query: &[f64],
        query_norm: f64,
    ) -> DenseBallNodeBound {
        let entry = node_queue_entry(
            node_index,
            &self.nodes[node_index],
            query,
            query_norm,
            self.dimension,
        );
        DenseBallNodeBound {
            node_index,
            min_id: entry.min_id,
            upper_bound: entry.upper_bound.0,
        }
    }

    pub(super) fn node_contents(&self, node_index: usize) -> DenseBallNodeContents<'_> {
        match &self.nodes[node_index].kind {
            BallNodeKind::Internal { children } => DenseBallNodeContents::Internal(children),
            BallNodeKind::Leaf {
                document_start,
                document_end,
            } => DenseBallNodeContents::Leaf(&self.documents[*document_start..*document_end]),
        }
    }

    pub(super) fn document(&self, id: PointOffsetType) -> Option<&DenseDocument> {
        self.identity_positions
            .binary_search_by_key(&id, |(identity, _)| *identity)
            .ok()
            .map(|position| &self.documents[self.identity_positions[position].1])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DenseBallNodeBound {
    pub node_index: usize,
    pub min_id: PointOffsetType,
    pub upper_bound: f64,
}

pub(super) enum DenseBallNodeContents<'a> {
    Internal(&'a [usize]),
    Leaf(&'a [DenseDocument]),
}

fn identity_positions(documents: &[DenseDocument]) -> Vec<(PointOffsetType, usize)> {
    let mut positions: Vec<_> = documents
        .iter()
        .enumerate()
        .map(|(position, document)| (document.id, position))
        .collect();
    positions.sort_unstable_by_key(|(id, _)| *id);
    positions
}

fn validate_and_sort_documents(
    documents: &mut [DenseDocument],
    block_size: usize,
) -> Result<usize, DenseBallError> {
    if block_size == 0 {
        return Err(DenseBallError::ZeroBlockSize);
    }
    let Some(first) = documents.first() else {
        return Err(DenseBallError::EmptyDocuments);
    };
    let dimension = first.vector.len();
    if dimension == 0 {
        return Err(DenseBallError::InvalidDocument(first.id));
    }
    for document in documents.iter() {
        if document.vector.len() != dimension
            || document
                .vector
                .iter()
                .any(|coordinate| !coordinate.is_finite())
        {
            return Err(DenseBallError::InvalidDocument(document.id));
        }
    }

    documents.sort_unstable_by_key(|document| document.id);
    if let Some(duplicate) = documents
        .array_windows()
        .find(|[left, right]| left.id == right.id)
    {
        return Err(DenseBallError::DuplicateDocument(duplicate[0].id));
    }
    Ok(dimension)
}

fn balanced_kmeans_order(
    documents: Vec<DenseDocument>,
    block_size: usize,
    iterations: usize,
    dimension: usize,
) -> Vec<DenseDocument> {
    let cluster_count = documents.len().div_ceil(block_size);
    if cluster_count == 1 {
        return documents;
    }

    balanced_kmeans_partitions(documents, cluster_count, block_size, iterations, dimension)
        .into_iter()
        .flatten()
        .collect()
}

fn build_cluster_tree(
    mut documents: Vec<DenseDocument>,
    leaf_size: usize,
    branch_factor: usize,
    iterations: usize,
    dimension: usize,
    ordered_documents: &mut Vec<DenseDocument>,
    nodes: &mut Vec<BallNode>,
) -> usize {
    let center = centroid(&documents, dimension);
    let radius = documents
        .iter()
        .map(|document| euclidean_distance(&document.vector, &center))
        .fold(0.0, f64::max);
    let radius = if radius == 0.0 { 0.0 } else { radius.next_up() };
    let min_id = documents.iter().map(|document| document.id).min().unwrap();
    let spherical_cap = spherical_cap(&documents, dimension);

    if documents.len() <= leaf_size {
        documents.sort_unstable_by_key(|document| document.id);
        let document_start = ordered_documents.len();
        ordered_documents.extend(documents);
        let document_end = ordered_documents.len();
        let node_index = nodes.len();
        nodes.push(BallNode {
            min_id,
            center,
            radius,
            spherical_cap,
            kind: BallNodeKind::Leaf {
                document_start,
                document_end,
            },
        });
        return node_index;
    }

    let cluster_count = branch_factor.min(documents.len().div_ceil(leaf_size));
    let capacity = documents.len().div_ceil(cluster_count);
    let partitions =
        balanced_kmeans_partitions(documents, cluster_count, capacity, iterations, dimension);
    let children = partitions
        .into_iter()
        .map(|partition| {
            build_cluster_tree(
                partition,
                leaf_size,
                branch_factor,
                iterations,
                dimension,
                ordered_documents,
                nodes,
            )
        })
        .collect();
    let node_index = nodes.len();
    nodes.push(BallNode {
        min_id,
        center,
        radius,
        spherical_cap,
        kind: BallNodeKind::Internal { children },
    });
    node_index
}

fn balanced_kmeans_partitions(
    documents: Vec<DenseDocument>,
    cluster_count: usize,
    capacity: usize,
    iterations: usize,
    dimension: usize,
) -> Vec<Vec<DenseDocument>> {
    debug_assert!(cluster_count > 1);
    debug_assert!(cluster_count <= documents.len());
    debug_assert!(cluster_count * capacity >= documents.len());

    let mut centers = farthest_first_centers(&documents, cluster_count, dimension);
    let mut assignments = vec![0; documents.len()];
    for _ in 0..iterations {
        assignments = capacity_bounded_assignments(&documents, &centers, capacity);
        let mut next_centers = vec![vec![0.0; dimension]; cluster_count];
        let mut counts = vec![0usize; cluster_count];
        for (document, &cluster) in documents.iter().zip(&assignments) {
            counts[cluster] += 1;
            for (sum, coordinate) in next_centers[cluster].iter_mut().zip(&document.vector) {
                *sum += f64::from(*coordinate);
            }
        }
        for cluster in 0..cluster_count {
            if counts[cluster] == 0 {
                next_centers[cluster].clone_from(&centers[cluster]);
                continue;
            }
            for coordinate in &mut next_centers[cluster] {
                *coordinate /= counts[cluster] as f64;
            }
        }
        centers = next_centers;
    }
    assignments = capacity_bounded_assignments(&documents, &centers, capacity);

    let mut clusters: Vec<Vec<DenseDocument>> = (0..cluster_count)
        .map(|_| Vec::with_capacity(capacity))
        .collect();
    for (document, cluster) in documents.into_iter().zip(assignments) {
        clusters[cluster].push(document);
    }
    for cluster in &mut clusters {
        cluster.sort_unstable_by_key(|document| document.id);
    }
    clusters
}

fn farthest_first_centers(
    documents: &[DenseDocument],
    cluster_count: usize,
    dimension: usize,
) -> Vec<Vec<f64>> {
    let global_center = centroid(documents, dimension);
    let first = documents
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            OrderedFloat(squared_distance(&left.vector, &global_center))
                .cmp(&OrderedFloat(squared_distance(
                    &right.vector,
                    &global_center,
                )))
                .then_with(|| documents[*right_index].id.cmp(&documents[*left_index].id))
        })
        .map(|(index, _)| index)
        .unwrap();
    let mut selected = vec![false; documents.len()];
    selected[first] = true;
    let mut centers = vec![
        documents[first]
            .vector
            .iter()
            .map(|coordinate| f64::from(*coordinate))
            .collect::<Vec<_>>(),
    ];

    while centers.len() < cluster_count {
        let next = documents
            .iter()
            .enumerate()
            .filter(|(index, _)| !selected[*index])
            .max_by(|(left_index, left), (right_index, right)| {
                let left_distance = centers
                    .iter()
                    .map(|center| squared_distance(&left.vector, center))
                    .fold(f64::INFINITY, f64::min);
                let right_distance = centers
                    .iter()
                    .map(|center| squared_distance(&right.vector, center))
                    .fold(f64::INFINITY, f64::min);
                OrderedFloat(left_distance)
                    .cmp(&OrderedFloat(right_distance))
                    .then_with(|| documents[*right_index].id.cmp(&documents[*left_index].id))
            })
            .map(|(index, _)| index)
            .unwrap();
        selected[next] = true;
        centers.push(
            documents[next]
                .vector
                .iter()
                .map(|coordinate| f64::from(*coordinate))
                .collect(),
        );
    }
    centers
}

fn capacity_bounded_assignments(
    documents: &[DenseDocument],
    centers: &[Vec<f64>],
    capacity: usize,
) -> Vec<usize> {
    let mut distances = Vec::with_capacity(documents.len());
    for (document_index, document) in documents.iter().enumerate() {
        let mut ranked_centers: Vec<_> = centers
            .iter()
            .enumerate()
            .map(|(cluster, center)| (cluster, squared_distance(&document.vector, center)))
            .collect();
        ranked_centers.sort_unstable_by(|left, right| {
            OrderedFloat(left.1)
                .cmp(&OrderedFloat(right.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        let confidence = ranked_centers
            .get(1)
            .map(|second| second.1 - ranked_centers[0].1)
            .unwrap_or(f64::INFINITY);
        distances.push((document_index, confidence, ranked_centers));
    }
    distances.sort_unstable_by(|left, right| {
        OrderedFloat(right.1)
            .cmp(&OrderedFloat(left.1))
            .then_with(|| documents[left.0].id.cmp(&documents[right.0].id))
    });

    let mut sizes = vec![0usize; centers.len()];
    let mut assignments = vec![0usize; documents.len()];
    for (document_index, _, ranked_centers) in distances {
        let cluster = ranked_centers
            .into_iter()
            .find(|(cluster, _)| sizes[*cluster] < capacity)
            .map(|(cluster, _)| cluster)
            .unwrap();
        sizes[cluster] += 1;
        assignments[document_index] = cluster;
    }

    for empty_cluster in 0..centers.len() {
        if sizes[empty_cluster] != 0 {
            continue;
        }
        let donor_document = documents
            .iter()
            .enumerate()
            .filter(|(document_index, _)| sizes[assignments[*document_index]] > 1)
            .max_by(|(left_index, left), (right_index, right)| {
                let left_cluster = assignments[*left_index];
                let right_cluster = assignments[*right_index];
                OrderedFloat(squared_distance(&left.vector, &centers[left_cluster]))
                    .cmp(&OrderedFloat(squared_distance(
                        &right.vector,
                        &centers[right_cluster],
                    )))
                    .then_with(|| right.id.cmp(&left.id))
            })
            .map(|(document_index, _)| document_index)
            .expect("more documents than clusters guarantees a donor");
        let donor_cluster = assignments[donor_document];
        sizes[donor_cluster] -= 1;
        sizes[empty_cluster] = 1;
        assignments[donor_document] = empty_cluster;
    }
    assignments
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseBallTelemetry {
    pub blocks: usize,
    pub nodes: usize,
    pub bound_evaluations: usize,
    pub internal_nodes_expanded: usize,
    pub blocks_expanded: usize,
    pub documents_evaluated: usize,
    pub points_emitted: usize,
    pub max_pending_blocks: usize,
    pub max_pending_points: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NodeQueueEntry {
    node_index: usize,
    min_id: PointOffsetType,
    upper_bound: OrderedFloat<f64>,
}

impl Ord for NodeQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound
            .cmp(&other.upper_bound)
            .then_with(|| other.min_id.cmp(&self.min_id))
            .then_with(|| other.node_index.cmp(&self.node_index))
    }
}

impl PartialOrd for NodeQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointQueueEntry(DenseScoredPoint);

impl Eq for PointQueueEntry {}

impl Ord for PointQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.id.cmp(&self.0.id))
    }
}

impl PartialOrd for PointQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct DenseBallStream<'a> {
    index: &'a DenseBallIndex,
    query: Vec<f64>,
    query_norm: f64,
    pending_nodes: BinaryHeap<NodeQueueEntry>,
    pending_points: BinaryHeap<PointQueueEntry>,
    telemetry: DenseBallTelemetry,
}

impl<'a> DenseBallStream<'a> {
    fn new(index: &'a DenseBallIndex, query: &[f32]) -> Result<Self, DenseBallError> {
        let (query, query_norm) = index.query_components(query)?;

        let mut pending_nodes = BinaryHeap::with_capacity(index.roots.len());
        for &node_index in &index.roots {
            pending_nodes.push(node_queue_entry(
                node_index,
                &index.nodes[node_index],
                &query,
                query_norm,
                index.dimension,
            ));
        }
        let telemetry = DenseBallTelemetry {
            blocks: index.leaf_block_count,
            nodes: index.nodes.len(),
            bound_evaluations: index.roots.len(),
            max_pending_blocks: pending_nodes.len(),
            ..Default::default()
        };

        Ok(Self {
            index,
            query,
            query_norm,
            pending_nodes,
            pending_points: BinaryHeap::new(),
            telemetry,
        })
    }

    pub fn telemetry(&self) -> DenseBallTelemetry {
        self.telemetry
    }

    fn expand_next_node(&mut self) {
        let node_entry = self.pending_nodes.pop().unwrap();
        let node = &self.index.nodes[node_entry.node_index];
        match &node.kind {
            BallNodeKind::Internal { children } => {
                self.telemetry.internal_nodes_expanded += 1;
                self.telemetry.bound_evaluations += children.len();
                for &child_index in children {
                    self.pending_nodes.push(node_queue_entry(
                        child_index,
                        &self.index.nodes[child_index],
                        &self.query,
                        self.query_norm,
                        self.index.dimension,
                    ));
                }
                self.telemetry.max_pending_blocks = self
                    .telemetry
                    .max_pending_blocks
                    .max(self.pending_nodes.len());
            }
            BallNodeKind::Leaf {
                document_start,
                document_end,
            } => {
                self.telemetry.blocks_expanded += 1;
                for document in &self.index.documents[*document_start..*document_end] {
                    self.telemetry.documents_evaluated += 1;
                    let score = self
                        .query
                        .iter()
                        .zip(&document.vector)
                        .map(|(query, coordinate)| query * f64::from(*coordinate))
                        .sum();
                    self.pending_points.push(PointQueueEntry(DenseScoredPoint {
                        id: document.id,
                        score,
                    }));
                }
                self.telemetry.max_pending_points = self
                    .telemetry
                    .max_pending_points
                    .max(self.pending_points.len());
            }
        }
    }

    fn next_point_is_fixed(&self) -> bool {
        let Some(point) = self.pending_points.peek() else {
            return false;
        };
        let Some(node) = self.pending_nodes.peek() else {
            return true;
        };

        point.0.score > node.upper_bound.0
            || (point.0.score == node.upper_bound.0 && point.0.id < node.min_id)
    }
}

impl Iterator for DenseBallStream<'_> {
    type Item = DenseScoredPoint;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_point_is_fixed() {
                self.telemetry.points_emitted += 1;
                return self.pending_points.pop().map(|point| point.0);
            }
            if self.pending_nodes.is_empty() {
                let point = self.pending_points.pop()?.0;
                self.telemetry.points_emitted += 1;
                return Some(point);
            }
            self.expand_next_node();
        }
    }
}

fn node_queue_entry(
    node_index: usize,
    node: &BallNode,
    query: &[f64],
    query_norm: f64,
    dimension: usize,
) -> NodeQueueEntry {
    let center_score = dot(query, &node.center);
    let euclidean_upper_bound = if node.radius == 0.0 {
        center_score
    } else {
        let radius_contribution = query_norm * node.radius;
        let raw_bound = center_score + radius_contribution;
        let rounding_margin = f64::EPSILON
            * (center_score.abs() + radius_contribution.abs() + 1.0)
            * (dimension as f64 + 4.0);
        (raw_bound + rounding_margin).next_up()
    };
    let cap = &node.spherical_cap;
    let query_direction_cosine = (dot(query, &cap.direction) / query_norm).clamp(-1.0, 1.0);
    let directional_upper_bound = if query_direction_cosine >= cap.min_cosine {
        1.0
    } else {
        let orthogonal_query = (1.0 - query_direction_cosine.powi(2)).max(0.0).sqrt();
        let orthogonal_cap = (1.0 - cap.min_cosine.powi(2)).max(0.0).sqrt();
        query_direction_cosine * cap.min_cosine + orthogonal_query * orthogonal_cap
    }
    .clamp(-1.0, 1.0);
    let norm = if directional_upper_bound >= 0.0 {
        cap.max_norm
    } else {
        cap.min_norm
    };
    let cap_raw_bound = query_norm * norm * directional_upper_bound;
    let cap_rounding_margin = f64::EPSILON
        * (cap_raw_bound.abs() + query_norm * cap.max_norm + 1.0)
        * (dimension as f64 + 12.0)
        * 8.0;
    let mut cap_upper_bound = (cap_raw_bound + cap_rounding_margin).next_up();
    if cap.contains_zero {
        cap_upper_bound = cap_upper_bound.max(0.0);
    }
    let upper_bound = euclidean_upper_bound.min(cap_upper_bound);
    NodeQueueEntry {
        node_index,
        min_id: node.min_id,
        upper_bound: OrderedFloat(upper_bound),
    }
}

fn centroid(documents: &[DenseDocument], dimension: usize) -> Vec<f64> {
    let mut center = vec![0.0; dimension];
    for document in documents {
        for (sum, coordinate) in center.iter_mut().zip(&document.vector) {
            *sum += f64::from(*coordinate);
        }
    }
    for coordinate in &mut center {
        *coordinate /= documents.len() as f64;
    }
    center
}

fn spherical_cap(documents: &[DenseDocument], dimension: usize) -> SphericalCap {
    let mut unit_vectors = Vec::with_capacity(documents.len());
    let mut direction = vec![0.0; dimension];
    let mut min_norm = f64::INFINITY;
    let mut max_norm = 0.0f64;
    let mut contains_zero = false;

    for document in documents {
        let norm = document
            .vector
            .iter()
            .map(|coordinate| f64::from(*coordinate).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            contains_zero = true;
            continue;
        }
        min_norm = min_norm.min(norm);
        max_norm = max_norm.max(norm);
        let unit: Vec<_> = document
            .vector
            .iter()
            .map(|coordinate| f64::from(*coordinate) / norm)
            .collect();
        for (sum, coordinate) in direction.iter_mut().zip(&unit) {
            *sum += coordinate;
        }
        unit_vectors.push(unit);
    }

    if unit_vectors.is_empty() {
        return SphericalCap {
            direction,
            min_cosine: 1.0,
            min_norm: 0.0,
            max_norm: 0.0,
            contains_zero: true,
        };
    }

    let direction_norm = dot(&direction, &direction).sqrt();
    if direction_norm == 0.0 {
        direction.clone_from(&unit_vectors[0]);
    } else {
        for coordinate in &mut direction {
            *coordinate /= direction_norm;
        }
    }
    let rounding_margin = f64::EPSILON * (dimension as f64 + 4.0) * 8.0;
    let min_cosine = unit_vectors
        .iter()
        .map(|unit| dot(unit, &direction))
        .fold(1.0, f64::min)
        .clamp(-1.0, 1.0);

    SphericalCap {
        direction,
        min_cosine: (min_cosine - rounding_margin).clamp(-1.0, 1.0).next_down(),
        min_norm: (min_norm - rounding_margin).max(0.0).next_down(),
        max_norm: (max_norm + rounding_margin).next_up(),
        contains_zero,
    }
}

fn euclidean_distance(vector: &[f32], center: &[f64]) -> f64 {
    squared_distance(vector, center).sqrt()
}

fn squared_distance(vector: &[f32], center: &[f64]) -> f64 {
    vector
        .iter()
        .zip(center)
        .map(|(coordinate, center)| {
            let difference = f64::from(*coordinate) - center;
            difference * difference
        })
        .sum::<f64>()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

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
                    .map(|(coordinate, query)| f64::from(*coordinate) * f64::from(*query))
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

    fn generated_case() -> impl Strategy<Value = (Vec<DenseDocument>, Vec<f32>, usize)> {
        (1usize..48, 1usize..9)
            .prop_flat_map(|(document_count, dimension)| {
                (
                    Just(document_count),
                    Just(dimension),
                    prop::collection::vec(-5i8..6, document_count * dimension),
                    prop::collection::vec(-5i8..6, dimension)
                        .prop_filter("query must be non-zero", |query| {
                            query.iter().any(|coordinate| *coordinate != 0)
                        }),
                    1usize..document_count + 1,
                )
            })
            .prop_map(
                |(document_count, dimension, coordinates, query, block_size)| {
                    let documents = (0..document_count)
                        .map(|document| DenseDocument {
                            id: document as PointOffsetType,
                            vector: coordinates[document * dimension..(document + 1) * dimension]
                                .iter()
                                .map(|coordinate| f32::from(*coordinate))
                                .collect(),
                        })
                        .collect();
                    (
                        documents,
                        query.into_iter().map(f32::from).collect(),
                        block_size,
                    )
                },
            )
    }

    #[test]
    fn rejects_zero_query() {
        let index = DenseBallIndex::build(
            vec![DenseDocument {
                id: 0,
                vector: vec![1.0, 2.0],
            }],
            1,
        )
        .unwrap();

        assert!(matches!(
            index.stream(&[0.0, 0.0]),
            Err(DenseBallError::InvalidQuery)
        ));
    }

    #[test]
    fn preserves_identity_order_for_equal_scores() {
        let documents: Vec<_> = (0..16)
            .rev()
            .map(|id| DenseDocument {
                id,
                vector: vec![1.0, 1.0],
            })
            .collect();
        let index = DenseBallIndex::build(documents, 3).unwrap();

        let mut prefix_stream = index.stream(&[1.0, -1.0]).unwrap();
        let prefix: Vec<_> = prefix_stream.by_ref().take(3).collect();
        assert_eq!(prefix_stream.telemetry().blocks_expanded, 1);
        assert_eq!(
            prefix.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );

        let actual: Vec<_> = index.stream(&[1.0, -1.0]).unwrap().collect();

        assert_eq!(
            actual.iter().map(|point| point.id).collect::<Vec<_>>(),
            (0..16).collect::<Vec<_>>()
        );
    }

    #[test]
    fn clustered_build_preserves_exact_order_and_capacity() {
        let documents: Vec<_> = (0..37)
            .map(|id| DenseDocument {
                id,
                vector: vec![
                    (id % 5) as f32,
                    ((id * 7) % 11) as f32,
                    ((id * 13) % 17) as f32,
                ],
            })
            .collect();
        let query = vec![0.75, -0.5, 1.25];
        let expected = exhaustive(&documents, &query);

        let index = DenseBallIndex::build_clustered(documents, 8, 4).unwrap();
        let actual: Vec<_> = index.stream(&query).unwrap().collect();

        assert_eq!(index.block_count(), 5);
        assert_eq!(actual, expected);
    }

    #[test]
    fn hierarchical_build_refines_internal_nodes_before_leaves() {
        let documents: Vec<_> = (0..64)
            .map(|id| DenseDocument {
                id,
                vector: vec![(id / 16) as f32, (id % 16) as f32],
            })
            .collect();
        let query = vec![1.0, -0.25];
        let expected = exhaustive(&documents, &query);
        let index = DenseBallIndex::build_hierarchical_clustered(documents, 4, 4, 3).unwrap();

        let mut stream = index.stream(&query).unwrap();
        let actual: Vec<_> = stream.by_ref().collect();

        assert!(index.node_count() > index.block_count());
        assert!(stream.telemetry().internal_nodes_expanded > 0);
        assert_eq!(actual, expected);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1_000))]

        #[test]
        fn dense_ball_stream_matches_exhaustive_order(
            (documents, query, block_size) in generated_case()
        ) {
            let expected = exhaustive(&documents, &query);
            let index = DenseBallIndex::build(documents, block_size).unwrap();
            let actual: Vec<_> = index.stream(&query).unwrap().collect();

            prop_assert_eq!(actual, expected);
        }


        #[test]
        fn clustered_dense_ball_stream_matches_exhaustive_order(
            (documents, query, block_size) in generated_case()
        ) {
            let expected = exhaustive(&documents, &query);
            let index = DenseBallIndex::build_clustered(documents, block_size, 3).unwrap();
            let actual: Vec<_> = index.stream(&query).unwrap().collect();

            prop_assert_eq!(actual, expected);
        }


        #[test]
        fn hierarchical_dense_ball_stream_matches_exhaustive_order(
            (documents, query, block_size) in generated_case()
        ) {
            let expected = exhaustive(&documents, &query);
            let index = DenseBallIndex::build_hierarchical_clustered(
                documents,
                block_size,
                4,
                3,
            ).unwrap();
            let actual: Vec<_> = index.stream(&query).unwrap().collect();

            prop_assert_eq!(actual, expected);
        }
    }
}
