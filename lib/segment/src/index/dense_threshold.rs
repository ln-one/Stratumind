// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact Dense rank streams from sorted coordinate access and a threshold certificate.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use common::types::PointOffsetType;
use ordered_float::OrderedFloat;

use super::dense_ball::{DenseDocument, DenseScoredPoint};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseThresholdError {
    EmptyDocuments,
    DuplicateDocument(PointOffsetType),
    InvalidDocument(PointOffsetType),
    InvalidQuery,
    TooManyDocuments,
}

impl Display for DenseThresholdError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDocuments => {
                formatter.write_str("Dense Threshold requires at least one document")
            }
            Self::DuplicateDocument(id) => {
                write!(formatter, "Dense Threshold document {id} occurs more than once")
            }
            Self::InvalidDocument(id) => write!(
                formatter,
                "Dense Threshold document {id} has a mismatched dimension or non-finite coordinate",
            ),
            Self::InvalidQuery => formatter.write_str(
                "Dense Threshold query must match the index dimension and contain finite coordinates",
            ),
            Self::TooManyDocuments => formatter.write_str(
                "Dense Threshold requires the document count to fit in a 32-bit ordinal",
            ),
        }
    }
}

impl Error for DenseThresholdError {}

#[derive(Clone, Debug)]
pub struct DenseThresholdIndex {
    documents: Vec<DenseDocument>,
    /// One descending document-ordinal list per coordinate.
    coordinate_orders: Vec<Vec<u32>>,
    dimension: usize,
}

impl DenseThresholdIndex {
    pub fn build(mut documents: Vec<DenseDocument>) -> Result<Self, DenseThresholdError> {
        let Some(first) = documents.first() else {
            return Err(DenseThresholdError::EmptyDocuments);
        };
        let dimension = first.vector.len();
        if dimension == 0 {
            return Err(DenseThresholdError::InvalidDocument(first.id));
        }
        if documents.len() > u32::MAX as usize {
            return Err(DenseThresholdError::TooManyDocuments);
        }
        let mut identities = HashSet::with_capacity(documents.len());
        for document in &documents {
            if !identities.insert(document.id) {
                return Err(DenseThresholdError::DuplicateDocument(document.id));
            }
            if document.vector.len() != dimension
                || document.vector.iter().any(|value| !value.is_finite())
            {
                return Err(DenseThresholdError::InvalidDocument(document.id));
            }
        }
        documents.sort_unstable_by_key(|document| document.id);

        let mut coordinate_orders = Vec::with_capacity(dimension);
        for coordinate in 0..dimension {
            let mut order: Vec<_> = (0..documents.len() as u32).collect();
            order.sort_unstable_by(|&left, &right| {
                OrderedFloat(documents[right as usize].vector[coordinate])
                    .cmp(&OrderedFloat(documents[left as usize].vector[coordinate]))
                    .then_with(|| {
                        documents[left as usize]
                            .id
                            .cmp(&documents[right as usize].id)
                    })
            });
            coordinate_orders.push(order);
        }

        Ok(Self {
            documents,
            coordinate_orders,
            dimension,
        })
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn auxiliary_bytes(&self) -> usize {
        self.coordinate_orders
            .iter()
            .map(|order| order.len() * size_of::<u32>())
            .sum()
    }

    pub fn stream(&self, query: &[f32]) -> Result<DenseThresholdStream<'_>, DenseThresholdError> {
        DenseThresholdStream::new(self, query)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseThresholdTelemetry {
    pub sequential_accesses: usize,
    pub duplicate_accesses: usize,
    pub exact_scores: usize,
    pub threshold_updates: usize,
    pub points_emitted: usize,
}

#[derive(Clone, Copy, Debug)]
struct ActiveCoordinate {
    coordinate: usize,
    query_value: f64,
    cursor: usize,
    reverse: bool,
}

impl ActiveCoordinate {
    fn ordinal(self, index: &DenseThresholdIndex) -> usize {
        let order = &index.coordinate_orders[self.coordinate];
        let position = if self.reverse {
            order.len() - 1 - self.cursor
        } else {
            self.cursor
        };
        order[position] as usize
    }

    fn contribution(self, index: &DenseThresholdIndex) -> f64 {
        self.query_value * f64::from(index.documents[self.ordinal(index)].vector[self.coordinate])
    }

    fn next_contribution(self, index: &DenseThresholdIndex) -> Option<f64> {
        (self.cursor + 1 < index.document_count()).then(|| {
            let mut next = self;
            next.cursor += 1;
            next.contribution(index)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AdvanceChoice {
    active_coordinate: usize,
    coordinate: usize,
    threshold_drop: f64,
}

impl Eq for AdvanceChoice {}

impl Ord for AdvanceChoice {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.threshold_drop)
            .cmp(&OrderedFloat(other.threshold_drop))
            .then_with(|| other.coordinate.cmp(&self.coordinate))
    }
}

impl PartialOrd for AdvanceChoice {
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

pub struct DenseThresholdStream<'a> {
    index: &'a DenseThresholdIndex,
    query: Vec<f64>,
    active_coordinates: Vec<ActiveCoordinate>,
    advance_choices: BinaryHeap<AdvanceChoice>,
    pending_exact: BinaryHeap<ExactCandidate>,
    seen: Vec<bool>,
    unseen_upper_bound: f64,
    all_seen: bool,
    zero_query_cursor: Option<usize>,
    telemetry: DenseThresholdTelemetry,
}

impl<'a> DenseThresholdStream<'a> {
    fn new(index: &'a DenseThresholdIndex, query: &[f32]) -> Result<Self, DenseThresholdError> {
        if query.len() != index.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(DenseThresholdError::InvalidQuery);
        }
        let query: Vec<_> = query.iter().map(|value| f64::from(*value)).collect();
        let mut active_coordinates = Vec::new();
        let mut unseen_upper_bound = 0.0;
        for (coordinate, &query_value) in query.iter().enumerate() {
            if query_value == 0.0 {
                continue;
            }
            let active = ActiveCoordinate {
                coordinate,
                query_value,
                cursor: 0,
                reverse: query_value < 0.0,
            };
            unseen_upper_bound = upper_add(unseen_upper_bound, active.contribution(index));
            active_coordinates.push(active);
        }
        let zero_query_cursor = active_coordinates.is_empty().then_some(0);
        let advance_choices = active_coordinates
            .iter()
            .copied()
            .enumerate()
            .map(|(active_coordinate, active)| AdvanceChoice {
                active_coordinate,
                coordinate: active.coordinate,
                threshold_drop: threshold_drop(active, index),
            })
            .collect();

        Ok(Self {
            index,
            query,
            active_coordinates,
            advance_choices,
            pending_exact: BinaryHeap::new(),
            seen: vec![false; index.document_count()],
            unseen_upper_bound,
            all_seen: false,
            zero_query_cursor,
            telemetry: DenseThresholdTelemetry::default(),
        })
    }

    pub fn telemetry(&self) -> DenseThresholdTelemetry {
        self.telemetry
    }

    fn next_point_is_fixed(&self) -> bool {
        let Some(candidate) = self.pending_exact.peek() else {
            return false;
        };
        self.all_seen || candidate.0.score > self.unseen_upper_bound
    }

    fn advance(&mut self) {
        let choice = self
            .advance_choices
            .pop()
            .expect("a non-zero query has an unfinished coordinate until all points are seen");
        let active = &mut self.active_coordinates[choice.active_coordinate];
        let old_frontier = active.contribution(self.index);
        let ordinal = active.ordinal(self.index);
        self.telemetry.sequential_accesses += 1;
        if self.seen[ordinal] {
            self.telemetry.duplicate_accesses += 1;
        } else {
            self.seen[ordinal] = true;
            let document = &self.index.documents[ordinal];
            let score = document
                .vector
                .iter()
                .zip(&self.query)
                .map(|(left, right)| f64::from(*left) * *right)
                .sum();
            self.telemetry.exact_scores += 1;
            self.pending_exact.push(ExactCandidate(DenseScoredPoint {
                id: document.id,
                score,
            }));
        }

        active.cursor += 1;
        self.telemetry.threshold_updates += 1;
        if active.cursor == self.index.document_count() {
            // One complete coordinate list contains every document identity.
            self.all_seen = true;
            self.unseen_upper_bound = f64::NEG_INFINITY;
            return;
        }
        let new_frontier = active.contribution(self.index);
        self.unseen_upper_bound = upper_add(
            upper_add(self.unseen_upper_bound, -old_frontier),
            new_frontier,
        );
        self.advance_choices.push(AdvanceChoice {
            active_coordinate: choice.active_coordinate,
            coordinate: active.coordinate,
            threshold_drop: threshold_drop(*active, self.index),
        });
    }
}

impl Iterator for DenseThresholdStream<'_> {
    type Item = DenseScoredPoint;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(cursor) = &mut self.zero_query_cursor {
            let document = self.index.documents.get(*cursor)?;
            *cursor += 1;
            self.telemetry.points_emitted += 1;
            return Some(DenseScoredPoint {
                id: document.id,
                score: 0.0,
            });
        }

        loop {
            if self.next_point_is_fixed() {
                self.telemetry.points_emitted += 1;
                return self.pending_exact.pop().map(|candidate| candidate.0);
            }
            if self.all_seen {
                let candidate = self.pending_exact.pop()?.0;
                self.telemetry.points_emitted += 1;
                return Some(candidate);
            }
            self.advance();
        }
    }
}

fn upper_add(left: f64, right: f64) -> f64 {
    (left + right).next_up()
}

fn threshold_drop(active: ActiveCoordinate, index: &DenseThresholdIndex) -> f64 {
    match active.next_contribution(index) {
        Some(next) => (active.contribution(index) - next).max(0.0),
        None => f64::INFINITY,
    }
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

    fn generated_case() -> impl Strategy<Value = (Vec<DenseDocument>, Vec<f32>)> {
        (1usize..48, 1usize..9)
            .prop_flat_map(|(document_count, dimension)| {
                (
                    Just(document_count),
                    Just(dimension),
                    prop::collection::vec(-1_000.0f32..1_000.0, document_count * dimension),
                    prop::collection::vec(-1_000.0f32..1_000.0, dimension),
                )
            })
            .prop_map(|(document_count, dimension, coordinates, query)| {
                let documents = (0..document_count)
                    .map(|document| DenseDocument {
                        id: document as PointOffsetType,
                        vector: coordinates[document * dimension..(document + 1) * dimension]
                            .to_vec(),
                    })
                    .collect();
                (documents, query)
            })
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
            let index = DenseThresholdIndex::build(documents.clone()).unwrap();

            let actual: Vec<_> = index.stream(&query).unwrap().collect();

            assert_eq!(actual, exhaustive(&documents, &query));
        }
    }

    #[test]
    fn separated_coordinate_emits_without_scoring_the_corpus() {
        let documents: Vec<_> = (0..1_000)
            .map(|id| DenseDocument {
                id,
                vector: vec![id as f32, 1.0],
            })
            .collect();
        let index = DenseThresholdIndex::build(documents).unwrap();
        let mut stream = index.stream(&[1.0, 0.0]).unwrap();

        let top: Vec<_> = stream.by_ref().take(20).collect();

        assert_eq!(top[0].id, 999);
        assert_eq!(stream.telemetry().exact_scores, 20);
        assert_eq!(stream.telemetry().sequential_accesses, 20);
    }

    #[test]
    fn ties_fall_back_to_identity_order() {
        let documents: Vec<_> = [7, 3, 11, 2]
            .into_iter()
            .map(|id| DenseDocument {
                id,
                vector: vec![1.0, -2.0],
            })
            .collect();
        let index = DenseThresholdIndex::build(documents).unwrap();

        let points: Vec<_> = index.stream(&[1.0, 1.0]).unwrap().collect();

        assert_eq!(
            points.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![2, 3, 7, 11]
        );
    }

    #[test]
    fn zero_query_uses_frozen_identity_order_without_coordinate_access() {
        let documents: Vec<_> = [7, 3, 11, 2]
            .into_iter()
            .map(|id| DenseDocument {
                id,
                vector: vec![id as f32, 1.0],
            })
            .collect();
        let index = DenseThresholdIndex::build(documents).unwrap();
        let mut stream = index.stream(&[0.0, -0.0]).unwrap();

        let points: Vec<_> = stream.by_ref().collect();

        assert_eq!(
            points.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![2, 3, 7, 11]
        );
        assert_eq!(stream.telemetry().sequential_accesses, 0);
        assert_eq!(stream.telemetry().exact_scores, 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1_000))]

        #[test]
        fn threshold_stream_matches_exhaustive_order((documents, query) in generated_case()) {
            let expected = exhaustive(&documents, &query);
            let index = DenseThresholdIndex::build(documents).unwrap();
            let actual: Vec<_> = index.stream(&query).unwrap().collect();

            prop_assert_eq!(actual, expected);
        }
    }
}
