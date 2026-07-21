// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::error::Error;
use std::fmt::{Display, Formatter};

use common::types::{PointOffsetType, ScoredPointOffset};
use ordered_float::OrderedFloat;
use serde::Serialize;

use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimOffset, DimWeight};

#[derive(Clone, Debug, PartialEq)]
pub struct SparseDocument {
    pub id: PointOffsetType,
    pub vector: RemappedSparseVector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockMaxError {
    ZeroBlockSize,
    DuplicateDocument(PointOffsetType),
    InvalidDocument(PointOffsetType),
    InvalidQuery,
}

impl Display for BlockMaxError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBlockSize => formatter.write_str("Block Max block size must be positive"),
            Self::DuplicateDocument(id) => {
                write!(formatter, "Block Max document {id} occurs more than once")
            }
            Self::InvalidDocument(id) => write!(
                formatter,
                "Block Max document {id} must have sorted unique dimensions and finite non-negative impacts"
            ),
            Self::InvalidQuery => formatter.write_str(
                "Block Max query must have sorted unique dimensions and finite non-negative weights",
            ),
        }
    }
}

impl Error for BlockMaxError {}

#[derive(Clone, Debug)]
struct Block {
    document_start: usize,
    document_end: usize,
    min_id: PointOffsetType,
    envelope: RemappedSparseVector,
}

#[derive(Clone, Debug)]
pub struct BlockMaxIndex {
    documents: Vec<SparseDocument>,
    blocks: Vec<Block>,
    block_size: usize,
}

impl BlockMaxIndex {
    pub fn build(
        mut documents: Vec<SparseDocument>,
        block_size: usize,
    ) -> Result<Self, BlockMaxError> {
        if block_size == 0 {
            return Err(BlockMaxError::ZeroBlockSize);
        }
        for document in &documents {
            if !valid_non_negative_vector(&document.vector) {
                return Err(BlockMaxError::InvalidDocument(document.id));
            }
        }

        documents.sort_unstable_by_key(|document| document.id);
        if let Some(duplicate) = documents
            .array_windows()
            .find(|[left, right]| left.id == right.id)
        {
            return Err(BlockMaxError::DuplicateDocument(duplicate[0].id));
        }

        let blocks = documents
            .chunks(block_size)
            .enumerate()
            .map(|(block_index, documents)| {
                let document_start = block_index * block_size;
                let document_end = document_start + documents.len();
                Block {
                    document_start,
                    document_end,
                    min_id: documents[0].id,
                    envelope: envelope(documents),
                }
            })
            .collect();

        Ok(Self {
            documents,
            blocks,
            block_size,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn envelope_nonzero_count(&self) -> usize {
        self.blocks.iter().map(|block| block.envelope.len()).sum()
    }

    pub fn stream(&self, query: RemappedSparseVector) -> Result<BlockMaxStream<'_>, BlockMaxError> {
        BlockMaxStream::new(self, query)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct BlockMaxTelemetry {
    pub blocks: usize,
    pub bound_evaluations: usize,
    pub zero_bound_blocks: usize,
    pub blocks_expanded: usize,
    pub documents_evaluated: usize,
    pub points_emitted: usize,
    pub max_pending_blocks: usize,
    pub max_pending_points: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockQueueEntry {
    block_index: usize,
    min_id: PointOffsetType,
    upper_bound: OrderedFloat<f32>,
}

impl Ord for BlockQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound
            .cmp(&other.upper_bound)
            .then_with(|| other.min_id.cmp(&self.min_id))
            .then_with(|| other.block_index.cmp(&self.block_index))
    }
}

impl PartialOrd for BlockQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointQueueEntry(ScoredPointOffset);

impl Eq for PointQueueEntry {}

impl Ord for PointQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.idx.cmp(&self.0.idx))
    }
}

impl PartialOrd for PointQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct BlockMaxStream<'a> {
    index: &'a BlockMaxIndex,
    query: RemappedSparseVector,
    pending_blocks: BinaryHeap<BlockQueueEntry>,
    pending_points: BinaryHeap<PointQueueEntry>,
    telemetry: BlockMaxTelemetry,
}

impl<'a> BlockMaxStream<'a> {
    fn new(index: &'a BlockMaxIndex, query: RemappedSparseVector) -> Result<Self, BlockMaxError> {
        if !valid_non_negative_vector(&query) {
            return Err(BlockMaxError::InvalidQuery);
        }

        let mut pending_blocks = BinaryHeap::with_capacity(index.blocks.len());
        let mut telemetry = BlockMaxTelemetry {
            blocks: index.blocks.len(),
            bound_evaluations: index.blocks.len(),
            ..Default::default()
        };
        for (block_index, block) in index.blocks.iter().enumerate() {
            let upper_bound = block.envelope.score(&query).unwrap_or_default();
            if upper_bound == 0.0 {
                telemetry.zero_bound_blocks += 1;
                continue;
            }
            pending_blocks.push(BlockQueueEntry {
                block_index,
                min_id: block.min_id,
                upper_bound: OrderedFloat(upper_bound),
            });
        }
        telemetry.max_pending_blocks = pending_blocks.len();

        Ok(Self {
            index,
            query,
            pending_blocks,
            pending_points: BinaryHeap::new(),
            telemetry,
        })
    }

    pub fn telemetry(&self) -> BlockMaxTelemetry {
        self.telemetry
    }

    fn expand_next_block(&mut self) {
        let block_entry = self.pending_blocks.pop().unwrap();
        let block = &self.index.blocks[block_entry.block_index];
        self.telemetry.blocks_expanded += 1;

        for document in &self.index.documents[block.document_start..block.document_end] {
            self.telemetry.documents_evaluated += 1;
            if let Some(score) = document.vector.score(&self.query)
                && score != 0.0
            {
                self.pending_points.push(PointQueueEntry(ScoredPointOffset {
                    idx: document.id,
                    score,
                }));
            }
        }
        self.telemetry.max_pending_points = self
            .telemetry
            .max_pending_points
            .max(self.pending_points.len());
    }

    fn next_point_is_fixed(&self) -> bool {
        let Some(point) = self.pending_points.peek() else {
            return false;
        };
        let Some(block) = self.pending_blocks.peek() else {
            return true;
        };

        point.0.score > block.upper_bound.0
            || (point.0.score == block.upper_bound.0 && point.0.idx < block.min_id)
    }
}

impl Iterator for BlockMaxStream<'_> {
    type Item = ScoredPointOffset;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_point_is_fixed() {
                self.telemetry.points_emitted += 1;
                return self.pending_points.pop().map(|point| point.0);
            }
            if self.pending_blocks.is_empty() {
                let point = self.pending_points.pop()?.0;
                self.telemetry.points_emitted += 1;
                return Some(point);
            }
            self.expand_next_block();
        }
    }
}

fn valid_non_negative_vector(vector: &RemappedSparseVector) -> bool {
    vector.is_sorted()
        && vector
            .values
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
}

fn envelope(documents: &[SparseDocument]) -> RemappedSparseVector {
    let mut maxima = BTreeMap::<DimOffset, DimWeight>::new();
    for document in documents {
        for (&dimension, &weight) in document.vector.indices.iter().zip(&document.vector.values) {
            maxima
                .entry(dimension)
                .and_modify(|maximum| *maximum = maximum.max(weight))
                .or_insert(weight);
        }
    }
    let (indices, values) = maxima.into_iter().unzip();
    RemappedSparseVector { indices, values }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn exhaustive(
        documents: &[SparseDocument],
        query: &RemappedSparseVector,
    ) -> Vec<ScoredPointOffset> {
        let mut points: Vec<_> = documents
            .iter()
            .filter_map(|document| {
                document.vector.score(query).and_then(|score| {
                    (score != 0.0).then_some(ScoredPointOffset {
                        idx: document.id,
                        score,
                    })
                })
            })
            .collect();
        points.sort_unstable_by(|left, right| {
            OrderedFloat(right.score)
                .cmp(&OrderedFloat(left.score))
                .then_with(|| left.idx.cmp(&right.idx))
        });
        points
    }

    fn generated_case() -> impl Strategy<Value = (Vec<SparseDocument>, RemappedSparseVector, usize)>
    {
        (1usize..48, 1usize..16)
            .prop_flat_map(|(document_count, dimensions)| {
                (
                    Just(document_count),
                    Just(dimensions),
                    prop::collection::vec(0u8..6, document_count * dimensions),
                    prop::collection::vec(0u8..6, dimensions),
                    1usize..document_count + 1,
                )
            })
            .prop_map(
                |(document_count, dimensions, document_weights, query_weights, block_size)| {
                    let documents = (0..document_count)
                        .map(|document| {
                            let weights = &document_weights
                                [document * dimensions..(document + 1) * dimensions];
                            let (indices, values) = weights
                                .iter()
                                .enumerate()
                                .filter(|(_, weight)| **weight != 0)
                                .map(|(dimension, weight)| (dimension as u32, f32::from(*weight)))
                                .unzip();
                            SparseDocument {
                                id: document as PointOffsetType,
                                vector: RemappedSparseVector { indices, values },
                            }
                        })
                        .collect();
                    let (indices, values) = query_weights
                        .into_iter()
                        .enumerate()
                        .filter(|(_, weight)| *weight != 0)
                        .map(|(dimension, weight)| (dimension as u32, f32::from(weight)))
                        .unzip();
                    (
                        documents,
                        RemappedSparseVector { indices, values },
                        block_size,
                    )
                },
            )
    }

    #[test]
    fn preserves_identity_order_across_tied_blocks() {
        let documents = vec![
            SparseDocument {
                id: 2,
                vector: RemappedSparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
            },
            SparseDocument {
                id: 1,
                vector: RemappedSparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
            },
        ];
        let index = BlockMaxIndex::build(documents, 1).unwrap();
        let query = RemappedSparseVector {
            indices: vec![0],
            values: vec![1.0],
        };

        let points: Vec<_> = index.stream(query).unwrap().collect();

        assert_eq!(points[0].idx, 1);
        assert_eq!(points[1].idx, 2);
    }

    #[test]
    fn skips_zero_bound_blocks_without_scoring_documents() {
        let documents = vec![
            SparseDocument {
                id: 0,
                vector: RemappedSparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
            },
            SparseDocument {
                id: 1,
                vector: RemappedSparseVector {
                    indices: vec![1],
                    values: vec![1.0],
                },
            },
        ];
        let index = BlockMaxIndex::build(documents, 1).unwrap();
        let query = RemappedSparseVector {
            indices: vec![0],
            values: vec![1.0],
        };
        let mut stream = index.stream(query).unwrap();

        assert_eq!(stream.next().unwrap().idx, 0);
        assert!(stream.next().is_none());
        assert_eq!(stream.telemetry().zero_bound_blocks, 1);
        assert_eq!(stream.telemetry().documents_evaluated, 1);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1_000))]

        #[test]
        fn block_max_stream_matches_exhaustive_order(
            (documents, query, block_size) in generated_case()
        ) {
            let expected = exhaustive(&documents, &query);
            let index = BlockMaxIndex::build(documents, block_size).unwrap();
            let actual: Vec<_> = index.stream(query).unwrap().collect();

            prop_assert_eq!(actual, expected);
        }
    }
}
