// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Sparse stream merged across a frozen set of Segments.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use parking_lot::Mutex;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::modifier::Modifier;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::types::{Filter, VectorNameBuf};
use sparse::common::sparse_vector::SparseVector;

use crate::exact_rank_fallback::materialized_exact_source;
use crate::exact_shard_stream::{
    BatchReply, ExactShardStreamTelemetry, ExactSourceMode, SegmentRankPlan, SegmentScoreSource,
};
use crate::locked_segment::LockedSegment;

pub type SparseShardTelemetry = ExactShardStreamTelemetry;

#[derive(Clone)]
pub struct SparseRankPlan {
    vector_name: VectorNameBuf,
    query: SparseVector,
    filter: Option<Filter>,
    posting_batch_size: usize,
    query_context: Arc<QueryContext>,
}

impl SparseRankPlan {
    pub fn new(
        segments: &[LockedSegment],
        vector_name: VectorNameBuf,
        query: SparseVector,
        filter: Option<Filter>,
        posting_batch_size: usize,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<Self> {
        if posting_batch_size == 0 {
            return Err(OperationError::validation_error(
                "exact Sparse posting batch size must be positive",
            ));
        }
        let query_context = Arc::new(build_query_context(
            segments,
            &vector_name,
            &query,
            stopped,
        )?);
        Ok(Self {
            vector_name,
            query,
            filter,
            posting_batch_size,
            query_context,
        })
    }
}

impl SegmentRankPlan for SparseRankPlan {
    const CHANNEL: &'static str = "Sparse";

    fn open_segment(
        &self,
        _source: usize,
        segment: LockedSegment,
        _stopped: Arc<AtomicBool>,
    ) -> OperationResult<SegmentScoreSource> {
        let vector_name = self.vector_name.clone();
        let query = self.query.clone();
        let filter = self.filter.clone();
        let posting_batch_size = self.posting_batch_size;
        let query_context = self.query_context.clone();
        match segment {
            LockedSegment::Original(segment) => {
                let state = {
                    let segment = segment.read();
                    let segment_query_context = query_context.get_segment_query_context();
                    segment.with_view(|view| {
                        view.open_exact_sparse_rank_state(
                            &vector_name,
                            &query,
                            posting_batch_size,
                            &segment_query_context,
                        )
                    })
                };
                let state = match state {
                    Ok(state) => state,
                    Err(OperationError::WrongSparse) => {
                        let segment = segment.read();
                        let query_vector: QueryVector = VectorInternal::from(query).into();
                        return materialized_exact_source(
                            &*segment,
                            &vector_name,
                            &query_vector,
                            filter.as_ref(),
                            &query_context,
                        );
                    }
                    Err(error) => return Err(error),
                };
                let state = Arc::new(Mutex::new(state));
                Ok(SegmentScoreSource::on_demand(
                    move |limit| {
                        let segment = segment.read();
                        let mut state = state.lock();
                        let segment_query_context = query_context.get_segment_query_context();
                        segment
                            .with_view(|view| {
                                view.advance_exact_sparse_rank_state(
                                    &vector_name,
                                    filter.as_ref(),
                                    &mut state,
                                    limit,
                                    &segment_query_context,
                                )
                            })
                            .map(|points| BatchReply {
                                eof: points.is_empty(),
                                points,
                            })
                    },
                    ExactSourceMode::Exact,
                ))
            }
            LockedSegment::Proxy(proxy) => {
                let proxy = proxy.read();
                let query_vector: QueryVector = VectorInternal::from(query).into();
                materialized_exact_source(
                    &*proxy,
                    &vector_name,
                    &query_vector,
                    filter.as_ref(),
                    &query_context,
                )
            }
        }
    }
}

fn build_query_context(
    segments: &[LockedSegment],
    vector_name: &str,
    query: &SparseVector,
    stopped: Arc<AtomicBool>,
) -> OperationResult<QueryContext> {
    let requires_idf = segments.iter().any(|segment| {
        segment
            .get()
            .read()
            .config()
            .sparse_vector_data
            .get(vector_name)
            .is_some_and(|config| config.modifier == Some(Modifier::Idf))
    });
    let mut query_context =
        QueryContext::new(usize::MAX, HwMeasurementAcc::disposable()).with_is_stopped(stopped);
    if requires_idf {
        query_context.init_idf(vector_name, &query.indices);
    }
    for segment in segments {
        segment
            .get()
            .read()
            .fill_query_context(&mut query_context)?;
    }
    Ok(query_context)
}

#[cfg(test)]
#[path = "sparse_rank_plan/tests.rs"]
mod tests;
