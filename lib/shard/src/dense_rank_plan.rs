// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Dense stream merged across a frozen set of Segments.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use parking_lot::Mutex;
#[cfg(test)]
use segment::common::operation_error::OperationError;
use segment::common::operation_error::OperationResult;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::entry::ReadSegmentEntry;
use segment::index::dense_rank_state::DenseExecutionPolicy;
use segment::types::{Filter, VectorNameBuf};

use crate::exact_rank_fallback::materialized_exact_source;
use crate::exact_shard_stream::{
    BatchReply, ExactShardStreamTelemetry, ExactSourceMode, SegmentRankPlan, SegmentScoreSource,
};
use crate::locked_segment::LockedSegment;

pub type DenseShardTelemetry = ExactShardStreamTelemetry;

#[derive(Clone)]
pub struct DenseRankPlan {
    vector_name: VectorNameBuf,
    query: Vec<f32>,
    filter: Option<Filter>,
    policy: DenseExecutionPolicy,
}

impl DenseRankPlan {
    pub fn new(
        vector_name: VectorNameBuf,
        query: Vec<f32>,
        filter: Option<Filter>,
        policy: DenseExecutionPolicy,
    ) -> Self {
        Self {
            vector_name,
            query,
            filter,
            policy,
        }
    }
}

impl SegmentRankPlan for DenseRankPlan {
    const CHANNEL: &'static str = "Dense";

    fn open_segment(
        &self,
        _source: usize,
        segment: LockedSegment,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<SegmentScoreSource> {
        let vector_name = self.vector_name.clone();
        let query = self.query.clone();
        let filter = self.filter.clone();
        let policy = self.policy;
        match segment {
            LockedSegment::Original(segment) => {
                let mut query_context =
                    QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                        .with_is_stopped(stopped.clone());
                let state = {
                    let segment = segment.read();
                    segment.fill_query_context(&mut query_context)?;
                    let segment_query_context = query_context.get_segment_query_context();
                    segment.with_view(|view| {
                        view.open_exact_dense_segment_state(
                            &vector_name,
                            &query,
                            filter.as_ref(),
                            policy,
                            &segment_query_context,
                        )
                    })?
                };
                let state = Arc::new(Mutex::new(state));
                let query_context = Arc::new(query_context);
                Ok(SegmentScoreSource::on_demand(
                    move |limit| {
                        let segment = segment.read();
                        let mut state = state.lock();
                        let segment_query_context = query_context.get_segment_query_context();
                        segment
                            .with_view(|view| {
                                view.advance_exact_dense_segment_state(
                                    &vector_name,
                                    &query,
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
                let query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                    .with_is_stopped(stopped);
                let query_vector: QueryVector = VectorInternal::Dense(query).into();
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

#[cfg(test)]
#[path = "dense_rank_plan/tests.rs"]
mod tests;
