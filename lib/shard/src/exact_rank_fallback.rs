// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Shared exhaustive fallback for Segment representations without an exact rank state.

use std::sync::Arc;

use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use segment::common::operation_error::OperationResult;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::QueryVector;
use segment::entry::ReadSegmentEntry;
use segment::types::{Filter, SearchParams, WithPayload, WithVector};

use crate::exact_shard_stream::{BatchReply, ExactSourceMode, SegmentScoreSource};

pub(crate) fn materialized_exact_source(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &QueryVector,
    filter: Option<&Filter>,
    query_context: &QueryContext,
) -> OperationResult<SegmentScoreSource> {
    let query_vectors = [query];
    let params = SearchParams {
        exact: true,
        ..Default::default()
    };
    let mut points = segment
        .search_batch(
            vector_name,
            &query_vectors,
            &WithPayload::default(),
            &WithVector::default(),
            filter,
            segment.available_point_count_without_deferred(),
            Some(&params),
            &query_context.get_segment_query_context(),
        )?
        .pop()
        .unwrap_or_default();
    points.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.id.cmp(&right.id))
            .then_with(|| right.version.cmp(&left.version))
    });

    let points = Arc::new(Mutex::new((points, 0usize)));
    Ok(SegmentScoreSource::on_demand(
        move |limit| {
            let mut state = points.lock();
            let start = state.1;
            let end = start.saturating_add(limit).min(state.0.len());
            let batch = state.0[start..end].to_vec();
            state.1 = end;
            Ok(BatchReply {
                points: batch,
                eof: end == state.0.len(),
            })
        },
        ExactSourceMode::ExhaustiveFallback,
    ))
}
