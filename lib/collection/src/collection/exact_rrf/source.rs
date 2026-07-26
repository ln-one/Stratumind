// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use segment::common::operation_error::{OperationError, OperationResult};
use segment::common::reciprocal_rank_fusion::ExactRrfBatchStream;
use segment::index::exact_rank_stream::{
    ExactScoreBatch, ExactScoreBatchStream, ExactScoredIdentity, KWayExactScoreStream,
};
use segment::types::{PointIdType, ScoredPoint};

pub(super) fn exact_score_batch_stream(
    mut next: impl FnMut(usize) -> OperationResult<Vec<ScoredPoint>> + 'static,
    versions: Rc<RefCell<HashMap<PointIdType, u64>>>,
) -> ExactScoreBatchStream<'static> {
    Box::new(move |max_results| {
        let points = next(max_results)?;
        let exhausted = points.len() < max_results;
        let observed_versions = versions.borrow();
        let mut batch_versions = HashMap::with_capacity(points.len());
        for point in &points {
            if let Some(observed) = observed_versions.get(&point.id)
                && *observed != point.version
            {
                return Err(OperationError::inconsistent_storage(format!(
                    "exact RRF observed point {} at conflicting versions {} and {}",
                    point.id, observed, point.version,
                )));
            }
            if let Some(observed) = batch_versions.insert(point.id, point.version)
                && observed != point.version
            {
                return Err(OperationError::inconsistent_storage(format!(
                    "exact RRF batch observed point {} at conflicting versions {} and {}",
                    point.id, observed, point.version,
                )));
            }
        }
        drop(observed_versions);
        versions.borrow_mut().extend(batch_versions);

        Ok(ExactScoreBatch {
            points: points
                .into_iter()
                .map(|point| ExactScoredIdentity {
                    id: point.id,
                    score: point.score,
                })
                .collect(),
            exhausted,
        })
    })
}

pub(super) fn observed_rank_batch_stream(
    mut merged: KWayExactScoreStream<'static>,
    materialized: Arc<AtomicUsize>,
) -> ExactRrfBatchStream<'static> {
    Box::new(move |max_results| {
        let batch = merged.next_rank_batch(max_results)?;
        materialized.fetch_add(batch.point_ids.len(), AtomicOrdering::Relaxed);
        Ok(batch)
    })
}
