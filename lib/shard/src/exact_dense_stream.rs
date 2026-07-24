// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Dense stream merged across a frozen set of Segments.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::entry::ReadSegmentEntry;
use segment::index::exact_dense_stream::DenseExecutionPolicy;
use segment::types::{Filter, ScoredPoint, SearchParams, VectorNameBuf, WithPayload, WithVector};

use crate::exact_score_stream::{
    BatchReply, ExactBatchExecutor, ExactShardPointVersions, ExactShardScoreStream,
    ExactShardStreamTelemetry, ExactSourceMode, SegmentScoreSource,
};
use crate::locked_segment::LockedSegment;

pub type DenseShardTelemetry = ExactShardStreamTelemetry;

/// One exact Dense rank stream for all Segments in a Shard snapshot.
///
/// Original Segments use the native certified cursor. Proxy Segments are
/// materialized exactly once as a safe fallback. Both plans are exhaustive
/// order equivalent and no repeated Top-N query is issued.
pub struct ExactDenseShardStream {
    inner: ExactShardScoreStream,
}

impl ExactDenseShardStream {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: Vec<f32>,
        filter: Option<Filter>,
        policy: DenseExecutionPolicy,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
        batch_executor: ExactBatchExecutor,
    ) -> OperationResult<Self> {
        let point_versions = ExactShardPointVersions::build(&segments, &stopped)?;
        Self::open_with_point_versions(
            segments,
            vector_name,
            query,
            filter,
            policy,
            batch_size,
            stopped,
            batch_executor,
            point_versions,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_with_point_versions(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: Vec<f32>,
        filter: Option<Filter>,
        policy: DenseExecutionPolicy,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
        batch_executor: ExactBatchExecutor,
        point_versions: Arc<ExactShardPointVersions>,
    ) -> OperationResult<Self> {
        if batch_size == 0 {
            return Err(OperationError::validation_error(
                "native Dense Shard batch size must be positive",
            ));
        }

        let mut sources = Vec::with_capacity(segments.len());
        for (source, segment) in segments.into_iter().enumerate() {
            sources.push(open_segment_rank_source(
                source,
                segment,
                vector_name.clone(),
                query.clone(),
                filter.clone(),
                policy,
                stopped.clone(),
                &batch_executor,
            )?);
        }

        Ok(Self {
            inner: ExactShardScoreStream::open(
                sources,
                point_versions,
                batch_size,
                stopped,
                "Dense",
            )?,
        })
    }

    pub fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        self.inner.next_result()
    }

    pub fn telemetry(&self) -> DenseShardTelemetry {
        self.inner.telemetry()
    }
}

#[allow(clippy::too_many_arguments)]
fn open_segment_rank_source(
    source: usize,
    segment: LockedSegment,
    vector_name: VectorNameBuf,
    query: Vec<f32>,
    filter: Option<Filter>,
    policy: DenseExecutionPolicy,
    stopped: Arc<AtomicBool>,
    batch_executor: &ExactBatchExecutor,
) -> OperationResult<SegmentScoreSource> {
    match segment {
        LockedSegment::Original(segment) => {
            let mut query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
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
            let spawner = batch_executor.clone();
            Ok(SegmentScoreSource::on_demand(
                move |limit| {
                    let segment = segment.clone();
                    let state = state.clone();
                    let vector_name = vector_name.clone();
                    let query = query.clone();
                    let query_context = query_context.clone();
                    spawner.run_batch(format!("exact-dense-segment-{source}-pull"), move || {
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
                    })
                },
                ExactSourceMode::Exact,
                "Dense",
            ))
        }
        LockedSegment::Proxy(proxy) => {
            let proxy = proxy.read();
            let query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                .with_is_stopped(stopped);
            let points = materialize_exact(
                &*proxy,
                &vector_name,
                &query,
                filter.as_ref(),
                &query_context,
            )?;
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
                "Dense",
            ))
        }
    }
}

fn materialize_exact(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &[f32],
    filter: Option<&Filter>,
    query_context: &QueryContext,
) -> OperationResult<Vec<ScoredPoint>> {
    let query_vector: QueryVector = VectorInternal::Dense(query.to_vec()).into();
    let query_vectors = [&query_vector];
    let params = SearchParams {
        exact: true,
        ..Default::default()
    };
    let mut result = segment
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
    result.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.id.cmp(&right.id))
            .then_with(|| right.version.cmp(&left.version))
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering as AtomicOrdering;

    use common::counter::hardware_counter::HardwareCounterCell;
    use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
    use segment::entry::SegmentEntry;
    use segment::json_path::JsonPath;
    use segment::payload_json;
    use segment::segment::Segment;
    use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
    use segment::types::{Condition, Distance, FieldCondition, PointIdType};
    use tempfile::TempDir;

    use super::*;

    fn make_segment(path: &TempDir, lane: u64) -> (Segment, Vec<(PointIdType, f32)>) {
        let mut segment = build_simple_segment(path.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let mut expected = Vec::new();
        for index in 0..256u64 {
            let id: PointIdType = (20_000 - (index * 2 + lane)).into();
            let score = 1.0 + (index % 5) as f32;
            segment
                .upsert_point(
                    index,
                    id,
                    only_default_vector(&[score, index as f32]),
                    &hardware_counter,
                )
                .unwrap();
            let visible = index % 3 != 1;
            segment
                .set_full_payload(
                    index,
                    id,
                    &payload_json! {"visible": visible},
                    &hardware_counter,
                )
                .unwrap();
            if visible {
                expected.push((id, score));
            }
        }
        (segment, expected)
    }

    #[test]
    fn shard_stream_merges_exact_segments_exactly_and_resumes() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let (first, mut expected) = make_segment(&first_dir, 0);
        let (second, second_expected) = make_segment(&second_dir, 1);
        expected.extend(second_expected);
        expected.sort_unstable_by(|left, right| {
            OrderedFloat(right.1)
                .cmp(&OrderedFloat(left.1))
                .then_with(|| left.0.cmp(&right.0))
        });

        let filter = Filter::new_must(Condition::Field(FieldCondition::new_match(
            JsonPath::new("visible"),
            true.into(),
        )));
        let stopped = Arc::new(AtomicBool::new(false));
        let first = LockedSegment::new(first);
        let LockedSegment::Original(first_handle) = first.clone() else {
            unreachable!()
        };
        let mut stream = ExactDenseShardStream::open(
            vec![first, LockedSegment::new(second)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            Some(filter),
            DenseExecutionPolicy::default(),
            17,
            stopped,
            ExactBatchExecutor::dedicated_threads_for_tests(),
        )
        .unwrap();
        assert!(
            first_handle.try_write().is_some(),
            "session initialization must release the Segment read guard"
        );
        assert_eq!(stream.telemetry().batch_requests, 2);
        assert_eq!(stream.telemetry().points_received, 2);

        let mut actual = Vec::new();
        for _ in 0..9 {
            actual.push(stream.next_result().unwrap().unwrap());
            assert!(
                first_handle.try_write().is_some(),
                "each completed pull must release the Segment read guard"
            );
        }
        while let Some(point) = stream.next_result().unwrap() {
            actual.push(point);
        }

        assert_eq!(
            actual
                .iter()
                .map(|point| (point.id, point.score))
                .collect::<Vec<_>>(),
            expected
        );
        let telemetry = stream.telemetry();
        assert_eq!(telemetry.exact_sources, 2);
        assert_eq!(telemetry.exhaustive_fallback_sources, 0);
        assert_eq!(telemetry.points_emitted, expected.len());
    }

    #[test]
    fn shard_stream_reports_cancellation_instead_of_eof() {
        let segment_dir = tempfile::tempdir().unwrap();
        let (segment, _) = make_segment(&segment_dir, 0);
        let stopped = Arc::new(AtomicBool::new(false));
        let mut stream = ExactDenseShardStream::open(
            vec![LockedSegment::new(segment)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            None,
            DenseExecutionPolicy::default(),
            8,
            stopped.clone(),
            ExactBatchExecutor::dedicated_threads_for_tests(),
        )
        .unwrap();
        stopped.store(true, AtomicOrdering::Relaxed);
        assert!(matches!(
            stream.next_result(),
            Err(OperationError::Cancelled { .. })
        ));
        stopped.store(false, AtomicOrdering::Relaxed);
        assert!(matches!(
            stream.next_result(),
            Err(OperationError::Cancelled { .. })
        ));
    }

    #[test]
    fn stale_higher_scoring_segment_copy_is_removed_before_ranking() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let mut first = build_simple_segment(first_dir.path(), 2, Distance::Dot).unwrap();
        let mut second = build_simple_segment(second_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let shared: PointIdType = 7_u64.into();
        first
            .upsert_point(
                10,
                shared,
                only_default_vector(&[100.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();
        second
            .upsert_point(
                11,
                shared,
                only_default_vector(&[1.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();
        second
            .upsert_point(
                5,
                8_u64.into(),
                only_default_vector(&[2.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();

        let mut stream = ExactDenseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            None,
            DenseExecutionPolicy::default(),
            8,
            Arc::new(AtomicBool::new(false)),
            ExactBatchExecutor::dedicated_threads_for_tests(),
        )
        .unwrap();
        let ranking = std::iter::from_fn(|| stream.next_result().transpose())
            .collect::<OperationResult<Vec<_>>>()
            .unwrap();
        assert_eq!(
            ranking
                .iter()
                .map(|point| (point.id, point.version, point.score))
                .collect::<Vec<_>>(),
            vec![(8_u64.into(), 5, 2.0), (shared, 11, 1.0)]
        );
        assert_eq!(stream.telemetry().duplicates_suppressed, 1);
    }

    #[test]
    fn equal_version_copies_in_multiple_segments_fail_closed() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let mut first = build_simple_segment(first_dir.path(), 2, Distance::Dot).unwrap();
        let mut second = build_simple_segment(second_dir.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        for segment in [&mut first, &mut second] {
            segment
                .upsert_point(
                    10,
                    7_u64.into(),
                    only_default_vector(&[1.0, 0.0]),
                    &hardware_counter,
                )
                .unwrap();
        }
        let error = ExactDenseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            None,
            DenseExecutionPolicy::default(),
            8,
            Arc::new(AtomicBool::new(false)),
            ExactBatchExecutor::dedicated_threads_for_tests(),
        )
        .err()
        .expect("duplicate authoritative versions must fail");
        assert!(error.to_string().contains("in multiple Segments"));
    }
}
