// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Sparse stream merged across a frozen set of Segments.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::modifier::Modifier;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::entry::ReadSegmentEntry;
use segment::types::{Filter, ScoredPoint, SearchParams, VectorNameBuf, WithPayload, WithVector};
use sparse::common::sparse_vector::SparseVector;

use crate::exact_score_stream::{
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
                        let points = materialize_exact(
                            &*segment,
                            &vector_name,
                            &query,
                            filter.as_ref(),
                            &query_context,
                        )?;
                        return Ok(materialized_source(points, "Sparse"));
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
                    "Sparse",
                ))
            }
            LockedSegment::Proxy(proxy) => {
                let proxy = proxy.read();
                let points = materialize_exact(
                    &*proxy,
                    &vector_name,
                    &query,
                    filter.as_ref(),
                    &query_context,
                )?;
                Ok(materialized_source(points, "Sparse"))
            }
        }
    }
}

fn materialized_source(points: Vec<ScoredPoint>, channel: &'static str) -> SegmentScoreSource {
    let points = Arc::new(Mutex::new((points, 0usize)));
    SegmentScoreSource::on_demand(
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
        channel,
    )
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

fn materialize_exact(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &SparseVector,
    filter: Option<&Filter>,
    query_context: &QueryContext,
) -> OperationResult<Vec<ScoredPoint>> {
    let query_vector: QueryVector = VectorInternal::from(query.clone()).into();
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
    use std::collections::HashMap;
    use std::sync::atomic::Ordering as AtomicOrdering;

    use common::counter::hardware_counter::HardwareCounterCell;
    use ordered_float::OrderedFloat;
    use segment::data_types::named_vectors::NamedVectors;
    use segment::entry::SegmentEntry;
    use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
    use segment::json_path::JsonPath;
    use segment::payload_json;
    use segment::segment::Segment;
    use segment::segment_constructor::build_segment;
    use segment::types::{
        Condition, FieldCondition, PointIdType, SegmentConfig, SparseVectorDataConfig,
        SparseVectorStorageType, VectorStorageDatatype,
    };
    use tempfile::TempDir;

    use super::*;
    use crate::exact_score_stream::ExactShardStream;

    const VECTOR_NAME: &str = "sparse";

    fn make_segment(path: &TempDir, lane: u64) -> (Segment, Vec<(PointIdType, f32)>) {
        let config = SegmentConfig {
            vector_data: Default::default(),
            sparse_vector_data: HashMap::from([(
                VECTOR_NAME.to_owned(),
                SparseVectorDataConfig {
                    index: SparseIndexConfig {
                        full_scan_threshold: Some(1),
                        index_type: SparseIndexType::MutableRam,
                        datatype: Some(VectorStorageDatatype::Float32),
                    },
                    storage_type: SparseVectorStorageType::Mmap,
                    modifier: None,
                },
            )]),
            payload_storage_type: Default::default(),
        };
        let mut segment = build_segment(path.path(), &config, None, true).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let mut expected = Vec::new();
        for index in 0..256u64 {
            let id: PointIdType = (20_000 - (index * 2 + lane)).into();
            let score = 1.0 + (index % 5) as f32;
            let mut vectors = NamedVectors::default();
            vectors.insert(
                VECTOR_NAME.to_owned(),
                SparseVector {
                    indices: vec![11],
                    values: vec![score],
                }
                .into(),
            );
            segment
                .upsert_point(index, id, vectors, &hardware_counter)
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

    fn open_stream(
        segments: Vec<LockedSegment>,
        query: SparseVector,
        filter: Option<Filter>,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<ExactShardStream<SparseRankPlan>> {
        let plan = SparseRankPlan::new(
            &segments,
            VECTOR_NAME.to_owned(),
            query,
            filter,
            4_096,
            stopped.clone(),
        )?;
        ExactShardStream::open(segments, plan, batch_size, stopped)
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
        let mut stream = open_stream(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            Some(filter),
            17,
            stopped,
        )
        .unwrap();
        assert_eq!(stream.telemetry().batch_requests, 2);
        assert_eq!(stream.telemetry().points_received, 2);

        let mut actual = Vec::new();
        for _ in 0..9 {
            actual.push(stream.next_result().unwrap().unwrap());
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
        let mut stream = open_stream(
            vec![LockedSegment::new(segment)],
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            16,
            stopped.clone(),
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
    fn idf_config_uses_exact_qdrant_fallback() {
        let segment_dir = tempfile::tempdir().unwrap();
        let config = SegmentConfig {
            vector_data: Default::default(),
            sparse_vector_data: HashMap::from([(
                VECTOR_NAME.to_owned(),
                SparseVectorDataConfig {
                    index: SparseIndexConfig {
                        full_scan_threshold: Some(1),
                        index_type: SparseIndexType::MutableRam,
                        datatype: Some(VectorStorageDatatype::Float32),
                    },
                    storage_type: SparseVectorStorageType::Mmap,
                    modifier: Some(Modifier::Idf),
                },
            )]),
            payload_storage_type: Default::default(),
        };
        let mut segment = build_segment(segment_dir.path(), &config, None, true).unwrap();
        let mut vectors = NamedVectors::default();
        vectors.insert(
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![2.0],
            }
            .into(),
        );
        segment
            .upsert_point(0, 42u64.into(), vectors, &HardwareCounterCell::new())
            .unwrap();

        let mut stream = open_stream(
            vec![LockedSegment::new(segment)],
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            16,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();

        assert_eq!(stream.telemetry().exact_sources, 0);
        assert_eq!(stream.telemetry().exhaustive_fallback_sources, 1);
        assert_eq!(stream.next_result().unwrap().unwrap().id, 42u64.into());
        assert!(stream.next_result().unwrap().is_none());
    }

    #[test]
    fn idf_fallback_uses_one_shard_global_query_context() {
        fn build_idf_segment(
            path: &TempDir,
            base_id: u64,
            matching: usize,
            impact: f32,
        ) -> Segment {
            let config = SegmentConfig {
                vector_data: Default::default(),
                sparse_vector_data: HashMap::from([(
                    VECTOR_NAME.to_owned(),
                    SparseVectorDataConfig {
                        index: SparseIndexConfig {
                            full_scan_threshold: Some(1),
                            index_type: SparseIndexType::MutableRam,
                            datatype: Some(VectorStorageDatatype::Float32),
                        },
                        storage_type: SparseVectorStorageType::Mmap,
                        modifier: Some(Modifier::Idf),
                    },
                )]),
                payload_storage_type: Default::default(),
            };
            let mut segment = build_segment(path.path(), &config, None, true).unwrap();
            let hardware_counter = HardwareCounterCell::new();
            for index in 0..10_u64 {
                let mut vectors = NamedVectors::default();
                let matches = index < matching as u64;
                vectors.insert(
                    VECTOR_NAME.to_owned(),
                    SparseVector {
                        indices: vec![if matches { 11 } else { 99 }],
                        values: vec![if matches { impact } else { 1.0 }],
                    }
                    .into(),
                );
                segment
                    .upsert_point(index, (base_id + index).into(), vectors, &hardware_counter)
                    .unwrap();
            }
            segment
        }

        let rare_dir = tempfile::tempdir().unwrap();
        let common_dir = tempfile::tempdir().unwrap();
        let rare = build_idf_segment(&rare_dir, 100, 1, 1.0);
        let common = build_idf_segment(&common_dir, 200, 10, 2.0);
        let mut stream = open_stream(
            vec![LockedSegment::new(rare), LockedSegment::new(common)],
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            8,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let ranking = std::iter::from_fn(|| stream.next_result().transpose())
            .collect::<OperationResult<Vec<_>>>()
            .unwrap();

        assert_eq!(ranking.len(), 11);
        assert_eq!(ranking[0].id, 200_u64.into());
        assert_eq!(ranking.last().unwrap().id, 100_u64.into());
        assert_eq!(stream.telemetry().exhaustive_fallback_sources, 2);
    }
}
