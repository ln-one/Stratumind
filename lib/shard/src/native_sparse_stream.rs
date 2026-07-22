// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Sparse stream merged across a frozen set of Segments.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use common::counter::hardware_accumulator::HwMeasurementAcc;
use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::modifier::Modifier;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::entry::ReadSegmentEntry;
use segment::types::{
    Filter, PointIdType, ScoredPoint, SearchParams, VectorNameBuf, WithPayload, WithVector,
};
use sparse::common::sparse_vector::SparseVector;

use crate::locked_segment::LockedSegment;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerMode {
    Native,
    ExhaustiveFallback,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeSparseShardTelemetry {
    pub sources: usize,
    pub native_sources: usize,
    pub exhaustive_fallback_sources: usize,
    pub points_pulled: usize,
    pub points_emitted: usize,
    pub duplicates_suppressed: usize,
}

struct BatchReply {
    points: Vec<ScoredPoint>,
    eof: bool,
}

enum WorkerCommand {
    Pull {
        limit: usize,
        reply: SyncSender<OperationResult<BatchReply>>,
    },
    Stop,
}

struct SegmentSource {
    commands: SyncSender<WorkerCommand>,
    worker: Option<JoinHandle<()>>,
    buffer: VecDeque<ScoredPoint>,
    eof: bool,
    mode: WorkerMode,
}

impl SegmentSource {
    fn refill(&mut self, batch_size: usize) -> OperationResult<()> {
        if !self.buffer.is_empty() || self.eof {
            return Ok(());
        }
        let (reply_tx, reply_rx) = sync_channel(1);
        self.commands
            .send(WorkerCommand::Pull {
                limit: batch_size,
                reply: reply_tx,
            })
            .map_err(|_| {
                OperationError::service_error_light(
                    "native Sparse Segment worker stopped before a pull request",
                )
            })?;
        let batch = reply_rx.recv().map_err(|_| {
            OperationError::service_error_light(
                "native Sparse Segment worker stopped before returning a pull result",
            )
        })??;
        self.buffer = VecDeque::from(batch.points);
        self.eof = batch.eof;
        Ok(())
    }

    fn pop(&mut self, batch_size: usize) -> OperationResult<Option<ScoredPoint>> {
        self.refill(batch_size)?;
        Ok(self.buffer.pop_front())
    }
}

impl Drop for SegmentSource {
    fn drop(&mut self) {
        let _ = self.commands.send(WorkerCommand::Stop);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Debug)]
struct PendingPoint {
    source: usize,
    point: ScoredPoint,
}

impl Eq for PendingPoint {}

impl PartialEq for PendingPoint {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.point.id == other.point.id
            && self.point.version == other.point.version
            && self.point.score == other.point.score
    }
}

impl Ord for PendingPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.point.score)
            .cmp(&OrderedFloat(other.point.score))
            // Smaller frozen identity wins an equal-score tie.
            .then_with(|| other.point.id.cmp(&self.point.id))
            // Prefer the newest copy when one identity is duplicated at the
            // same score during Segment maintenance.
            .then_with(|| self.point.version.cmp(&other.point.version))
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl PartialOrd for PendingPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A single exact Sparse rank stream for all Segments in one Shard snapshot.
///
/// Original Segments keep their native posting cursor paused in a dedicated
/// worker. Proxy Segments, and index representations without native support,
/// are materialized exactly once as an explicit safe fallback. No repeated
/// Top-N query is issued.
pub struct NativeSparseShardStream {
    sources: Vec<SegmentSource>,
    pending: BinaryHeap<PendingPoint>,
    seen: HashSet<PointIdType>,
    source_batch_size: usize,
    stopped: Arc<AtomicBool>,
    telemetry: NativeSparseShardTelemetry,
}

impl NativeSparseShardStream {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: SparseVector,
        filter: Option<Filter>,
        source_batch_size: usize,
        posting_batch_size: usize,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<Self> {
        if source_batch_size == 0 || posting_batch_size == 0 {
            return Err(OperationError::validation_error(
                "native Sparse Shard batch sizes must be positive",
            ));
        }

        let mut sources = Vec::with_capacity(segments.len());
        for (source, segment) in segments.into_iter().enumerate() {
            sources.push(spawn_segment_worker(
                source,
                segment,
                vector_name.clone(),
                query.clone(),
                filter.clone(),
                source_batch_size,
                posting_batch_size,
                stopped.clone(),
            )?);
        }

        let mut telemetry = NativeSparseShardTelemetry {
            sources: sources.len(),
            ..Default::default()
        };
        for source in &sources {
            match source.mode {
                WorkerMode::Native => telemetry.native_sources += 1,
                WorkerMode::ExhaustiveFallback => telemetry.exhaustive_fallback_sources += 1,
            }
        }

        let mut stream = Self {
            sources,
            pending: BinaryHeap::new(),
            seen: HashSet::new(),
            source_batch_size,
            stopped,
            telemetry,
        };
        for source in 0..stream.sources.len() {
            stream.pull_source(source)?;
        }
        Ok(stream)
    }

    pub fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        loop {
            if self.stopped.load(AtomicOrdering::Relaxed) {
                return Err(OperationError::cancelled(
                    "native Sparse Shard stream was cancelled",
                ));
            }
            let Some(pending) = self.pending.pop() else {
                return Ok(None);
            };
            self.pull_source(pending.source)?;
            if !self.seen.insert(pending.point.id) {
                self.telemetry.duplicates_suppressed += 1;
                continue;
            }
            self.telemetry.points_emitted += 1;
            return Ok(Some(pending.point));
        }
    }

    pub fn telemetry(&self) -> NativeSparseShardTelemetry {
        self.telemetry
    }

    fn pull_source(&mut self, source: usize) -> OperationResult<()> {
        if let Some(point) = self.sources[source].pop(self.source_batch_size)? {
            self.telemetry.points_pulled += 1;
            self.pending.push(PendingPoint { source, point });
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_segment_worker(
    source: usize,
    segment: LockedSegment,
    vector_name: VectorNameBuf,
    query: SparseVector,
    filter: Option<Filter>,
    source_batch_size: usize,
    posting_batch_size: usize,
    stopped: Arc<AtomicBool>,
) -> OperationResult<SegmentSource> {
    let (command_tx, command_rx) = sync_channel(1);
    let (ready_tx, ready_rx) = sync_channel(1);
    let worker = thread::Builder::new()
        .name(format!("native-sparse-segment-{source}"))
        .spawn(move || {
            run_segment_worker(
                segment,
                vector_name,
                query,
                filter,
                source_batch_size,
                posting_batch_size,
                stopped,
                command_rx,
                ready_tx,
            );
        })
        .map_err(|error| {
            OperationError::service_error_light(format!(
                "failed to start native Sparse Segment worker: {error}"
            ))
        })?;

    let mode = match ready_rx.recv() {
        Ok(result) => result?,
        Err(_) => {
            let _ = worker.join();
            return Err(OperationError::service_error_light(
                "native Sparse Segment worker stopped during initialization",
            ));
        }
    };
    Ok(SegmentSource {
        commands: command_tx,
        worker: Some(worker),
        buffer: VecDeque::new(),
        eof: false,
        mode,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_segment_worker(
    segment: LockedSegment,
    vector_name: VectorNameBuf,
    query: SparseVector,
    filter: Option<Filter>,
    initial_limit: usize,
    posting_batch_size: usize,
    stopped: Arc<AtomicBool>,
    commands: Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<WorkerMode>>,
) {
    match segment {
        LockedSegment::Original(segment) => {
            let segment = segment.read();
            let mut query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                .with_is_stopped(stopped);
            let requires_idf = segment
                .config()
                .sparse_vector_data
                .get(&vector_name)
                .is_some_and(|config| config.modifier == Some(Modifier::Idf));
            if requires_idf {
                query_context.init_idf(&vector_name, &query.indices);
            }
            if let Err(error) = segment.fill_query_context(&mut query_context) {
                let _ = ready.send(Err(error));
                return;
            }
            let segment_query_context = query_context.get_segment_query_context();
            let mut ready = Some(ready);
            let native = segment.with_view(|view| {
                view.with_native_sparse_stream(
                    &vector_name,
                    &query,
                    filter.as_ref(),
                    initial_limit,
                    posting_batch_size,
                    &segment_query_context,
                    |next| {
                        ready
                            .take()
                            .expect("native worker sends readiness once")
                            .send(Ok(WorkerMode::Native))
                            .map_err(|_| {
                                OperationError::cancelled(
                                    "native Sparse Shard stream closed during initialization",
                                )
                            })?;
                        serve_native(next, &commands)
                    },
                )
            });

            if let Err(error) = native
                && let Some(ready) = ready
            {
                if matches!(error, OperationError::WrongSparse) {
                    run_materialized_worker(
                        &*segment,
                        &vector_name,
                        &query,
                        filter.as_ref(),
                        &query_context,
                        &commands,
                        ready,
                    );
                } else {
                    let _ = ready.send(Err(error));
                }
            }
        }
        LockedSegment::Proxy(proxy) => {
            let proxy = proxy.read();
            let query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                .with_is_stopped(stopped);
            run_materialized_worker(
                &*proxy,
                &vector_name,
                &query,
                filter.as_ref(),
                &query_context,
                &commands,
                ready,
            );
        }
    }
}

fn serve_native(
    next: &mut dyn FnMut() -> OperationResult<Option<ScoredPoint>>,
    commands: &Receiver<WorkerCommand>,
) -> OperationResult<()> {
    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::Pull { limit, reply } => {
                let mut points = Vec::with_capacity(limit);
                let mut eof = false;
                for _ in 0..limit {
                    match next() {
                        Ok(Some(point)) => points.push(point),
                        Ok(None) => {
                            eof = true;
                            break;
                        }
                        Err(error) => {
                            let _ = reply.send(Err(error));
                            return Ok(());
                        }
                    }
                }
                if reply.send(Ok(BatchReply { points, eof })).is_err() {
                    return Ok(());
                }
            }
            WorkerCommand::Stop => return Ok(()),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_materialized_worker(
    segment: &dyn ReadSegmentEntry,
    vector_name: &str,
    query: &SparseVector,
    filter: Option<&Filter>,
    query_context: &QueryContext,
    commands: &Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<WorkerMode>>,
) {
    let result = materialize_exact(segment, vector_name, query, filter, query_context);
    match result {
        Ok(points) => {
            if ready.send(Ok(WorkerMode::ExhaustiveFallback)).is_ok() {
                serve_materialized(points, commands);
            }
        }
        Err(error) => {
            let _ = ready.send(Err(error));
        }
    }
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

fn serve_materialized(points: Vec<ScoredPoint>, commands: &Receiver<WorkerCommand>) {
    let mut points = VecDeque::from(points);
    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::Pull { limit, reply } => {
                let batch: Vec<_> = (0..limit).filter_map(|_| points.pop_front()).collect();
                let eof = points.is_empty();
                if reply.send(Ok(BatchReply { points: batch, eof })).is_err() {
                    return;
                }
            }
            WorkerCommand::Stop => return,
        }
    }
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
        Condition, FieldCondition, SegmentConfig, SparseVectorDataConfig, SparseVectorStorageType,
        VectorStorageDatatype,
    };
    use tempfile::TempDir;

    use super::*;

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

    #[test]
    fn shard_stream_merges_native_segments_exactly_and_resumes() {
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
        let mut stream = NativeSparseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            Some(filter),
            17,
            4_096,
            stopped,
        )
        .unwrap();

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
        assert_eq!(telemetry.native_sources, 2);
        assert_eq!(telemetry.exhaustive_fallback_sources, 0);
        assert_eq!(telemetry.points_emitted, expected.len());
    }

    #[test]
    fn shard_stream_reports_cancellation_instead_of_eof() {
        let segment_dir = tempfile::tempdir().unwrap();
        let (segment, _) = make_segment(&segment_dir, 0);
        let stopped = Arc::new(AtomicBool::new(false));
        let mut stream = NativeSparseShardStream::open(
            vec![LockedSegment::new(segment)],
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            16,
            4_096,
            stopped.clone(),
        )
        .unwrap();
        stopped.store(true, AtomicOrdering::Relaxed);
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

        let mut stream = NativeSparseShardStream::open(
            vec![LockedSegment::new(segment)],
            VECTOR_NAME.to_owned(),
            SparseVector {
                indices: vec![11],
                values: vec![1.0],
            },
            None,
            16,
            4_096,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();

        assert_eq!(stream.telemetry().native_sources, 0);
        assert_eq!(stream.telemetry().exhaustive_fallback_sources, 1);
        assert_eq!(stream.next_result().unwrap().unwrap().id, 42u64.into());
        assert!(stream.next_result().unwrap().is_none());
    }
}
