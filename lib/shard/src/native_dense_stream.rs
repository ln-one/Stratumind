// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact pull-based Dense stream merged across a frozen set of Segments.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use common::counter::hardware_accumulator::HwMeasurementAcc;
use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorInternal};
use segment::entry::ReadSegmentEntry;
use segment::index::native_dense_stream::NativeDensePolicy;
use segment::types::{
    Filter, PointIdType, ScoredPoint, SearchParams, VectorNameBuf, WithPayload, WithVector,
};

use crate::locked_segment::LockedSegment;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerMode {
    Native,
    ExhaustiveFallback,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeDenseShardTelemetry {
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
                    "native Dense Segment worker stopped before a pull request",
                )
            })?;
        let batch = reply_rx.recv().map_err(|_| {
            OperationError::service_error_light(
                "native Dense Segment worker stopped before returning a pull result",
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
            .then_with(|| other.point.id.cmp(&self.point.id))
            .then_with(|| self.point.version.cmp(&other.point.version))
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl PartialOrd for PendingPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// One exact Dense rank stream for all Segments in a Shard snapshot.
///
/// Original Segments use the native certified cursor. Proxy Segments are
/// materialized exactly once as a safe fallback. Both plans are exhaustive
/// order equivalent and no repeated Top-N query is issued.
pub struct NativeDenseShardStream {
    sources: Vec<SegmentSource>,
    pending: BinaryHeap<PendingPoint>,
    seen: HashSet<PointIdType>,
    batch_size: usize,
    stopped: Arc<AtomicBool>,
    telemetry: NativeDenseShardTelemetry,
}

impl NativeDenseShardStream {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        segments: Vec<LockedSegment>,
        vector_name: VectorNameBuf,
        query: Vec<f32>,
        filter: Option<Filter>,
        policy: NativeDensePolicy,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
    ) -> OperationResult<Self> {
        if batch_size == 0 {
            return Err(OperationError::validation_error(
                "native Dense Shard batch size must be positive",
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
                policy,
                batch_size,
                stopped.clone(),
            )?);
        }

        let mut telemetry = NativeDenseShardTelemetry {
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
            batch_size,
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
                    "native Dense Shard stream was cancelled",
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

    pub fn telemetry(&self) -> NativeDenseShardTelemetry {
        self.telemetry
    }

    fn pull_source(&mut self, source: usize) -> OperationResult<()> {
        if let Some(point) = self.sources[source].pop(self.batch_size)? {
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
    query: Vec<f32>,
    filter: Option<Filter>,
    policy: NativeDensePolicy,
    initial_limit: usize,
    stopped: Arc<AtomicBool>,
) -> OperationResult<SegmentSource> {
    let (command_tx, command_rx) = sync_channel(1);
    let (ready_tx, ready_rx) = sync_channel(1);
    let worker = thread::Builder::new()
        .name(format!("native-dense-segment-{source}"))
        .spawn(move || {
            run_segment_worker(
                segment,
                vector_name,
                query,
                filter,
                policy,
                initial_limit,
                stopped,
                command_rx,
                ready_tx,
            );
        })
        .map_err(|error| {
            OperationError::service_error_light(format!(
                "failed to start native Dense Segment worker: {error}"
            ))
        })?;

    let mode = match ready_rx.recv() {
        Ok(result) => result?,
        Err(_) => {
            let _ = worker.join();
            return Err(OperationError::service_error_light(
                "native Dense Segment worker stopped during initialization",
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
    query: Vec<f32>,
    filter: Option<Filter>,
    policy: NativeDensePolicy,
    initial_limit: usize,
    stopped: Arc<AtomicBool>,
    commands: Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<WorkerMode>>,
) {
    match segment {
        LockedSegment::Original(segment) => {
            let segment = segment.read();
            let mut query_context = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable())
                .with_is_stopped(stopped);
            if let Err(error) = segment.fill_query_context(&mut query_context) {
                let _ = ready.send(Err(error));
                return;
            }
            let segment_query_context = query_context.get_segment_query_context();
            let mut ready = Some(ready);
            let native = segment.with_view(|view| {
                view.with_native_dense_stream(
                    &vector_name,
                    &query,
                    filter.as_ref(),
                    initial_limit,
                    policy,
                    &segment_query_context,
                    |next| {
                        ready
                            .take()
                            .expect("native worker sends readiness once")
                            .send(Ok(WorkerMode::Native))
                            .map_err(|_| {
                                OperationError::cancelled(
                                    "native Dense Shard stream closed during initialization",
                                )
                            })?;
                        serve_native(next, &commands)
                    },
                )
            });
            if let Err(error) = native
                && let Some(ready) = ready
            {
                let _ = ready.send(Err(error));
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
    query: &[f32],
    filter: Option<&Filter>,
    query_context: &QueryContext,
    commands: &Receiver<WorkerCommand>,
    ready: SyncSender<OperationResult<WorkerMode>>,
) {
    match materialize_exact(segment, vector_name, query, filter, query_context) {
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
    use std::sync::atomic::Ordering as AtomicOrdering;

    use common::counter::hardware_counter::HardwareCounterCell;
    use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
    use segment::entry::SegmentEntry;
    use segment::json_path::JsonPath;
    use segment::payload_json;
    use segment::segment::Segment;
    use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
    use segment::types::{Condition, Distance, FieldCondition};
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
        let mut stream = NativeDenseShardStream::open(
            vec![LockedSegment::new(first), LockedSegment::new(second)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            Some(filter),
            NativeDensePolicy::default(),
            17,
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
        let mut stream = NativeDenseShardStream::open(
            vec![LockedSegment::new(segment)],
            DEFAULT_VECTOR_NAME.to_owned(),
            vec![1.0, 0.0],
            None,
            NativeDensePolicy::default(),
            8,
            stopped.clone(),
        )
        .unwrap();
        stopped.store(true, AtomicOrdering::Relaxed);
        assert!(matches!(
            stream.next_result(),
            Err(OperationError::Cancelled { .. })
        ));
    }
}
