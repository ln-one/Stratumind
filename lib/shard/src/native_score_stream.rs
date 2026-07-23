// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Shared pull/merge machinery for exact Segment score streams in one Shard.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::DeferredBehavior;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::types::{PointIdType, ScoredPoint, SeqNumberType};

use crate::locked_segment::LockedSegment;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AuthoritativePointVersion {
    version: SeqNumberType,
    source: usize,
}

/// Authoritative visible point owner for one frozen Shard generation.
///
/// Qdrant may temporarily retain older point copies across Segments. Exact
/// channel streams must remove those copies before score ordering; seeing a
/// stale high score first and deduplicating later is not correct.
pub struct NativeShardPointVersions {
    points: HashMap<PointIdType, AuthoritativePointVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeSegmentGeneration {
    path: PathBuf,
    version: SeqNumberType,
    is_proxy: bool,
}

struct CachedPointVersions {
    generation: Vec<NativeSegmentGeneration>,
    point_versions: Arc<NativeShardPointVersions>,
}

/// Reuses the authoritative owner map while a Shard's ordered Segment
/// generation is unchanged. Segment update versions cover in-place writes;
/// paths and proxy kinds cover optimizer replacements without pinning retired
/// Segment handles in memory.
#[derive(Default)]
pub struct NativeShardPointVersionsCache {
    cached: Mutex<Option<CachedPointVersions>>,
}

impl NativeShardPointVersionsCache {
    pub fn get_or_build(
        &self,
        segments: &[LockedSegment],
        stopped: &AtomicBool,
    ) -> OperationResult<Arc<NativeShardPointVersions>> {
        let generation = segments
            .iter()
            .map(|segment| {
                let segment = segment.get().read();
                NativeSegmentGeneration {
                    path: segment.data_path().to_path_buf(),
                    version: segment.version(),
                    is_proxy: segment.is_proxy(),
                }
            })
            .collect::<Vec<_>>();

        let mut cached = self.cached.lock();
        if let Some(cached) = cached.as_ref()
            && cached.generation == generation
        {
            return Ok(cached.point_versions.clone());
        }

        let point_versions = NativeShardPointVersions::build(segments, stopped)?;
        *cached = Some(CachedPointVersions {
            generation,
            point_versions: point_versions.clone(),
        });
        Ok(point_versions)
    }
}

impl NativeShardPointVersions {
    pub fn build(segments: &[LockedSegment], stopped: &AtomicBool) -> OperationResult<Arc<Self>> {
        let hardware_counter = HardwareCounterCell::disposable();
        let mut points = HashMap::<PointIdType, AuthoritativePointVersion>::new();
        for (source, segment) in segments.iter().enumerate() {
            let segment = segment.get().read();
            let ids = segment.read_filtered(
                None,
                None,
                None,
                stopped,
                &hardware_counter,
                DeferredBehavior::Exclude,
            )?;
            for id in ids {
                let version = segment.point_version(id).ok_or_else(|| {
                    OperationError::inconsistent_storage(format!(
                        "native exact Shard snapshot lost visible point {id} while resolving versions"
                    ))
                })?;
                match points.get_mut(&id) {
                    None => {
                        points.insert(id, AuthoritativePointVersion { version, source });
                    }
                    Some(current) if version > current.version => {
                        *current = AuthoritativePointVersion { version, source };
                    }
                    Some(current) if version == current.version && source != current.source => {
                        return Err(OperationError::inconsistent_storage(format!(
                            "native exact Shard snapshot contains point {id} at version {version} in multiple Segments"
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(Arc::new(Self { points }))
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn contains(&self, source: usize, point: &ScoredPoint) -> bool {
        self.points.get(&point.id).is_some_and(|authoritative| {
            authoritative.source == source && authoritative.version == point.version
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeStreamWorkerMode {
    Native,
    ExhaustiveFallback,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeShardStreamTelemetry {
    pub sources: usize,
    pub visible_points: usize,
    pub native_sources: usize,
    pub exhaustive_fallback_sources: usize,
    pub points_pulled: usize,
    pub points_emitted: usize,
    pub duplicates_suppressed: usize,
    /// Physical replies fetched from Segment sessions.
    pub worker_pull_batches: usize,
    /// Physical scored points returned by Segment sessions, including points
    /// buffered ahead of the channel-global consumer.
    pub worker_points_received: usize,
}

pub(crate) struct BatchReply {
    points: Vec<ScoredPoint>,
    eof: bool,
}

pub(crate) enum WorkerCommand {
    Pull {
        limit: usize,
        reply: SyncSender<OperationResult<BatchReply>>,
    },
    Stop,
}

type BoxedNativeWorker = Box<dyn FnOnce() + Send + 'static>;
type SpawnNativeWorker =
    dyn Fn(String, BoxedNativeWorker) -> OperationResult<()> + Send + Sync + 'static;

/// Qdrant-runtime hook used by exact Shard sessions.
///
/// Production constructs this from an atomically capacity-reserved search
/// session. The Shard layer remains independent of Collection/Tokio runtime
/// types and cannot accidentally submit only part of a query's worker set.
#[derive(Clone)]
pub struct NativeWorkerSpawner {
    spawn: Arc<SpawnNativeWorker>,
}

impl NativeWorkerSpawner {
    pub fn new(
        spawn: impl Fn(String, BoxedNativeWorker) -> OperationResult<()> + Send + Sync + 'static,
    ) -> Self {
        Self {
            spawn: Arc::new(spawn),
        }
    }

    pub(crate) fn spawn(
        &self,
        name: String,
        worker: impl FnOnce() + Send + 'static,
    ) -> OperationResult<()> {
        (self.spawn)(name, Box::new(worker))
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn dedicated_threads_for_tests() -> Self {
        Self::new(|name, worker| {
            std::thread::Builder::new()
                .name(name)
                .spawn(worker)
                .map(|_| ())
                .map_err(|error| {
                    OperationError::service_error_light(format!(
                        "failed to start native test worker: {error}"
                    ))
                })
        })
    }
}

pub(crate) struct WorkerCompletionSignal(Option<SyncSender<()>>);

impl WorkerCompletionSignal {
    pub(crate) fn new(sender: SyncSender<()>) -> Self {
        Self(Some(sender))
    }
}

impl Drop for WorkerCompletionSignal {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}

/// One capacity-reserved Qdrant-runtime worker owns the pinned Segment read
/// state. This client owns only its bounded pull protocol and completion
/// handshake; it never creates or manages an OS thread.
pub(crate) struct SegmentScoreSource {
    commands: SyncSender<WorkerCommand>,
    completion: Option<Receiver<()>>,
    buffer: VecDeque<ScoredPoint>,
    eof: bool,
    mode: NativeStreamWorkerMode,
    channel: &'static str,
}

impl SegmentScoreSource {
    pub(crate) fn new(
        commands: SyncSender<WorkerCommand>,
        completion: Receiver<()>,
        mode: NativeStreamWorkerMode,
        channel: &'static str,
    ) -> Self {
        Self {
            commands,
            completion: Some(completion),
            buffer: VecDeque::new(),
            eof: false,
            mode,
            channel,
        }
    }

    fn pop(&mut self, batch_size: usize) -> OperationResult<(Option<ScoredPoint>, usize)> {
        let mut fetched = 0;
        if self.buffer.is_empty() && !self.eof {
            let (reply_tx, reply_rx) = sync_channel(1);
            self.commands
                .send(WorkerCommand::Pull {
                    limit: batch_size,
                    reply: reply_tx,
                })
                .map_err(|_| {
                    OperationError::service_error_light(format!(
                        "native {} Segment worker stopped before a pull request",
                        self.channel
                    ))
                })?;
            let batch = reply_rx.recv().map_err(|_| {
                OperationError::service_error_light(format!(
                    "native {} Segment worker stopped before returning a pull result",
                    self.channel
                ))
            })??;
            fetched = batch.points.len();
            self.buffer = VecDeque::from(batch.points);
            self.eof = batch.eof;
        }
        Ok((self.buffer.pop_front(), fetched))
    }
}

impl Drop for SegmentScoreSource {
    fn drop(&mut self) {
        let _ = self.commands.send(WorkerCommand::Stop);
        if let Some(completion) = self.completion.take() {
            let _ = completion.recv();
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

/// Exact score stream merged across every frozen Segment in one Shard.
pub(crate) struct NativeShardScoreStream {
    sources: Vec<SegmentScoreSource>,
    pending: BinaryHeap<PendingPoint>,
    seen: HashSet<PointIdType>,
    batch_size: usize,
    stopped: Arc<AtomicBool>,
    channel: &'static str,
    point_versions: Arc<NativeShardPointVersions>,
    telemetry: NativeShardStreamTelemetry,
    terminal_error: Option<OperationError>,
}

impl NativeShardScoreStream {
    pub(crate) fn open(
        sources: Vec<SegmentScoreSource>,
        point_versions: Arc<NativeShardPointVersions>,
        batch_size: usize,
        stopped: Arc<AtomicBool>,
        channel: &'static str,
    ) -> OperationResult<Self> {
        if batch_size == 0 {
            return Err(OperationError::validation_error(format!(
                "native {channel} Shard batch size must be positive"
            )));
        }
        let mut telemetry = NativeShardStreamTelemetry {
            sources: sources.len(),
            visible_points: point_versions.len(),
            ..Default::default()
        };
        for source in &sources {
            match source.mode {
                NativeStreamWorkerMode::Native => telemetry.native_sources += 1,
                NativeStreamWorkerMode::ExhaustiveFallback => {
                    telemetry.exhaustive_fallback_sources += 1;
                }
            }
        }
        let mut stream = Self {
            sources,
            pending: BinaryHeap::new(),
            seen: HashSet::new(),
            batch_size,
            stopped,
            channel,
            point_versions,
            telemetry,
            terminal_error: None,
        };
        for source in 0..stream.sources.len() {
            // A k-way merge needs only one head from each Segment. Pulling the
            // full continuation batch here would perform batch_size × Segment
            // work before dynamic WRRF has requested a single global rank.
            stream.pull_source(source, 1)?;
        }
        Ok(stream)
    }

    pub(crate) fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        let result = self.next_inner();
        if let Err(error) = &result {
            self.pending.clear();
            self.terminal_error = Some(error.clone());
        }
        result
    }

    fn next_inner(&mut self) -> OperationResult<Option<ScoredPoint>> {
        loop {
            if self.stopped.load(AtomicOrdering::Relaxed) {
                return Err(OperationError::cancelled(format!(
                    "native {} Shard stream was cancelled",
                    self.channel
                )));
            }
            let Some(pending) = self.pending.pop() else {
                return Ok(None);
            };
            self.pull_source(pending.source, self.batch_size)?;
            if !self.seen.insert(pending.point.id) {
                return Err(OperationError::inconsistent_storage(format!(
                    "native {} Shard stream emitted authoritative point {} more than once",
                    self.channel, pending.point.id,
                )));
            }
            self.telemetry.points_emitted += 1;
            return Ok(Some(pending.point));
        }
    }

    pub(crate) fn telemetry(&self) -> NativeShardStreamTelemetry {
        self.telemetry
    }

    fn pull_source(&mut self, source: usize, limit: usize) -> OperationResult<()> {
        loop {
            let (point, fetched) = self.sources[source].pop(limit)?;
            if fetched != 0 {
                self.telemetry.worker_pull_batches += 1;
                self.telemetry.worker_points_received += fetched;
            }
            let Some(point) = point else {
                return Ok(());
            };
            self.telemetry.points_pulled += 1;
            if !self.point_versions.contains(source, &point) {
                self.telemetry.duplicates_suppressed += 1;
                continue;
            }
            self.pending.push(PendingPoint { source, point });
            return Ok(());
        }
    }
}

pub(crate) fn serve_native(
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

pub(crate) fn serve_materialized(points: Vec<ScoredPoint>, commands: &Receiver<WorkerCommand>) {
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
    use common::counter::hardware_counter::HardwareCounterCell;
    use segment::data_types::vectors::only_default_vector;
    use segment::entry::SegmentEntry;
    use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
    use segment::types::Distance;

    use super::*;

    fn scored(id: u64, score: f32) -> ScoredPoint {
        ScoredPoint {
            id: id.into(),
            version: 0,
            score,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    #[test]
    fn producer_failure_is_sticky_and_never_becomes_eof() {
        let (command_tx, command_rx) = sync_channel(1);
        let (completion_tx, completion_rx) = sync_channel(1);
        std::thread::spawn(move || {
            let _completion = WorkerCompletionSignal::new(completion_tx);
            let WorkerCommand::Pull { reply, .. } = command_rx.recv().unwrap() else {
                panic!("first command must prime the source")
            };
            reply
                .send(Ok(BatchReply {
                    points: vec![scored(7, 1.0)],
                    eof: false,
                }))
                .unwrap();
            let WorkerCommand::Pull { reply, .. } = command_rx.recv().unwrap() else {
                panic!("second command must refill the source")
            };
            reply
                .send(Err(OperationError::service_error_light(
                    "synthetic producer failure",
                )))
                .unwrap();
        });
        let source = SegmentScoreSource::new(
            command_tx,
            completion_rx,
            NativeStreamWorkerMode::Native,
            "Test",
        );
        let stopped = Arc::new(AtomicBool::new(false));
        let point_versions = Arc::new(NativeShardPointVersions {
            points: HashMap::from([(
                7_u64.into(),
                AuthoritativePointVersion {
                    version: 0,
                    source: 0,
                },
            )]),
        });
        let mut stream =
            NativeShardScoreStream::open(vec![source], point_versions, 8, stopped, "Test").unwrap();

        let first = stream.next_result().unwrap_err();
        let second = stream.next_result().unwrap_err();
        assert!(first.to_string().contains("synthetic producer failure"));
        assert_eq!(first.to_string(), second.to_string());
    }

    #[test]
    fn point_version_cache_reuses_and_invalidates_frozen_generation() {
        let directory = tempfile::tempdir().unwrap();
        let mut segment = build_simple_segment(directory.path(), 2, Distance::Dot).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        segment
            .upsert_point(
                1,
                7_u64.into(),
                only_default_vector(&[1.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();
        let segment = LockedSegment::new(segment);
        let segments = vec![segment.clone()];
        let cache = NativeShardPointVersionsCache::default();
        let stopped = AtomicBool::new(false);

        let first = cache.get_or_build(&segments, &stopped).unwrap();
        let second = cache.get_or_build(&segments, &stopped).unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let LockedSegment::Original(segment) = segment else {
            unreachable!()
        };
        segment
            .write()
            .upsert_point(
                2,
                8_u64.into(),
                only_default_vector(&[2.0, 0.0]),
                &hardware_counter,
            )
            .unwrap();
        let third = cache.get_or_build(&segments, &stopped).unwrap();
        assert!(!Arc::ptr_eq(&second, &third));
        assert_eq!(third.len(), 2);
    }
}
