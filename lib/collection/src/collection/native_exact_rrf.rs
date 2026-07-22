// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Native exact Dense/Sparse channel execution followed by dynamic WRRF.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::common::reciprocal_rank_fusion::{
    DynamicRrfExecution, DynamicRrfPolicy, DynamicRrfScheduler, DynamicRrfStopReason,
    ExactRrfStream, execute_dynamic_rrf_with_policy,
};
use segment::index::native_dense_stream::NativeDensePolicy;
use segment::types::{ExtendedPointId, Filter, PointIdType, ScoredPoint, VectorNameBuf};
use shard::locked_segment::LockedSegment;
use shard::native_dense_stream::NativeDenseShardStream;
use shard::native_sparse_stream::NativeSparseShardStream;
use sparse::common::sparse_vector::SparseVector;

use super::Collection;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::{CollectionError, CollectionResult};

pub const DEFAULT_NATIVE_EXACT_BATCH_SIZE: usize = 64;
pub const DEFAULT_NATIVE_SPARSE_POSTING_BATCH_SIZE: usize = 4_096;

#[derive(Clone, Debug)]
pub struct NativeExactRrfRequest {
    pub dense_query: Vec<f32>,
    pub dense_using: VectorNameBuf,
    pub sparse_query: SparseVector,
    pub sparse_using: VectorNameBuf,
    pub filter: Option<Filter>,
    pub limit: usize,
    pub rrf_k: usize,
    pub weights: [f32; 2],
    pub batch_size: usize,
    pub sparse_posting_batch_size: usize,
    pub dense_policy: NativeDensePolicy,
}

#[derive(Debug)]
pub struct NativeExactRrfResult {
    pub point_ids: Vec<ExtendedPointId>,
    pub versions: HashMap<PointIdType, u64>,
    pub stop_reason: DynamicRrfStopReason,
    pub source_pulls: Vec<usize>,
    pub source_exhausted: Vec<bool>,
    pub certification_checks: usize,
    pub source_points_materialized: Vec<usize>,
    pub exhaustive_fallback_sources: usize,
    pub shard_count: usize,
}

trait ScoredStream: Send {
    fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>>;
}

impl ScoredStream for NativeDenseShardStream {
    fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        NativeDenseShardStream::next_result(self)
    }
}

impl ScoredStream for NativeSparseShardStream {
    fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        NativeSparseShardStream::next_result(self)
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

/// Merge Shard-local exact score streams into one exact channel rank stream.
/// Scores are compared only within this channel.
struct MergedChannelStream {
    sources: Vec<Box<dyn ScoredStream>>,
    pending: BinaryHeap<PendingPoint>,
    seen: HashSet<PointIdType>,
}

impl MergedChannelStream {
    fn new(sources: Vec<Box<dyn ScoredStream>>) -> OperationResult<Self> {
        let mut stream = Self {
            sources,
            pending: BinaryHeap::new(),
            seen: HashSet::new(),
        };
        for source in 0..stream.sources.len() {
            stream.pull_source(source)?;
        }
        Ok(stream)
    }

    fn next_result(&mut self) -> OperationResult<Option<ScoredPoint>> {
        loop {
            let Some(pending) = self.pending.pop() else {
                return Ok(None);
            };
            self.pull_source(pending.source)?;
            if self.seen.insert(pending.point.id) {
                return Ok(Some(pending.point));
            }
        }
    }

    fn pull_source(&mut self, source: usize) -> OperationResult<()> {
        if let Some(point) = self.sources[source].next_result()? {
            self.pending.push(PendingPoint { source, point });
        }
        Ok(())
    }
}

struct IdentityStream {
    inner: MergedChannelStream,
    versions: Arc<Mutex<HashMap<PointIdType, u64>>>,
    materialized: Arc<AtomicUsize>,
}

impl Iterator for IdentityStream {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next_result() {
            Ok(Some(point)) => {
                self.materialized.fetch_add(1, AtomicOrdering::Relaxed);
                self.versions
                    .lock()
                    .entry(point.id)
                    .and_modify(|version| *version = (*version).max(point.version))
                    .or_insert(point.version);
                Some(Ok(point.id))
            }
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        }
    }
}

impl Collection {
    /// Execute the local-native physical plan when every selected Shard has a
    /// readable local replica. Returns `None` when the safe router must use the
    /// ordinary replica/remote path instead. Plan selection cannot change the
    /// exact result contract.
    pub async fn native_exact_rrf(
        &self,
        request: NativeExactRrfRequest,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
    ) -> CollectionResult<Option<NativeExactRrfResult>> {
        let targets = {
            let shard_holder = self.shards_holder.read().await;
            shard_holder
                .select_shards(shard_selection)?
                .into_iter()
                .map(|(shard, _)| Arc::clone(shard))
                .collect::<Vec<_>>()
        };

        let mut snapshots = Vec::with_capacity(targets.len());
        for target in targets {
            let Some(snapshot) = target.native_segment_snapshot().await? else {
                return Ok(None);
            };
            snapshots.push(snapshot);
        }

        let stopped = Arc::new(AtomicBool::new(false));
        let task_stopped = stopped.clone();
        let shard_count = snapshots.len();
        let mut task = tokio::task::spawn_blocking(move || {
            execute_native_exact_rrf(snapshots, request, task_stopped)
        });
        let result = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, &mut task).await {
                Ok(result) => result,
                Err(_) => {
                    stopped.store(true, AtomicOrdering::Relaxed);
                    return Err(CollectionError::timeout(timeout, "native exact RRF"));
                }
            }
        } else {
            task.await
        }
        .map_err(|error| {
            CollectionError::service_error(format!("native exact RRF task failed: {error}"))
        })??;

        Ok(Some(NativeExactRrfResult {
            shard_count,
            ..result
        }))
    }
}

fn execute_native_exact_rrf(
    snapshots: Vec<Vec<LockedSegment>>,
    request: NativeExactRrfRequest,
    stopped: Arc<AtomicBool>,
) -> OperationResult<NativeExactRrfResult> {
    let batch_size = if request.batch_size == 0 {
        DEFAULT_NATIVE_EXACT_BATCH_SIZE
    } else {
        request.batch_size
    };
    let sparse_posting_batch_size = if request.sparse_posting_batch_size == 0 {
        DEFAULT_NATIVE_SPARSE_POSTING_BATCH_SIZE
    } else {
        request.sparse_posting_batch_size
    };

    // Sparse workers are opened for every Segment first. Their pinned read
    // views freeze the corpus while Dense opens over the same identities.
    let mut sparse_sources = Vec::with_capacity(snapshots.len());
    let mut exhaustive_fallback_sources = 0;
    for segments in snapshots.iter().cloned() {
        let stream = NativeSparseShardStream::open(
            segments,
            request.sparse_using.clone(),
            request.sparse_query.clone(),
            request.filter.clone(),
            batch_size,
            sparse_posting_batch_size,
            stopped.clone(),
        )?;
        exhaustive_fallback_sources += stream.telemetry().exhaustive_fallback_sources;
        sparse_sources.push(Box::new(stream) as Box<dyn ScoredStream>);
    }
    let dense_sources = snapshots
        .into_iter()
        .map(|segments| {
            NativeDenseShardStream::open(
                segments,
                request.dense_using.clone(),
                request.dense_query.clone(),
                request.filter.clone(),
                request.dense_policy,
                batch_size,
                stopped.clone(),
            )
            .map(|stream| Box::new(stream) as Box<dyn ScoredStream>)
        })
        .collect::<OperationResult<Vec<_>>>()?;

    let versions = Arc::new(Mutex::new(HashMap::new()));
    let materialized = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
    let dense_stream = IdentityStream {
        inner: MergedChannelStream::new(dense_sources)?,
        versions: versions.clone(),
        materialized: materialized[0].clone(),
    };
    let sparse_stream = IdentityStream {
        inner: MergedChannelStream::new(sparse_sources)?,
        versions: versions.clone(),
        materialized: materialized[1].clone(),
    };
    let streams: Vec<ExactRrfStream<'static>> =
        vec![Box::new(dense_stream), Box::new(sparse_stream)];
    let DynamicRrfExecution {
        point_ids,
        source_pulls,
        source_exhausted,
        certification_checks,
        stop_reason,
        ..
    } = execute_dynamic_rrf_with_policy(
        streams,
        request.limit,
        request.rrf_k,
        Some(&request.weights),
        DynamicRrfPolicy {
            scheduler: DynamicRrfScheduler::MaxNextContribution,
            ..DynamicRrfPolicy::default()
        },
    )?;
    let versions = Arc::try_unwrap(versions)
        .map_err(|_| OperationError::service_error_light("native exact RRF version map leaked"))?
        .into_inner();
    let source_points_materialized = materialized
        .into_iter()
        .map(|count| count.load(AtomicOrdering::Relaxed))
        .collect();

    Ok(NativeExactRrfResult {
        point_ids,
        versions,
        stop_reason,
        source_pulls,
        source_exhausted,
        certification_checks,
        source_points_materialized,
        exhaustive_fallback_sources,
        shard_count: 0,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use common::counter::hardware_counter::HardwareCounterCell;
    use segment::common::reciprocal_rank_fusion::exact_rrf_scoring;
    use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
    use segment::entry::SegmentEntry;
    use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
    use segment::segment::Segment;
    use segment::segment_constructor::build_segment;
    use segment::types::{
        Distance, Indexes, SegmentConfig, SparseVectorDataConfig, SparseVectorStorageType,
        VectorDataConfig, VectorStorageType,
    };

    use super::*;

    const SPARSE_NAME: &str = "sparse";

    fn scored(id: PointIdType, score: f32) -> ScoredPoint {
        ScoredPoint {
            id,
            version: 0,
            score,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    fn make_segment(
        lane: u64,
    ) -> (
        tempfile::TempDir,
        Segment,
        Vec<ScoredPoint>,
        Vec<ScoredPoint>,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let config = SegmentConfig {
            vector_data: HashMap::from([(
                DEFAULT_VECTOR_NAME.to_owned(),
                VectorDataConfig {
                    size: 2,
                    distance: Distance::Dot,
                    storage_type: VectorStorageType::default(),
                    index: Indexes::Plain {},
                    quantization_config: None,
                    multivector_config: None,
                    datatype: None,
                },
            )]),
            sparse_vector_data: HashMap::from([(
                SPARSE_NAME.to_owned(),
                SparseVectorDataConfig {
                    index: SparseIndexConfig::new(Some(1), SparseIndexType::MutableRam, None),
                    storage_type: SparseVectorStorageType::Mmap,
                    modifier: None,
                },
            )]),
            payload_storage_type: Default::default(),
        };
        let mut segment = build_segment(directory.path(), &config, None, true).unwrap();
        let hardware_counter = HardwareCounterCell::new();
        let mut dense = Vec::new();
        let mut sparse = Vec::new();
        for index in 0..128u64 {
            let id: PointIdType = (40_000 - (index * 2 + lane)).into();
            let dense_score = 1.0 + ((index * 5 + lane) % 17) as f32;
            let sparse_score = 1.0 + ((index * 7 + lane) % 19) as f32;
            let dense_vector = [dense_score, index as f32];
            let mut vectors = only_default_vector(&dense_vector);
            vectors.insert(
                SPARSE_NAME.to_owned(),
                SparseVector {
                    indices: vec![11],
                    values: vec![sparse_score],
                }
                .into(),
            );
            segment
                .upsert_point(index, id, vectors, &hardware_counter)
                .unwrap();
            dense.push(scored(id, dense_score));
            sparse.push(scored(id, sparse_score));
        }
        (directory, segment, dense, sparse)
    }

    #[test]
    fn native_execution_equals_exhaustive_dense_sparse_wrrf() {
        let (first_dir, first, mut dense, mut sparse) = make_segment(0);
        let (second_dir, second, second_dense, second_sparse) = make_segment(1);
        dense.extend(second_dense);
        sparse.extend(second_sparse);
        for ranking in [&mut dense, &mut sparse] {
            ranking.sort_unstable_by(|left, right| {
                OrderedFloat(right.score)
                    .cmp(&OrderedFloat(left.score))
                    .then_with(|| left.id.cmp(&right.id))
            });
        }
        let expected = exact_rrf_scoring(vec![dense, sparse], 60, Some(&[1.0, 1.0]))
            .unwrap()
            .into_iter()
            .take(20)
            .map(|point| point.id)
            .collect::<Vec<_>>();

        let actual = execute_native_exact_rrf(
            vec![vec![LockedSegment::new(first), LockedSegment::new(second)]],
            NativeExactRrfRequest {
                dense_query: vec![1.0, 0.0],
                dense_using: DEFAULT_VECTOR_NAME.to_owned(),
                sparse_query: SparseVector {
                    indices: vec![11],
                    values: vec![1.0],
                },
                sparse_using: SPARSE_NAME.to_owned(),
                filter: None,
                limit: 20,
                rrf_k: 60,
                weights: [1.0, 1.0],
                batch_size: 13,
                sparse_posting_batch_size: 4_096,
                dense_policy: NativeDensePolicy::default(),
            },
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();

        assert_eq!(actual.point_ids, expected);
        assert_eq!(actual.point_ids.len(), 20);
        assert!(actual.source_pulls.iter().all(|pulls| *pulls < 256));
        drop((first_dir, second_dir));
    }
}
