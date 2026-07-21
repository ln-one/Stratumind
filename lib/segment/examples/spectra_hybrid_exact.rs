// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::cell::Cell;
use std::env;
use std::rc::Rc;
use std::time::{Duration, Instant};

use common::types::{PointOffsetType, ScoredPointOffset};
use segment::common::reciprocal_rank_fusion::{
    DEFAULT_RRF_K, DynamicRrfStopReason, ExactRrfStream, exact_rrf_scoring, execute_dynamic_rrf,
};
use segment::index::dense_ball::{
    DenseBallIndex, DenseBallStream, DenseBallTelemetry, DenseDocument,
};
use segment::types::{ExtendedPointId, ScoredPoint};
use serde::Serialize;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::block_max::{BlockMaxIndex, BlockMaxStream, BlockMaxTelemetry, SparseDocument};

const DEFAULT_DOCUMENTS: usize = 20_000;
const DEFAULT_QUERIES: usize = 100;
const DEFAULT_TOP_K: usize = 20;
const DEFAULT_BLOCK_SIZE: usize = 128;
const DEFAULT_DIMENSION: usize = 32;
const CLUSTERS: usize = 16;
const RARE_TERMS: u32 = 256;

#[derive(Debug, Serialize)]
struct LatencySummary {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Default, Serialize)]
struct AggregatePhysicalTelemetry {
    sparse_blocks_expanded: u64,
    sparse_documents_evaluated: u64,
    dense_blocks_expanded: u64,
    dense_documents_evaluated: u64,
}

impl AggregatePhysicalTelemetry {
    fn add_sparse(&mut self, telemetry: BlockMaxTelemetry) {
        self.sparse_blocks_expanded += telemetry.blocks_expanded as u64;
        self.sparse_documents_evaluated += telemetry.documents_evaluated as u64;
    }

    fn add_dense(&mut self, telemetry: DenseBallTelemetry) {
        self.dense_blocks_expanded += telemetry.blocks_expanded as u64;
        self.dense_documents_evaluated += telemetry.documents_evaluated as u64;
    }
}

#[derive(Debug, Serialize)]
struct HybridResult {
    schema_version: usize,
    experiment: &'static str,
    qdrant_base_commit: &'static str,
    build_profile: &'static str,
    scenario: &'static str,
    documents: usize,
    queries: usize,
    top_k: usize,
    rrf_k: usize,
    block_size: usize,
    dense_dimension: usize,
    sparse_build_ns: u128,
    dense_build_ns: u128,
    sparse_blocks: usize,
    dense_blocks: usize,
    parity_mismatches: usize,
    fixed_before_exhaustion: usize,
    all_sources_exhausted: usize,
    dynamic_source_pulls: [u64; 2],
    certification_checks: u64,
    dynamic_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
    dynamic_physical: AggregatePhysicalTelemetry,
    exhaustive_physical: AggregatePhysicalTelemetry,
}

#[derive(Clone, Copy)]
enum Scenario {
    Clustered,
    Interleaved,
    AntiCorrelated,
    FlatTies,
}

impl Scenario {
    fn from_env() -> Self {
        match env::var("SPECTRA_SCENARIO").as_deref() {
            Ok("interleaved") => Self::Interleaved,
            Ok("anti_correlated") => Self::AntiCorrelated,
            Ok("flat_ties") => Self::FlatTies,
            Ok("clustered") | Err(_) => Self::Clustered,
            Ok(value) => panic!(
                "SPECTRA_SCENARIO must be clustered, interleaved, anti_correlated, or flat_ties; got {value}"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Clustered => "clustered",
            Self::Interleaved => "interleaved",
            Self::AntiCorrelated => "anti_correlated",
            Self::FlatTies => "flat_ties",
        }
    }
}

struct SparseIdentityStream<'a> {
    inner: BlockMaxStream<'a>,
    telemetry: Rc<Cell<BlockMaxTelemetry>>,
}

impl Iterator for SparseIdentityStream<'_> {
    type Item = ExtendedPointId;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry.set(self.inner.telemetry());
        point.map(|point| ExtendedPointId::from(point.idx as u64))
    }
}

struct DenseIdentityStream<'a> {
    inner: DenseBallStream<'a>,
    telemetry: Rc<Cell<DenseBallTelemetry>>,
}

impl Iterator for DenseIdentityStream<'_> {
    type Item = ExtendedPointId;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry.set(self.inner.telemetry());
        point.map(|point| ExtendedPointId::from(point.id as u64))
    }
}

fn main() {
    let document_count = env_usize("SPECTRA_DOCUMENTS", DEFAULT_DOCUMENTS);
    let query_count = env_usize("SPECTRA_QUERIES", DEFAULT_QUERIES);
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let block_size = env_usize("SPECTRA_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    let dimension = env_usize("SPECTRA_DENSE_DIMENSION", DEFAULT_DIMENSION);
    let scenario = Scenario::from_env();
    assert!(document_count >= CLUSTERS);
    assert!(query_count > 0);
    assert!(top_k > 0);
    assert!(block_size > 0);
    assert!(dimension >= CLUSTERS + 1);

    let (sparse_documents, dense_documents) = build_documents(document_count, dimension, scenario);
    let sparse_build_start = Instant::now();
    let sparse_index = BlockMaxIndex::build(sparse_documents, block_size).unwrap();
    let sparse_build_elapsed = sparse_build_start.elapsed();
    let dense_build_start = Instant::now();
    let dense_index = DenseBallIndex::build(dense_documents, block_size).unwrap();
    let dense_build_elapsed = dense_build_start.elapsed();

    let mut dynamic_latencies = Vec::with_capacity(query_count);
    let mut exhaustive_latencies = Vec::with_capacity(query_count);
    let mut dynamic_physical = AggregatePhysicalTelemetry::default();
    let mut exhaustive_physical = AggregatePhysicalTelemetry::default();
    let mut parity_mismatches = 0;
    let mut fixed_before_exhaustion = 0;
    let mut all_sources_exhausted = 0;
    let mut dynamic_source_pulls = [0u64; 2];
    let mut certification_checks = 0u64;

    for query_number in 0..query_count {
        let (sparse_query, dense_query) = build_query(query_number, dimension, scenario);

        let exhaustive_start = Instant::now();
        let (sparse_ranking, sparse_full_telemetry) =
            collect_sparse(&sparse_index, sparse_query.clone());
        let (dense_ranking, dense_full_telemetry) = collect_dense(&dense_index, &dense_query);
        let exhaustive = exhaustive_rrf(&[sparse_ranking, dense_ranking], top_k);
        exhaustive_latencies.push(exhaustive_start.elapsed());
        exhaustive_physical.add_sparse(sparse_full_telemetry);
        exhaustive_physical.add_dense(dense_full_telemetry);

        let sparse_telemetry = Rc::new(Cell::new(BlockMaxTelemetry::default()));
        let dense_telemetry = Rc::new(Cell::new(DenseBallTelemetry::default()));
        let sources: Vec<ExactRrfStream<'_>> = vec![
            Box::new(SparseIdentityStream {
                inner: sparse_index.stream(sparse_query).unwrap(),
                telemetry: Rc::clone(&sparse_telemetry),
            }),
            Box::new(DenseIdentityStream {
                inner: dense_index.stream(&dense_query).unwrap(),
                telemetry: Rc::clone(&dense_telemetry),
            }),
        ];
        let dynamic_start = Instant::now();
        let dynamic = execute_dynamic_rrf(sources, top_k, DEFAULT_RRF_K, None).unwrap();
        dynamic_latencies.push(dynamic_start.elapsed());

        parity_mismatches += usize::from(dynamic.point_ids != exhaustive);
        match dynamic.stop_reason {
            DynamicRrfStopReason::TopKFixed => fixed_before_exhaustion += 1,
            DynamicRrfStopReason::AllSourcesExhausted => all_sources_exhausted += 1,
        }
        dynamic_source_pulls[0] += dynamic.source_pulls[0] as u64;
        dynamic_source_pulls[1] += dynamic.source_pulls[1] as u64;
        certification_checks += dynamic.certification_checks as u64;
        dynamic_physical.add_sparse(sparse_telemetry.get());
        dynamic_physical.add_dense(dense_telemetry.get());
    }

    let result = HybridResult {
        schema_version: 1,
        experiment: "spectra-hybrid-dynamic-exact-v1",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        scenario: scenario.name(),
        documents: document_count,
        queries: query_count,
        top_k,
        rrf_k: DEFAULT_RRF_K,
        block_size,
        dense_dimension: dimension,
        sparse_build_ns: sparse_build_elapsed.as_nanos(),
        dense_build_ns: dense_build_elapsed.as_nanos(),
        sparse_blocks: sparse_index.block_count(),
        dense_blocks: dense_index.block_count(),
        parity_mismatches,
        fixed_before_exhaustion,
        all_sources_exhausted,
        dynamic_source_pulls,
        certification_checks,
        dynamic_latency: summarize_latency(&mut dynamic_latencies),
        exhaustive_latency: summarize_latency(&mut exhaustive_latencies),
        dynamic_physical,
        exhaustive_physical,
    };

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(
        parity_mismatches, 0,
        "dynamic hybrid WRRF diverged from exhaustive hybrid WRRF"
    );
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer"))
        })
        .unwrap_or(default)
}

fn build_documents(
    document_count: usize,
    dimension: usize,
    scenario: Scenario,
) -> (Vec<SparseDocument>, Vec<DenseDocument>) {
    let documents_per_cluster = document_count.div_ceil(CLUSTERS);
    let mut sparse_documents = Vec::with_capacity(document_count);
    let mut dense_documents = Vec::with_capacity(document_count);

    for id in 0..document_count {
        let cluster = match scenario {
            Scenario::Interleaved => id % CLUSTERS,
            Scenario::FlatTies => 0,
            Scenario::Clustered | Scenario::AntiCorrelated => {
                (id / documents_per_cluster).min(CLUSTERS - 1)
            }
        };
        let local = id % documents_per_cluster;
        let rare = 1 + CLUSTERS as u32 + id as u32 % RARE_TERMS;
        sparse_documents.push(SparseDocument {
            id: id as PointOffsetType,
            vector: RemappedSparseVector {
                indices: vec![0, 1 + cluster as u32, rare],
                values: vec![0.5, 2.0, 3.0],
            },
        });

        let mut vector = vec![0.0; dimension];
        vector[cluster] = 1.0;
        if !matches!(scenario, Scenario::FlatTies) {
            vector[CLUSTERS + local % (dimension - CLUSTERS)] =
                ((local * 17 % 23) as f32 - 11.0) / 220.0;
        }
        normalize(&mut vector);
        dense_documents.push(DenseDocument {
            id: id as PointOffsetType,
            vector,
        });
    }

    (sparse_documents, dense_documents)
}

fn build_query(
    query_number: usize,
    dimension: usize,
    scenario: Scenario,
) -> (RemappedSparseVector, Vec<f32>) {
    if matches!(scenario, Scenario::FlatTies) {
        let mut dense = vec![0.0; dimension];
        dense[0] = 1.0;
        return (
            RemappedSparseVector {
                indices: vec![0],
                values: vec![1.0],
            },
            dense,
        );
    }

    let sparse_cluster = query_number % CLUSTERS;
    let dense_cluster = if matches!(scenario, Scenario::AntiCorrelated) {
        CLUSTERS - 1 - sparse_cluster
    } else {
        sparse_cluster
    };
    let rare = 1 + CLUSTERS as u32 + (query_number * 37) as u32 % RARE_TERMS;
    let sparse = RemappedSparseVector {
        indices: vec![0, 1 + sparse_cluster as u32, rare],
        values: vec![1.0, 2.0, 3.0],
    };
    let mut dense = vec![0.0; dimension];
    dense[dense_cluster] = 1.0;
    dense[CLUSTERS + query_number % (dimension - CLUSTERS)] = 0.05;
    normalize(&mut dense);
    (sparse, dense)
}

fn normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|coordinate| coordinate * coordinate)
        .sum::<f32>()
        .sqrt();
    for coordinate in vector {
        *coordinate /= norm;
    }
}

fn collect_sparse(
    index: &BlockMaxIndex,
    query: RemappedSparseVector,
) -> (Vec<ExtendedPointId>, BlockMaxTelemetry) {
    let mut stream = index.stream(query).unwrap();
    let ranking = stream
        .by_ref()
        .map(|point: ScoredPointOffset| ExtendedPointId::from(point.idx as u64))
        .collect();
    (ranking, stream.telemetry())
}

fn collect_dense(
    index: &DenseBallIndex,
    query: &[f32],
) -> (Vec<ExtendedPointId>, DenseBallTelemetry) {
    let mut stream = index.stream(query).unwrap();
    let ranking = stream
        .by_ref()
        .map(|point| ExtendedPointId::from(point.id as u64))
        .collect();
    (ranking, stream.telemetry())
}

fn exhaustive_rrf(sources: &[Vec<ExtendedPointId>], top_k: usize) -> Vec<ExtendedPointId> {
    let responses = sources
        .iter()
        .map(|source| source.iter().copied().map(scored_point).collect())
        .collect();
    exact_rrf_scoring(responses, DEFAULT_RRF_K, None)
        .unwrap()
        .into_iter()
        .take(top_k)
        .map(|point| point.id)
        .collect()
}

fn scored_point(id: ExtendedPointId) -> ScoredPoint {
    ScoredPoint {
        id,
        version: 0,
        score: 0.0,
        payload: None,
        vector: None,
        shard_key: None,
        order_value: None,
    }
}

fn summarize_latency(latencies: &mut [Duration]) -> LatencySummary {
    latencies.sort_unstable();
    let total: u128 = latencies.iter().map(Duration::as_nanos).sum();
    LatencySummary {
        p50_ns: percentile(latencies, 50).as_nanos(),
        p95_ns: percentile(latencies, 95).as_nanos(),
        p99_ns: percentile(latencies, 99).as_nanos(),
        mean_ns: total / latencies.len() as u128,
    }
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    let index = (sorted.len() - 1) * percentile / 100;
    sorted[index]
}
