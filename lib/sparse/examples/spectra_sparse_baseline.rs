// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::env;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use serde::Serialize;
use sparse::SearchScratchPool;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::block_max::{BlockMaxIndex, BlockMaxTelemetry, SparseDocument};
use sparse::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::search_context::{SearchContext, SearchTelemetry};

const DEFAULT_DOCUMENTS: usize = 50_000;
const DEFAULT_QUERIES: usize = 200;
const DEFAULT_TOP_K: usize = 20;
const DEFAULT_BLOCK_SIZE: usize = 256;
const CLUSTERS: u32 = 16;
const RARE_TERMS: u32 = 256;

#[derive(Debug, Default, Serialize)]
struct AggregateTelemetry {
    posting_lists: u64,
    posting_elements: u64,
    posting_elements_visited: u64,
    posting_elements_remaining: u64,
    posting_elements_skipped: u64,
    batch_count: u64,
    scored_id_span: u64,
    prune_attempts: u64,
    prune_successes: u64,
    pruning_queries: u64,
    cancelled_queries: u64,
}

impl AggregateTelemetry {
    fn add(&mut self, telemetry: SearchTelemetry) {
        self.posting_lists += telemetry.posting_lists as u64;
        self.posting_elements += telemetry.posting_elements as u64;
        self.posting_elements_visited += telemetry.posting_elements_visited as u64;
        self.posting_elements_remaining += telemetry.posting_elements_remaining as u64;
        self.posting_elements_skipped += telemetry.posting_elements_skipped as u64;
        self.batch_count += telemetry.batch_count as u64;
        self.scored_id_span += telemetry.scored_id_span as u64;
        self.prune_attempts += telemetry.prune_attempts as u64;
        self.prune_successes += telemetry.prune_successes as u64;
        self.pruning_queries += u64::from(telemetry.use_pruning);
        self.cancelled_queries += u64::from(telemetry.cancelled);
    }
}

#[derive(Debug, Serialize)]
struct LatencySummary {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Serialize)]
struct ExecutorSummary {
    latency: LatencySummary,
    telemetry: AggregateTelemetry,
}

#[derive(Debug, Default, Serialize)]
struct AggregateBlockMaxTelemetry {
    blocks: u64,
    bound_evaluations: u64,
    zero_bound_blocks: u64,
    blocks_expanded: u64,
    documents_evaluated: u64,
    points_emitted: u64,
    max_pending_blocks: usize,
    max_pending_points: usize,
}

impl AggregateBlockMaxTelemetry {
    fn add(&mut self, telemetry: BlockMaxTelemetry) {
        self.blocks += telemetry.blocks as u64;
        self.bound_evaluations += telemetry.bound_evaluations as u64;
        self.zero_bound_blocks += telemetry.zero_bound_blocks as u64;
        self.blocks_expanded += telemetry.blocks_expanded as u64;
        self.documents_evaluated += telemetry.documents_evaluated as u64;
        self.points_emitted += telemetry.points_emitted as u64;
        self.max_pending_blocks = self.max_pending_blocks.max(telemetry.max_pending_blocks);
        self.max_pending_points = self.max_pending_points.max(telemetry.max_pending_points);
    }
}

#[derive(Debug, Serialize)]
struct BlockMaxExecutorSummary {
    build_ns: u128,
    block_size: usize,
    blocks: usize,
    envelope_nonzeros: usize,
    latency: LatencySummary,
    telemetry: AggregateBlockMaxTelemetry,
}

#[derive(Debug, Serialize)]
struct ExperimentResult {
    schema_version: usize,
    experiment: &'static str,
    qdrant_base_commit: &'static str,
    build_profile: &'static str,
    weight_pattern: &'static str,
    documents: usize,
    queries: usize,
    top_k: usize,
    native_parity_mismatches: usize,
    block_max_parity_mismatches: usize,
    native: ExecutorSummary,
    block_max: BlockMaxExecutorSummary,
    exhaustive: ExecutorSummary,
}

#[derive(Clone, Copy)]
enum WeightPattern {
    Descending,
    Ascending,
    Flat,
}

impl WeightPattern {
    fn from_env() -> Self {
        match env::var("SPECTRA_WEIGHT_PATTERN").as_deref() {
            Ok("ascending") => Self::Ascending,
            Ok("flat") => Self::Flat,
            Ok("descending") | Err(_) => Self::Descending,
            Ok(value) => {
                panic!("SPECTRA_WEIGHT_PATTERN must be descending, ascending, or flat; got {value}")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Descending => "descending",
            Self::Ascending => "ascending",
            Self::Flat => "flat",
        }
    }

    fn position_weight(self, document: usize, documents: usize) -> f32 {
        let position = document as f32 / documents as f32;
        match self {
            Self::Descending => 1.0 - position,
            Self::Ascending => position,
            Self::Flat => 1.0,
        }
    }
}

fn main() {
    let document_count = env_usize("SPECTRA_DOCUMENTS", DEFAULT_DOCUMENTS);
    let queries = env_usize("SPECTRA_QUERIES", DEFAULT_QUERIES);
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let block_size = env_usize("SPECTRA_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    assert!(document_count > 0, "SPECTRA_DOCUMENTS must be positive");
    assert!(queries > 0, "SPECTRA_QUERIES must be positive");
    assert!(top_k > 0, "SPECTRA_TOP_K must be positive");
    assert!(block_size > 0, "SPECTRA_BLOCK_SIZE must be positive");

    let weight_pattern = WeightPattern::from_env();
    let documents = build_documents(document_count, weight_pattern);
    let index = InvertedIndexBuilder::build_from_iterator(
        documents
            .iter()
            .map(|document| (document.id, document.vector.clone())),
    );
    let block_build_start = Instant::now();
    let block_index = BlockMaxIndex::build(documents.clone(), block_size).unwrap();
    let block_build_elapsed = block_build_start.elapsed();
    let query_vectors = build_queries(queries);
    let document_ids: Vec<_> = (0..document_count as PointOffsetType).collect();
    let pool = SearchScratchPool::new();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::new();

    let mut native_latencies = Vec::with_capacity(queries);
    let mut block_max_latencies = Vec::with_capacity(queries);
    let mut exhaustive_latencies = Vec::with_capacity(queries);
    let mut native_telemetry = AggregateTelemetry::default();
    let mut block_max_telemetry = AggregateBlockMaxTelemetry::default();
    let mut exhaustive_telemetry = AggregateTelemetry::default();
    let mut native_parity_mismatches = 0;
    let mut block_max_parity_mismatches = 0;

    for query in query_vectors {
        let (native, native_elapsed, native_progress) = run_native(
            &index,
            query.clone(),
            top_k,
            &pool,
            &stopped,
            &hardware_counter,
        );
        let (exhaustive, exhaustive_elapsed, exhaustive_progress) = run_exhaustive(
            &index,
            query.clone(),
            top_k,
            &document_ids,
            &pool,
            &stopped,
            &hardware_counter,
        );
        let (block_max, block_max_elapsed, block_max_progress) =
            run_block_max(&block_index, query, top_k);

        native_parity_mismatches += usize::from(native != exhaustive);
        block_max_parity_mismatches += usize::from(block_max != exhaustive);
        native_latencies.push(native_elapsed);
        block_max_latencies.push(block_max_elapsed);
        exhaustive_latencies.push(exhaustive_elapsed);
        native_telemetry.add(native_progress);
        block_max_telemetry.add(block_max_progress);
        exhaustive_telemetry.add(exhaustive_progress);
    }

    let result = ExperimentResult {
        schema_version: 1,
        experiment: "spectra-sparse-native-synthetic-v1",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        weight_pattern: weight_pattern.name(),
        documents: document_count,
        queries,
        top_k,
        native_parity_mismatches,
        block_max_parity_mismatches,
        native: ExecutorSummary {
            latency: summarize_latency(&mut native_latencies),
            telemetry: native_telemetry,
        },
        block_max: BlockMaxExecutorSummary {
            build_ns: block_build_elapsed.as_nanos(),
            block_size: block_index.block_size(),
            blocks: block_index.block_count(),
            envelope_nonzeros: block_index.envelope_nonzero_count(),
            latency: summarize_latency(&mut block_max_latencies),
            telemetry: block_max_telemetry,
        },
        exhaustive: ExecutorSummary {
            latency: summarize_latency(&mut exhaustive_latencies),
            telemetry: exhaustive_telemetry,
        },
    };

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(
        native_parity_mismatches, 0,
        "native Sparse results diverged from exhaustive search"
    );
    assert_eq!(
        block_max_parity_mismatches, 0,
        "Block Max results diverged from exhaustive search"
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

fn build_documents(documents: usize, weight_pattern: WeightPattern) -> Vec<SparseDocument> {
    (0..documents)
        .map(|document| {
            let cluster = 1 + document as u32 % CLUSTERS;
            let rare = 1 + CLUSTERS + document as u32 % RARE_TERMS;
            let position_weight = weight_pattern.position_weight(document, documents);
            let common_weight = 0.5 + 0.5 * position_weight;
            let cluster_weight = 1.0 + position_weight;
            let rare_weight = 2.0 + position_weight;
            SparseDocument {
                id: document as PointOffsetType,
                vector: RemappedSparseVector {
                    indices: vec![0, cluster, rare],
                    values: vec![common_weight, cluster_weight, rare_weight],
                },
            }
        })
        .collect()
}

fn build_queries(queries: usize) -> Vec<RemappedSparseVector> {
    (0..queries)
        .map(|query| RemappedSparseVector {
            indices: vec![
                0,
                1 + query as u32 % CLUSTERS,
                1 + CLUSTERS + (query * 37) as u32 % RARE_TERMS,
            ],
            values: vec![1.0, 2.0, 3.0],
        })
        .collect()
}

fn run_native(
    index: &InvertedIndexRam,
    query: RemappedSparseVector,
    top_k: usize,
    pool: &SearchScratchPool,
    stopped: &AtomicBool,
    hardware_counter: &HardwareCounterCell,
) -> (Vec<ScoredPointOffset>, Duration, SearchTelemetry) {
    let start = Instant::now();
    let mut scratch = pool.get();
    let mut context =
        SearchContext::new(query, top_k, index, &mut scratch, stopped, hardware_counter).unwrap();
    let result = context.search(&|_| true);
    let elapsed = start.elapsed();
    (result, elapsed, context.telemetry())
}

fn run_block_max(
    index: &BlockMaxIndex,
    query: RemappedSparseVector,
    top_k: usize,
) -> (Vec<ScoredPointOffset>, Duration, BlockMaxTelemetry) {
    let start = Instant::now();
    let mut stream = index.stream(query).unwrap();
    let result = stream.by_ref().take(top_k).collect();
    let elapsed = start.elapsed();
    (result, elapsed, stream.telemetry())
}

#[expect(clippy::too_many_arguments)]
fn run_exhaustive(
    index: &InvertedIndexRam,
    query: RemappedSparseVector,
    top_k: usize,
    document_ids: &[PointOffsetType],
    pool: &SearchScratchPool,
    stopped: &AtomicBool,
    hardware_counter: &HardwareCounterCell,
) -> (Vec<ScoredPointOffset>, Duration, SearchTelemetry) {
    let start = Instant::now();
    let mut scratch = pool.get();
    let mut context =
        SearchContext::new(query, top_k, index, &mut scratch, stopped, hardware_counter).unwrap();
    let result = context.plain_search(document_ids);
    let elapsed = start.elapsed();
    (result, elapsed, context.telemetry())
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
