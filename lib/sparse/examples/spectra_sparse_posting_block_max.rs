// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::borrow::Cow;
use std::env;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use serde::Serialize;
use sparse::SearchScratchPool;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::adaptive_top_k_stream::{AdaptiveTopKStream, AdaptiveTopKStreamTelemetry};
use sparse::index::inverted_index::InvertedIndex;
use sparse::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::posting_block_stream::{PostingBlockStream, PostingBlockStreamTelemetry};
use sparse::index::search_context::{SearchContext, SearchTelemetry};

const DEFAULT_DOCUMENTS: usize = 100_000;
const DEFAULT_QUERIES: usize = 500;
const DEFAULT_TOP_K: usize = 20;
const CLUSTERS: u32 = 16;
const RARE_TERMS: u32 = 256;

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

#[derive(Debug, Default, Serialize)]
struct AggregateTelemetry {
    posting_lists: u64,
    posting_elements: u64,
    posting_elements_visited: u64,
    posting_elements_skipped: u64,
    batch_count: u64,
    scored_id_span: u64,
    block_prune_attempts: u64,
    block_prune_successes: u64,
    router_disabled_queries: u64,
    trend_disabled_queries: u64,
}

impl AggregateTelemetry {
    fn add(&mut self, telemetry: SearchTelemetry) {
        self.posting_lists += telemetry.posting_lists as u64;
        self.posting_elements += telemetry.posting_elements as u64;
        self.posting_elements_visited += telemetry.posting_elements_visited as u64;
        self.posting_elements_skipped += telemetry.posting_elements_skipped as u64;
        self.batch_count += telemetry.batch_count as u64;
        self.scored_id_span += telemetry.scored_id_span as u64;
        self.block_prune_attempts += telemetry.block_prune_attempts as u64;
        self.block_prune_successes += telemetry.block_prune_successes as u64;
        self.router_disabled_queries += u64::from(telemetry.block_pruning_router_disabled);
        self.trend_disabled_queries += u64::from(telemetry.block_pruning_trend_disabled);
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
struct AggregatePostingStreamTelemetry {
    posting_lists: u64,
    batches: u64,
    bound_evaluations: u64,
    zero_bound_batches: u64,
    batches_expanded: u64,
    posting_elements_visited: u64,
    nonzero_documents_scored: u64,
    points_emitted: u64,
    max_pending_batches: usize,
    max_pending_points: usize,
    max_buffered_points: usize,
}

impl AggregatePostingStreamTelemetry {
    fn add(&mut self, telemetry: PostingBlockStreamTelemetry) {
        self.posting_lists += telemetry.posting_lists as u64;
        self.batches += telemetry.batches as u64;
        self.bound_evaluations += telemetry.bound_evaluations as u64;
        self.zero_bound_batches += telemetry.zero_bound_batches as u64;
        self.batches_expanded += telemetry.batches_expanded as u64;
        self.posting_elements_visited += telemetry.posting_elements_visited as u64;
        self.nonzero_documents_scored += telemetry.nonzero_documents_scored as u64;
        self.points_emitted += telemetry.points_emitted as u64;
        self.max_pending_batches = self.max_pending_batches.max(telemetry.max_pending_batches);
        self.max_pending_points = self.max_pending_points.max(telemetry.max_pending_points);
        self.max_buffered_points = self.max_buffered_points.max(telemetry.max_buffered_points);
    }
}

#[derive(Debug, Serialize)]
struct PostingStreamExecutorSummary {
    latency: LatencySummary,
    telemetry: AggregatePostingStreamTelemetry,
}

#[derive(Debug, Default, Serialize)]
struct AggregateAdaptiveTelemetry {
    refills: u64,
    final_requested_limit: usize,
    results_materialized: u64,
    posting_elements: u64,
    posting_elements_visited: u64,
    posting_elements_skipped: u64,
    batches: u64,
    points_emitted: u64,
}

impl AggregateAdaptiveTelemetry {
    fn add(&mut self, telemetry: AdaptiveTopKStreamTelemetry) {
        self.refills += telemetry.refills as u64;
        self.final_requested_limit = self
            .final_requested_limit
            .max(telemetry.final_requested_limit);
        self.results_materialized += telemetry.results_materialized as u64;
        self.posting_elements += telemetry.posting_elements as u64;
        self.posting_elements_visited += telemetry.posting_elements_visited as u64;
        self.posting_elements_skipped += telemetry.posting_elements_skipped as u64;
        self.batches += telemetry.batches as u64;
        self.points_emitted += telemetry.points_emitted as u64;
    }
}

#[derive(Debug, Serialize)]
struct AdaptiveExecutorSummary {
    latency: LatencySummary,
    telemetry: AggregateAdaptiveTelemetry,
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
    block_prune_failure_limit: Option<usize>,
    block_batch_size: usize,
    endpoint_trend_router: bool,
    adaptive_initial_limit: usize,
    adaptive_growth_factor: usize,
    compressed_build_ns: u128,
    compressed_index_bytes: usize,
    ordered_top_k_mismatches: usize,
    posting_stream_ordered_top_k_mismatches: usize,
    adaptive_ordered_top_k_mismatches: usize,
    baseline: ExecutorSummary,
    posting_block_max: ExecutorSummary,
    posting_block_stream: PostingStreamExecutorSummary,
    adaptive_native_stream: AdaptiveExecutorSummary,
}

fn main() {
    let documents = env_usize("SPECTRA_DOCUMENTS", DEFAULT_DOCUMENTS);
    let queries = env_usize("SPECTRA_QUERIES", DEFAULT_QUERIES);
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let pattern = WeightPattern::from_env();
    let block_prune_failure_limit = env_optional_usize("SPECTRA_BLOCK_FAILURE_LIMIT");
    let block_batch_size = env_usize("SPECTRA_BLOCK_BATCH_SIZE", 10_000);
    let endpoint_trend_router = env_bool("SPECTRA_BLOCK_TREND_ROUTER", false);
    let adaptive_initial_limit = env_usize("SPECTRA_STREAM_INITIAL_LIMIT", top_k);
    let adaptive_growth_factor = env_usize("SPECTRA_STREAM_GROWTH", 2);
    assert!(documents > 0 && queries > 0 && top_k > 0);

    let ram = build_index(documents, pattern);
    let temp_dir = tempfile::tempdir().unwrap();
    let build_started = Instant::now();
    let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
        Cow::Owned(ram),
        temp_dir.path(),
    )
    .unwrap();
    let compressed_build_ns = build_started.elapsed().as_nanos();
    let compressed_index_bytes = index.total_sparse_vectors_size();
    let query_vectors = build_queries(queries);
    let pool = SearchScratchPool::new();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::new();

    let mut baseline_latencies = Vec::with_capacity(queries);
    let mut block_latencies = Vec::with_capacity(queries);
    let mut stream_latencies = Vec::with_capacity(queries);
    let mut adaptive_latencies = Vec::with_capacity(queries);
    let mut baseline_telemetry = AggregateTelemetry::default();
    let mut block_telemetry = AggregateTelemetry::default();
    let mut stream_telemetry = AggregatePostingStreamTelemetry::default();
    let mut adaptive_telemetry = AggregateAdaptiveTelemetry::default();
    let mut mismatches = 0;
    let mut stream_mismatches = 0;
    let mut adaptive_mismatches = 0;

    for (query_index, query) in query_vectors.into_iter().enumerate() {
        let adaptive_early = query_index.is_multiple_of(2).then(|| {
            run_adaptive_stream(
                &index,
                query.clone(),
                top_k,
                adaptive_initial_limit,
                adaptive_growth_factor,
                block_batch_size,
                endpoint_trend_router,
                &pool,
                &stopped,
                &hardware_counter,
            )
        });
        let (baseline, block, stream) = match query_index % 3 {
            0 => {
                let baseline = run(
                    &index,
                    query.clone(),
                    top_k,
                    false,
                    None,
                    10_000,
                    false,
                    &pool,
                    &stopped,
                    &hardware_counter,
                );
                let block = run(
                    &index,
                    query.clone(),
                    top_k,
                    true,
                    block_prune_failure_limit,
                    block_batch_size,
                    endpoint_trend_router,
                    &pool,
                    &stopped,
                    &hardware_counter,
                );
                let stream = run_stream(
                    &index,
                    query.clone(),
                    top_k,
                    block_batch_size,
                    &hardware_counter,
                );
                (baseline, block, stream)
            }
            1 => {
                let block = run(
                    &index,
                    query.clone(),
                    top_k,
                    true,
                    block_prune_failure_limit,
                    block_batch_size,
                    endpoint_trend_router,
                    &pool,
                    &stopped,
                    &hardware_counter,
                );
                let stream = run_stream(
                    &index,
                    query.clone(),
                    top_k,
                    block_batch_size,
                    &hardware_counter,
                );
                let baseline = run(
                    &index,
                    query.clone(),
                    top_k,
                    false,
                    None,
                    10_000,
                    false,
                    &pool,
                    &stopped,
                    &hardware_counter,
                );
                (baseline, block, stream)
            }
            _ => {
                let stream = run_stream(
                    &index,
                    query.clone(),
                    top_k,
                    block_batch_size,
                    &hardware_counter,
                );
                let baseline = run(
                    &index,
                    query.clone(),
                    top_k,
                    false,
                    None,
                    10_000,
                    false,
                    &pool,
                    &stopped,
                    &hardware_counter,
                );
                let block = run(
                    &index,
                    query.clone(),
                    top_k,
                    true,
                    block_prune_failure_limit,
                    block_batch_size,
                    endpoint_trend_router,
                    &pool,
                    &stopped,
                    &hardware_counter,
                );
                (baseline, block, stream)
            }
        };
        let adaptive = adaptive_early.unwrap_or_else(|| {
            run_adaptive_stream(
                &index,
                query,
                top_k,
                adaptive_initial_limit,
                adaptive_growth_factor,
                block_batch_size,
                endpoint_trend_router,
                &pool,
                &stopped,
                &hardware_counter,
            )
        });
        mismatches += usize::from(baseline.0 != block.0);
        stream_mismatches += usize::from(baseline.0 != stream.0);
        adaptive_mismatches += usize::from(baseline.0 != adaptive.0);
        baseline_latencies.push(baseline.1);
        block_latencies.push(block.1);
        stream_latencies.push(stream.1);
        adaptive_latencies.push(adaptive.1);
        baseline_telemetry.add(baseline.2);
        block_telemetry.add(block.2);
        stream_telemetry.add(stream.2);
        adaptive_telemetry.add(adaptive.2);
    }

    let result = ExperimentResult {
        schema_version: 1,
        experiment: "spectra-sparse-posting-block-max-v1",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        weight_pattern: pattern.name(),
        documents,
        queries,
        top_k,
        block_prune_failure_limit,
        block_batch_size,
        endpoint_trend_router,
        adaptive_initial_limit,
        adaptive_growth_factor,
        compressed_build_ns,
        compressed_index_bytes,
        ordered_top_k_mismatches: mismatches,
        posting_stream_ordered_top_k_mismatches: stream_mismatches,
        adaptive_ordered_top_k_mismatches: adaptive_mismatches,
        baseline: ExecutorSummary {
            latency: summarize(&mut baseline_latencies),
            telemetry: baseline_telemetry,
        },
        posting_block_max: ExecutorSummary {
            latency: summarize(&mut block_latencies),
            telemetry: block_telemetry,
        },
        posting_block_stream: PostingStreamExecutorSummary {
            latency: summarize(&mut stream_latencies),
            telemetry: stream_telemetry,
        },
        adaptive_native_stream: AdaptiveExecutorSummary {
            latency: summarize(&mut adaptive_latencies),
            telemetry: adaptive_telemetry,
        },
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(mismatches, 0);
    assert_eq!(stream_mismatches, 0);
    assert_eq!(adaptive_mismatches, 0);
}

fn build_index(
    documents: usize,
    pattern: WeightPattern,
) -> sparse::index::inverted_index::inverted_index_ram::InvertedIndexRam {
    let mut builder = InvertedIndexBuilder::new();
    for document in 0..documents {
        let cluster = 1 + document as u32 % CLUSTERS;
        let rare = 1 + CLUSTERS + document as u32 % RARE_TERMS;
        let position = pattern.position_weight(document, documents);
        builder.add(
            document as PointOffsetType,
            RemappedSparseVector {
                indices: vec![0, cluster, rare],
                values: vec![0.5 + 0.5 * position, 1.0 + position, 2.0 + position],
            },
        );
    }
    builder.build()
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

#[expect(clippy::too_many_arguments)]
fn run(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    query: RemappedSparseVector,
    top_k: usize,
    block_pruning: bool,
    block_prune_failure_limit: Option<usize>,
    batch_size: usize,
    endpoint_trend_router: bool,
    pool: &SearchScratchPool,
    stopped: &AtomicBool,
    hardware_counter: &HardwareCounterCell,
) -> (Vec<ScoredPointOffset>, Duration, SearchTelemetry) {
    let started = Instant::now();
    let mut scratch = pool.get();
    let mut context =
        SearchContext::new(query, top_k, index, &mut scratch, stopped, hardware_counter).unwrap();
    context.set_block_pruning(block_pruning);
    context.set_block_prune_failure_limit(block_prune_failure_limit);
    if endpoint_trend_router {
        context.apply_block_max_endpoint_trend_router();
    }
    if context.block_pruning_enabled() {
        context.set_batch_size(batch_size);
    }
    let points = context.search(&|_| true);
    (points, started.elapsed(), context.telemetry())
}

fn run_stream(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    query: RemappedSparseVector,
    top_k: usize,
    batch_size: usize,
    hardware_counter: &HardwareCounterCell,
) -> (
    Vec<ScoredPointOffset>,
    Duration,
    PostingBlockStreamTelemetry,
) {
    let started = Instant::now();
    let arena = sparse::SearchScratchArena::new_slow();
    let mut stream =
        PostingBlockStream::new(index, query, batch_size, &arena, hardware_counter).unwrap();
    let points = stream.by_ref().take(top_k).collect();
    (points, started.elapsed(), stream.telemetry())
}

#[expect(clippy::too_many_arguments)]
fn run_adaptive_stream(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    query: RemappedSparseVector,
    top_k: usize,
    initial_limit: usize,
    growth_factor: usize,
    batch_size: usize,
    endpoint_trend_router: bool,
    pool: &SearchScratchPool,
    stopped: &AtomicBool,
    hardware_counter: &HardwareCounterCell,
) -> (
    Vec<ScoredPointOffset>,
    Duration,
    AdaptiveTopKStreamTelemetry,
) {
    let started = Instant::now();
    let mut stream = AdaptiveTopKStream::new(
        index,
        query,
        initial_limit,
        growth_factor,
        true,
        batch_size,
        endpoint_trend_router,
        pool,
        stopped,
        hardware_counter,
    );
    let points = stream.by_ref().take(top_k).collect();
    (points, started.elapsed(), stream.telemetry())
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn env_optional_usize(name: &str) -> Option<usize> {
    env::var(name).ok().map(|value| {
        let value = value.parse().unwrap_or_else(|_| panic!("invalid {name}"));
        assert!(value > 0, "{name} must be positive");
        value
    })
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => panic!("invalid {name}"),
        })
        .unwrap_or(default)
}

fn summarize(latencies: &mut [Duration]) -> LatencySummary {
    latencies.sort_unstable();
    let percentile = |numerator: usize| latencies[(latencies.len() - 1) * numerator / 100];
    LatencySummary {
        p50_ns: percentile(50).as_nanos(),
        p95_ns: percentile(95).as_nanos(),
        p99_ns: percentile(99).as_nanos(),
        mean_ns: latencies.iter().map(Duration::as_nanos).sum::<u128>() / latencies.len() as u128,
    }
}
