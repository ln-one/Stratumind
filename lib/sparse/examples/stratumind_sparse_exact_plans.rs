// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Matched exact Sparse-plan comparison on a frozen vector snapshot.

use std::borrow::Cow;
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use fs_err::File;
use serde::{Deserialize, Serialize};
use sparse::SearchScratchPool;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::inverted_index::InvertedIndex;
use sparse::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::posting_block_stream::{PostingBlockStream, PostingBlockStreamTelemetry};
use sparse::index::search_context::{SearchContext, SearchTelemetry};

#[derive(Debug, Deserialize)]
struct CorpusRow {
    ordinal: PointOffsetType,
    sparse_indices: Vec<u32>,
    sparse_values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct QueryRow {
    sparse_indices: Vec<u32>,
    sparse_values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct LatencySummary {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Default, Serialize)]
struct QdrantTelemetry {
    posting_lists: u64,
    posting_elements: u64,
    posting_elements_visited: u64,
    posting_elements_skipped: u64,
    block_prune_attempts: u64,
    block_prune_successes: u64,
}

impl QdrantTelemetry {
    fn add(&mut self, value: SearchTelemetry) {
        self.posting_lists += value.posting_lists as u64;
        self.posting_elements += value.posting_elements as u64;
        self.posting_elements_visited += value.posting_elements_visited as u64;
        self.posting_elements_skipped += value.posting_elements_skipped as u64;
        self.block_prune_attempts += value.block_prune_attempts as u64;
        self.block_prune_successes += value.block_prune_successes as u64;
    }
}

#[derive(Debug, Default, Serialize)]
struct StreamTelemetry {
    posting_lists: u64,
    batches: u64,
    bound_evaluations: u64,
    batches_expanded: u64,
    posting_elements_visited: u64,
    nonzero_documents_scored: u64,
    points_emitted: u64,
    max_pending_batches: usize,
    max_buffered_points: usize,
}

impl StreamTelemetry {
    fn add(&mut self, value: PostingBlockStreamTelemetry) {
        self.posting_lists += value.posting_lists as u64;
        self.batches += value.batches as u64;
        self.bound_evaluations += value.bound_evaluations as u64;
        self.batches_expanded += value.batches_expanded as u64;
        self.posting_elements_visited += value.posting_elements_visited as u64;
        self.nonzero_documents_scored += value.nonzero_documents_scored as u64;
        self.points_emitted += value.points_emitted as u64;
        self.max_pending_batches = self.max_pending_batches.max(value.max_pending_batches);
        self.max_buffered_points = self.max_buffered_points.max(value.max_buffered_points);
    }
}

#[derive(Debug, Serialize)]
struct ResultRow {
    schema_version: usize,
    experiment: &'static str,
    dataset_dir: String,
    documents: usize,
    queries: usize,
    top_k: usize,
    batch_size: usize,
    compressed_build_ns: u128,
    compressed_index_bytes: usize,
    accepted_prefixes: usize,
    tied_boundary_fallbacks: usize,
    ordered_top_k_mismatches: usize,
    qdrant_latency: LatencySummary,
    stream_latency: LatencySummary,
    qdrant_latency_ns_by_query: Vec<u128>,
    stream_latency_ns_by_query: Vec<u128>,
    qdrant_telemetry: QdrantTelemetry,
    stream_telemetry: StreamTelemetry,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", 20);
    let batch_size = env_usize("SPECTRA_BLOCK_BATCH_SIZE", 4096);
    let query_limit = env::var("SPECTRA_QUERY_LIMIT")
        .ok()
        .map(|value| value.parse::<usize>().expect("invalid SPECTRA_QUERY_LIMIT"));
    assert!(top_k > 0 && batch_size > 0);

    let corpus: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut queries: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Some(limit) = query_limit {
        queries.truncate(limit);
    }
    assert!(!corpus.is_empty() && !queries.is_empty() && top_k <= corpus.len());

    let mut builder = InvertedIndexBuilder::new();
    for (position, row) in corpus.iter().enumerate() {
        assert_eq!(row.ordinal as usize, position);
        builder.add(
            row.ordinal,
            RemappedSparseVector {
                indices: row.sparse_indices.clone(),
                values: row.sparse_values.clone(),
            },
        );
    }
    let index_dir = tempfile::Builder::new()
        .prefix("stratumind-sparse-exact-plans")
        .tempdir()
        .unwrap();
    let build_started = Instant::now();
    let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
        Cow::Owned(builder.build()),
        index_dir.path(),
    )
    .unwrap();
    let compressed_build_ns = build_started.elapsed().as_nanos();
    let compressed_index_bytes = index.total_sparse_vectors_size();

    let pool = SearchScratchPool::new();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::new();
    let mut qdrant_latencies = Vec::with_capacity(queries.len());
    let mut stream_latencies = Vec::with_capacity(queries.len());
    let mut qdrant_telemetry = QdrantTelemetry::default();
    let mut stream_telemetry = StreamTelemetry::default();
    let mut accepted_prefixes = 0;
    let mut tied_boundary_fallbacks = 0;
    let mut mismatches = 0;

    for (query_index, row) in queries.into_iter().enumerate() {
        let query = RemappedSparseVector {
            indices: row.sparse_indices,
            values: row.sparse_values,
        };
        let (qdrant, qdrant_elapsed, qdrant_progress, stream, stream_elapsed, stream_progress) =
            if query_index.is_multiple_of(2) {
                let qdrant = run_qdrant(
                    &index,
                    query.clone(),
                    top_k,
                    &pool,
                    &stopped,
                    &hardware_counter,
                );
                let stream = run_stream(&index, query, top_k, batch_size, &hardware_counter);
                (qdrant.0, qdrant.1, qdrant.2, stream.0, stream.1, stream.2)
            } else {
                let stream =
                    run_stream(&index, query.clone(), top_k, batch_size, &hardware_counter);
                let qdrant = run_qdrant(&index, query, top_k, &pool, &stopped, &hardware_counter);
                (qdrant.0, qdrant.1, qdrant.2, stream.0, stream.1, stream.2)
            };
        match certify_prefix(qdrant, top_k) {
            Some(prefix) => {
                accepted_prefixes += 1;
                mismatches += usize::from(prefix != stream);
            }
            None => tied_boundary_fallbacks += 1,
        }
        qdrant_latencies.push(qdrant_elapsed);
        stream_latencies.push(stream_elapsed);
        qdrant_telemetry.add(qdrant_progress);
        stream_telemetry.add(stream_progress);
    }

    let result = ResultRow {
        schema_version: 1,
        experiment: "stratumind-sparse-exact-plans-v1",
        dataset_dir: dataset_dir.display().to_string(),
        documents: corpus.len(),
        queries: qdrant_latencies.len(),
        top_k,
        batch_size,
        compressed_build_ns,
        compressed_index_bytes,
        accepted_prefixes,
        tied_boundary_fallbacks,
        ordered_top_k_mismatches: mismatches,
        qdrant_latency: summarize(&qdrant_latencies),
        stream_latency: summarize(&stream_latencies),
        qdrant_latency_ns_by_query: nanos(&qdrant_latencies),
        stream_latency_ns_by_query: nanos(&stream_latencies),
        qdrant_telemetry,
        stream_telemetry,
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(mismatches, 0);
}

fn run_qdrant(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    query: RemappedSparseVector,
    top_k: usize,
    pool: &SearchScratchPool,
    stopped: &AtomicBool,
    hardware_counter: &HardwareCounterCell,
) -> (Vec<ScoredPointOffset>, Duration, SearchTelemetry) {
    let started = Instant::now();
    let mut scratch = pool.get();
    let mut context = SearchContext::new(
        query,
        top_k.saturating_add(1),
        index,
        &mut scratch,
        stopped,
        hardware_counter,
    )
    .unwrap();
    let points = context.search(&|_| true);
    (points, started.elapsed(), context.telemetry())
}

fn certify_prefix(
    mut points: Vec<ScoredPointOffset>,
    top_k: usize,
) -> Option<Vec<ScoredPointOffset>> {
    points.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.idx.cmp(&right.idx))
    });
    if points.len() <= top_k {
        return Some(points);
    }
    if points[top_k - 1].score <= points[top_k].score {
        return None;
    }
    points.truncate(top_k);
    Some(points)
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
    let mut stream = PostingBlockStream::new(index, query, batch_size, &arena, hardware_counter)
        .expect("native Sparse stream must open");
    let points = stream.by_ref().take(top_k).collect();
    (points, started.elapsed(), stream.telemetry())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    BufReader::new(File::open(path).unwrap())
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect()
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}

fn nanos(values: &[Duration]) -> Vec<u128> {
    values.iter().map(Duration::as_nanos).collect()
}

fn summarize(values: &[Duration]) -> LatencySummary {
    let mut values = nanos(values);
    values.sort_unstable();
    LatencySummary {
        p50_ns: percentile(&values, 50),
        p95_ns: percentile(&values, 95),
        p99_ns: percentile(&values, 99),
        mean_ns: values.iter().sum::<u128>() / values.len() as u128,
    }
}

fn percentile(values: &[u128], percentile: usize) -> u128 {
    values[(values.len() - 1) * percentile / 100]
}
