// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use serde::{Deserialize, Serialize};
use sparse::SearchScratchPool;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::block_max::{BlockMaxIndex, BlockMaxTelemetry, SparseDocument};
use sparse::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::search_context::{SearchContext, SearchTelemetry};

const DEFAULT_TOP_K: usize = 100;
const DEFAULT_BLOCK_SIZE: usize = 128;

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
    throughput_queries_per_second: f64,
}

#[derive(Debug, Default, Serialize)]
struct NativeTelemetry {
    posting_lists: u64,
    posting_elements: u64,
    posting_elements_visited: u64,
    posting_elements_skipped: u64,
    prune_attempts: u64,
    prune_successes: u64,
}

impl NativeTelemetry {
    fn add(&mut self, telemetry: SearchTelemetry) {
        self.posting_lists += telemetry.posting_lists as u64;
        self.posting_elements += telemetry.posting_elements as u64;
        self.posting_elements_visited += telemetry.posting_elements_visited as u64;
        self.posting_elements_skipped += telemetry.posting_elements_skipped as u64;
        self.prune_attempts += telemetry.prune_attempts as u64;
        self.prune_successes += telemetry.prune_successes as u64;
    }
}

#[derive(Debug, Default, Serialize)]
struct BlockTelemetry {
    bound_evaluations: u64,
    zero_bound_blocks: u64,
    blocks_expanded: u64,
    documents_evaluated: u64,
    points_emitted: u64,
}

impl BlockTelemetry {
    fn add(&mut self, telemetry: BlockMaxTelemetry) {
        self.bound_evaluations += telemetry.bound_evaluations as u64;
        self.zero_bound_blocks += telemetry.zero_bound_blocks as u64;
        self.blocks_expanded += telemetry.blocks_expanded as u64;
        self.documents_evaluated += telemetry.documents_evaluated as u64;
        self.points_emitted += telemetry.points_emitted as u64;
    }
}

#[derive(Debug, Serialize)]
struct ExecutorSummary<T> {
    latency: LatencySummary,
    telemetry: T,
}

#[derive(Debug, Serialize)]
struct SparseSnapshotResult {
    schema_version: usize,
    experiment: &'static str,
    qdrant_base_commit: &'static str,
    build_profile: &'static str,
    dataset_dir: String,
    documents: usize,
    queries: usize,
    top_k: usize,
    block_size: usize,
    native_posting_pruning: bool,
    native_build_ns: u128,
    block_max_build_ns: u128,
    native_posting_lists: usize,
    native_searchable_bytes: usize,
    block_max_blocks: usize,
    block_max_envelope_nonzeros: usize,
    block_max_searchable_bytes_estimate: usize,
    native_parity_mismatches: usize,
    native_ordered_identity_mismatches: usize,
    native_identity_set_mismatches: usize,
    native_max_absolute_score_delta: f32,
    native_first_mismatch: Option<ParityMismatch>,
    native_first_identity_set_mismatch: Option<ParityMismatch>,
    block_max_parity_mismatches: usize,
    native: ExecutorSummary<NativeTelemetry>,
    block_max: ExecutorSummary<BlockTelemetry>,
    exhaustive: ExecutorSummary<NativeTelemetry>,
}

#[derive(Debug, Serialize)]
struct ParityPoint {
    id: PointOffsetType,
    score: f32,
}

#[derive(Debug, Serialize)]
struct ParityMismatch {
    query_index: usize,
    native: Vec<ParityPoint>,
    exhaustive: Vec<ParityPoint>,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let block_size = env_usize("SPECTRA_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    let native_posting_pruning = env_bool("SPECTRA_NATIVE_POSTING_PRUNING", true);
    assert!(top_k > 0);
    assert!(block_size > 0);

    let corpus: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut query_rows: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Ok(limit) = env::var("SPECTRA_QUERY_LIMIT") {
        query_rows.truncate(limit.parse().expect("invalid SPECTRA_QUERY_LIMIT"));
    }
    assert!(!corpus.is_empty());
    assert!(!query_rows.is_empty());
    assert!(top_k <= corpus.len());

    let documents: Vec<_> = corpus
        .into_iter()
        .enumerate()
        .map(|(position, row)| {
            assert_eq!(
                row.ordinal as usize, position,
                "ordinals must be contiguous"
            );
            SparseDocument {
                id: row.ordinal,
                vector: RemappedSparseVector {
                    indices: row.sparse_indices,
                    values: row.sparse_values,
                },
            }
        })
        .collect();
    let document_sparse_nonzeros: usize =
        documents.iter().map(|document| document.vector.len()).sum();
    let native_build_started = Instant::now();
    let native_index = InvertedIndexBuilder::build_from_iterator(
        documents
            .iter()
            .map(|document| (document.id, document.vector.clone())),
    );
    let native_build_ns = native_build_started.elapsed().as_nanos();
    let block_build_started = Instant::now();
    let block_index = BlockMaxIndex::build(documents, block_size).unwrap();
    let block_max_build_ns = block_build_started.elapsed().as_nanos();
    let document_ids: Vec<_> = (0..block_index.document_count() as PointOffsetType).collect();
    let queries: Vec<_> = query_rows
        .into_iter()
        .map(|row| RemappedSparseVector {
            indices: row.sparse_indices,
            values: row.sparse_values,
        })
        .collect();

    let pool = SearchScratchPool::new();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::new();
    let mut native_latencies = Vec::with_capacity(queries.len());
    let mut block_latencies = Vec::with_capacity(queries.len());
    let mut exhaustive_latencies = Vec::with_capacity(queries.len());
    let mut native_telemetry = NativeTelemetry::default();
    let mut block_telemetry = BlockTelemetry::default();
    let mut exhaustive_telemetry = NativeTelemetry::default();
    let mut native_parity_mismatches = 0;
    let mut native_ordered_identity_mismatches = 0;
    let mut native_identity_set_mismatches = 0;
    let mut native_max_absolute_score_delta = 0.0_f32;
    let mut native_first_mismatch = None;
    let mut native_first_identity_set_mismatch = None;
    let mut block_max_parity_mismatches = 0;

    for (query_index, query) in queries.into_iter().enumerate() {
        let (native, native_elapsed, native_progress) = run_native(
            &native_index,
            query.clone(),
            top_k,
            &pool,
            &stopped,
            &hardware_counter,
            native_posting_pruning,
        );
        let (exhaustive, exhaustive_elapsed, exhaustive_progress) = run_exhaustive(
            &native_index,
            query.clone(),
            top_k,
            &document_ids,
            &pool,
            &stopped,
            &hardware_counter,
        );
        let (block, block_elapsed, block_progress) = run_block_max(&block_index, query, top_k);

        let native_mismatch = native != exhaustive;
        native_parity_mismatches += usize::from(native_mismatch);
        native_ordered_identity_mismatches += usize::from(
            native
                .iter()
                .map(|point| point.idx)
                .ne(exhaustive.iter().map(|point| point.idx)),
        );
        for (native_point, exhaustive_point) in native.iter().zip(&exhaustive) {
            if native_point.idx == exhaustive_point.idx {
                native_max_absolute_score_delta = native_max_absolute_score_delta
                    .max((native_point.score - exhaustive_point.score).abs());
            }
        }
        if native_mismatch {
            let mut native_ids = native.iter().map(|point| point.idx).collect::<Vec<_>>();
            let mut exhaustive_ids = exhaustive.iter().map(|point| point.idx).collect::<Vec<_>>();
            native_ids.sort_unstable();
            exhaustive_ids.sort_unstable();
            let identity_set_mismatch = native_ids != exhaustive_ids;
            native_identity_set_mismatches += usize::from(identity_set_mismatch);
            if identity_set_mismatch {
                native_first_identity_set_mismatch.get_or_insert_with(|| ParityMismatch {
                    query_index,
                    native: parity_points(&native),
                    exhaustive: parity_points(&exhaustive),
                });
            }
            native_first_mismatch.get_or_insert_with(|| ParityMismatch {
                query_index,
                native: parity_points(&native),
                exhaustive: parity_points(&exhaustive),
            });
        }
        block_max_parity_mismatches += usize::from(block != exhaustive);
        native_latencies.push(native_elapsed);
        block_latencies.push(block_elapsed);
        exhaustive_latencies.push(exhaustive_elapsed);
        native_telemetry.add(native_progress);
        block_telemetry.add(block_progress);
        exhaustive_telemetry.add(exhaustive_progress);
    }

    let query_count = native_latencies.len();
    let result = SparseSnapshotResult {
        schema_version: 1,
        experiment: "spectra-sparse-native-snapshot-v1",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        dataset_dir: dataset_dir.display().to_string(),
        documents: block_index.document_count(),
        queries: query_count,
        top_k,
        block_size,
        native_posting_pruning,
        native_build_ns,
        block_max_build_ns,
        native_posting_lists: native_index.postings.len(),
        native_searchable_bytes: native_index.total_posting_elements_size(),
        block_max_blocks: block_index.block_count(),
        block_max_envelope_nonzeros: block_index.envelope_nonzero_count(),
        block_max_searchable_bytes_estimate: (document_sparse_nonzeros
            + block_index.envelope_nonzero_count())
            * (size_of::<u32>() + size_of::<f32>()),
        native_parity_mismatches,
        native_ordered_identity_mismatches,
        native_identity_set_mismatches,
        native_max_absolute_score_delta,
        native_first_mismatch,
        native_first_identity_set_mismatch,
        block_max_parity_mismatches,
        native: ExecutorSummary {
            latency: summarize_latency(&mut native_latencies),
            telemetry: native_telemetry,
        },
        block_max: ExecutorSummary {
            latency: summarize_latency(&mut block_latencies),
            telemetry: block_telemetry,
        },
        exhaustive: ExecutorSummary {
            latency: summarize_latency(&mut exhaustive_latencies),
            telemetry: exhaustive_telemetry,
        },
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(native_ordered_identity_mismatches, 0);
    assert_eq!(block_max_parity_mismatches, 0);
}

fn parity_points(points: &[ScoredPointOffset]) -> Vec<ParityPoint> {
    points
        .iter()
        .map(|point| ParityPoint {
            id: point.idx,
            score: point.score,
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
    posting_pruning: bool,
) -> (Vec<ScoredPointOffset>, Duration, SearchTelemetry) {
    let started = Instant::now();
    let mut scratch = pool.get();
    let mut context =
        SearchContext::new(query, top_k, index, &mut scratch, stopped, hardware_counter).unwrap();
    context.set_posting_pruning(posting_pruning);
    let result = context.search(&|_| true);
    (result, started.elapsed(), context.telemetry())
}

fn run_exhaustive(
    index: &InvertedIndexRam,
    query: RemappedSparseVector,
    top_k: usize,
    document_ids: &[PointOffsetType],
    pool: &SearchScratchPool,
    stopped: &AtomicBool,
    hardware_counter: &HardwareCounterCell,
) -> (Vec<ScoredPointOffset>, Duration, SearchTelemetry) {
    let started = Instant::now();
    let mut scratch = pool.get();
    let mut context =
        SearchContext::new(query, top_k, index, &mut scratch, stopped, hardware_counter).unwrap();
    let result = context.plain_search(document_ids);
    (result, started.elapsed(), context.telemetry())
}

fn run_block_max(
    index: &BlockMaxIndex,
    query: RemappedSparseVector,
    top_k: usize,
) -> (Vec<ScoredPointOffset>, Duration, BlockMaxTelemetry) {
    let started = Instant::now();
    let mut stream = index.stream(query).unwrap();
    let result = stream.by_ref().take(top_k).collect();
    (result, started.elapsed(), stream.telemetry())
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
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

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    BufReader::new(File::open(path).unwrap())
        .lines()
        .enumerate()
        .filter_map(|(position, line)| {
            let line = line.unwrap();
            (!line.trim().is_empty()).then(|| {
                serde_json::from_str(&line).unwrap_or_else(|error| {
                    panic!("invalid {}:{}: {error}", path.display(), position + 1)
                })
            })
        })
        .collect()
}

fn summarize_latency(durations: &mut [Duration]) -> LatencySummary {
    durations.sort_unstable();
    let total_ns: u128 = durations.iter().map(Duration::as_nanos).sum();
    LatencySummary {
        p50_ns: percentile(durations, 50).as_nanos(),
        p95_ns: percentile(durations, 95).as_nanos(),
        p99_ns: percentile(durations, 99).as_nanos(),
        mean_ns: total_ns / durations.len() as u128,
        throughput_queries_per_second: durations.len() as f64 / (total_ns as f64 / 1_000_000_000.0),
    }
}

fn percentile(durations: &[Duration], percentile: usize) -> Duration {
    let index = (durations.len() * percentile)
        .div_ceil(100)
        .saturating_sub(1);
    durations[index]
}
