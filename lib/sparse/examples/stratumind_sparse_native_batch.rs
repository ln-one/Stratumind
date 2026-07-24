// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Isolated comparison of Eager, point-oriented Native, and certified-batch
//! Native Sparse continuation over the same Qdrant compressed postings.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use fs_err::File;
use serde::{Deserialize, Serialize};
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::inverted_index::InvertedIndex;
use sparse::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::native_rank_stream::NativeSearchContextRankStream;
use sparse::index::posting_block_stream::{
    ExactSparseCursor, PostingBlockMaxCursor, PostingBlockMaxVariant, PostingBlockStreamTelemetry,
};
use sparse::index::search_context::SearchContext;
use sparse::{SearchScratchArena, SearchScratchPool};

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

#[derive(Debug, Clone, Serialize)]
struct LatencySummary {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Default, Serialize)]
struct Measurement {
    ordered_mismatches: usize,
    latency: Option<LatencySummary>,
    latency_ns_by_query: Vec<u128>,
    query_posting_elements: u64,
    posting_elements_visited: u64,
    bound_evaluations: u64,
    batches_expanded: u64,
    points_emitted: u64,
    max_buffered_points: usize,
    range_direct_batches: u64,
    score_buffer_slots: u64,
    score_buffer_touched: u64,
    max_score_buffer_slots: usize,
    max_touched_slots: usize,
    sorted_batches: u64,
    chunks_decoded: u64,
    max_score_buffer_capacity: usize,
    max_result_buffer_capacity: usize,
    max_pending_batch_capacity: usize,
    pauses: u64,
    posting_open_ns: u64,
    bound_plan_ns: u64,
    bound_heapify_ns: u64,
    range_score_ns: u64,
    score_scan_ns: u64,
    result_heapify_ns: u64,
    proof_ns: u64,
    delivery_ns: u64,
}

impl Measurement {
    fn record(
        &mut self,
        expected: &[ScoredPointOffset],
        points: &[ScoredPointOffset],
        elapsed: Duration,
        telemetry: PostingBlockStreamTelemetry,
    ) {
        self.ordered_mismatches += usize::from(points != expected);
        self.latency_ns_by_query.push(elapsed.as_nanos());
        self.query_posting_elements += telemetry.query_posting_elements as u64;
        self.posting_elements_visited += telemetry.posting_elements_visited as u64;
        self.bound_evaluations += telemetry.bound_evaluations as u64;
        self.batches_expanded += telemetry.batches_expanded as u64;
        self.points_emitted += telemetry.points_emitted as u64;
        self.max_buffered_points = self.max_buffered_points.max(telemetry.max_buffered_points);
        self.range_direct_batches += telemetry.range_direct_batches as u64;
        self.score_buffer_slots += telemetry.score_buffer_slots as u64;
        self.score_buffer_touched += telemetry.score_buffer_touched as u64;
        self.max_score_buffer_slots = self
            .max_score_buffer_slots
            .max(telemetry.max_score_buffer_slots);
        self.max_touched_slots = self.max_touched_slots.max(telemetry.max_touched_slots);
        self.sorted_batches += telemetry.sorted_batches as u64;
        self.chunks_decoded += telemetry.chunks_decoded as u64;
        self.max_score_buffer_capacity = self
            .max_score_buffer_capacity
            .max(telemetry.score_buffer_capacity);
        self.max_result_buffer_capacity = self
            .max_result_buffer_capacity
            .max(telemetry.result_buffer_capacity);
        self.max_pending_batch_capacity = self
            .max_pending_batch_capacity
            .max(telemetry.pending_batch_capacity);
        self.pauses += telemetry.pause_count as u64;
        self.posting_open_ns = self
            .posting_open_ns
            .saturating_add(telemetry.phase.posting_open_ns);
        self.bound_plan_ns = self
            .bound_plan_ns
            .saturating_add(telemetry.phase.bound_plan_ns);
        self.bound_heapify_ns = self
            .bound_heapify_ns
            .saturating_add(telemetry.phase.bound_heapify_ns);
        self.range_score_ns = self
            .range_score_ns
            .saturating_add(telemetry.phase.range_score_ns);
        self.score_scan_ns = self
            .score_scan_ns
            .saturating_add(telemetry.phase.score_scan_ns);
        self.result_heapify_ns = self
            .result_heapify_ns
            .saturating_add(telemetry.phase.result_heapify_ns);
        self.proof_ns = self.proof_ns.saturating_add(telemetry.phase.proof_ns);
        self.delivery_ns = self.delivery_ns.saturating_add(telemetry.phase.delivery_ns);
    }

    fn finish(&mut self) {
        self.latency = Some(summarize_nanos(&self.latency_ns_by_query));
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
    scan_span: usize,
    result_batch: usize,
    plan_profiles: BTreeMap<String, PhysicalProfile>,
    warmup_rounds: usize,
    measured_rounds: usize,
    index_build_ns: u128,
    query_profiles: Vec<QueryProfile>,
    qdrant_fixed_top_k: Measurement,
    posting_block_max_v1: Measurement,
    posting_block_max_native: Measurement,
    range_direct_dense: Measurement,
    range_direct_touched: Measurement,
    range_direct_sorted: Measurement,
    native_suffix_batch: Measurement,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PhysicalProfile {
    scan_span: usize,
    result_batch: usize,
}

#[derive(Debug, Serialize)]
struct QueryProfile {
    query_index: usize,
    query_nnz: usize,
    query_posting_elements: usize,
}

struct PlanResult {
    points: Vec<ScoredPointOffset>,
    elapsed: Duration,
    telemetry: PostingBlockStreamTelemetry,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", 256);
    let scan_span = env_usize("SPECTRA_SPARSE_SCAN_SPAN", 4_096);
    let result_batch = env_usize("SPECTRA_SPARSE_RESULT_BATCH", 32);
    let plan_profiles = BTreeMap::from([
        (
            "posting_block_max_v1".to_owned(),
            PhysicalProfile {
                scan_span: env_usize("SPECTRA_PBM_V1_SCAN_SPAN", scan_span),
                result_batch: env_usize("SPECTRA_PBM_V1_RESULT_BATCH", result_batch),
            },
        ),
        (
            "posting_block_max_native".to_owned(),
            PhysicalProfile {
                scan_span: env_usize("SPECTRA_PBM_NATIVE_SCAN_SPAN", scan_span),
                result_batch: env_usize("SPECTRA_PBM_NATIVE_RESULT_BATCH", result_batch),
            },
        ),
        (
            "range_direct_dense".to_owned(),
            PhysicalProfile {
                scan_span: env_usize("SPECTRA_PBM_DIRECT_DENSE_SCAN_SPAN", scan_span),
                result_batch: env_usize("SPECTRA_PBM_DIRECT_DENSE_RESULT_BATCH", result_batch),
            },
        ),
        (
            "range_direct_touched".to_owned(),
            PhysicalProfile {
                scan_span: env_usize("SPECTRA_PBM_TOUCHED_SCAN_SPAN", scan_span),
                result_batch: env_usize("SPECTRA_PBM_TOUCHED_RESULT_BATCH", result_batch),
            },
        ),
        (
            "range_direct_sorted".to_owned(),
            PhysicalProfile {
                scan_span: env_usize("SPECTRA_PBM_SORTED_SCAN_SPAN", scan_span),
                result_batch: env_usize("SPECTRA_PBM_SORTED_RESULT_BATCH", result_batch),
            },
        ),
    ]);
    let warmup_rounds = env_usize("SPECTRA_WARMUP_ROUNDS", 2);
    let measured_rounds = env_usize("SPECTRA_MEASURED_ROUNDS", 5).max(1);
    let query_limit = env::var("SPECTRA_QUERY_LIMIT")
        .ok()
        .map(|value| value.parse::<usize>().expect("invalid SPECTRA_QUERY_LIMIT"));
    assert!(top_k > 0 && scan_span > 0 && result_batch > 0);
    assert!(
        plan_profiles
            .values()
            .all(|profile| profile.scan_span > 0 && profile.result_batch > 0)
    );

    let corpus: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut queries: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Some(limit) = query_limit {
        queries.truncate(limit);
    }
    assert!(!corpus.is_empty() && !queries.is_empty());

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
    let build_started = Instant::now();
    let index_dir = tempfile::tempdir().unwrap();
    let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
        Cow::Owned(builder.build()),
        index_dir.path(),
    )
    .unwrap();
    let index_build_ns = build_started.elapsed().as_nanos();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::new();
    let query_profiles = queries
        .iter()
        .enumerate()
        .map(|(query_index, query)| QueryProfile {
            query_index,
            query_nnz: query.sparse_indices.len(),
            query_posting_elements: query
                .sparse_indices
                .iter()
                .map(|&term| {
                    index
                        .posting_list_len(term, &hardware_counter)
                        .unwrap_or_default()
                })
                .sum(),
        })
        .collect();

    let scratch_pool = SearchScratchPool::new();
    let mut qdrant_measurement = Measurement::default();
    let mut v1_measurement = Measurement::default();
    let mut native_pbm_measurement = Measurement::default();
    let mut direct_dense_measurement = Measurement::default();
    let mut direct_touched_measurement = Measurement::default();
    let mut direct_sorted_measurement = Measurement::default();
    let mut native_measurement = Measurement::default();
    for round in 0..warmup_rounds + measured_rounds {
        for (query_index, row) in queries.iter().enumerate() {
            let query = RemappedSparseVector {
                indices: row.sparse_indices.clone(),
                values: row.sparse_values.clone(),
            };
            let mut results = std::array::from_fn::<_, 7, _>(|_| None);
            let first = (round + query_index) % results.len();
            for offset in 0..results.len() {
                let condition = (first + offset) % results.len();
                results[condition] = Some(match condition {
                    0 => run_qdrant(
                        &index,
                        query.clone(),
                        top_k,
                        &scratch_pool,
                        &hardware_counter,
                        &stopped,
                    ),
                    1 => run_posting_block_max(
                        &index,
                        query.clone(),
                        top_k,
                        plan_profiles["posting_block_max_v1"].scan_span,
                        plan_profiles["posting_block_max_v1"].result_batch,
                        PostingBlockMaxVariant::V1,
                        &hardware_counter,
                        &stopped,
                    ),
                    2 => run_posting_block_max(
                        &index,
                        query.clone(),
                        top_k,
                        plan_profiles["posting_block_max_native"].scan_span,
                        plan_profiles["posting_block_max_native"].result_batch,
                        PostingBlockMaxVariant::CompressedMetadata,
                        &hardware_counter,
                        &stopped,
                    ),
                    3 => run_posting_block_max(
                        &index,
                        query.clone(),
                        top_k,
                        plan_profiles["range_direct_dense"].scan_span,
                        plan_profiles["range_direct_dense"].result_batch,
                        PostingBlockMaxVariant::RangeDirectDense,
                        &hardware_counter,
                        &stopped,
                    ),
                    4 => run_posting_block_max(
                        &index,
                        query.clone(),
                        top_k,
                        plan_profiles["range_direct_touched"].scan_span,
                        plan_profiles["range_direct_touched"].result_batch,
                        PostingBlockMaxVariant::RangeDirectTouched,
                        &hardware_counter,
                        &stopped,
                    ),
                    5 => run_posting_block_max(
                        &index,
                        query.clone(),
                        top_k,
                        plan_profiles["range_direct_sorted"].scan_span,
                        plan_profiles["range_direct_sorted"].result_batch,
                        PostingBlockMaxVariant::RangeDirectSorted,
                        &hardware_counter,
                        &stopped,
                    ),
                    6 => run_native_batch(
                        &index,
                        query.clone(),
                        top_k,
                        scan_span,
                        result_batch,
                        &hardware_counter,
                        &stopped,
                    ),
                    _ => unreachable!(),
                });
            }
            let [
                qdrant,
                v1,
                native_pbm,
                direct_dense,
                direct_touched,
                direct_sorted,
                native,
            ] = results.map(Option::unwrap);
            assert_eq!(v1.points, native_pbm.points);
            assert_eq!(v1.points, direct_dense.points);
            assert_eq!(v1.points, direct_touched.points);
            assert_eq!(v1.points, direct_sorted.points);
            assert_eq!(v1.points, native.points);
            if round >= warmup_rounds {
                qdrant_measurement.record(
                    &v1.points,
                    &qdrant.points,
                    qdrant.elapsed,
                    qdrant.telemetry,
                );
                v1_measurement.record(&v1.points, &v1.points, v1.elapsed, v1.telemetry);
                native_pbm_measurement.record(
                    &v1.points,
                    &native_pbm.points,
                    native_pbm.elapsed,
                    native_pbm.telemetry,
                );
                direct_dense_measurement.record(
                    &v1.points,
                    &direct_dense.points,
                    direct_dense.elapsed,
                    direct_dense.telemetry,
                );
                direct_touched_measurement.record(
                    &v1.points,
                    &direct_touched.points,
                    direct_touched.elapsed,
                    direct_touched.telemetry,
                );
                direct_sorted_measurement.record(
                    &v1.points,
                    &direct_sorted.points,
                    direct_sorted.elapsed,
                    direct_sorted.telemetry,
                );
                native_measurement.record(
                    &v1.points,
                    &native.points,
                    native.elapsed,
                    native.telemetry,
                );
            }
        }
    }
    qdrant_measurement.finish();
    v1_measurement.finish();
    native_pbm_measurement.finish();
    direct_dense_measurement.finish();
    direct_touched_measurement.finish();
    direct_sorted_measurement.finish();
    native_measurement.finish();

    let result = ResultRow {
        schema_version: 3,
        experiment: "stratumind-posting-block-max-native-v1",
        dataset_dir: dataset_dir.display().to_string(),
        documents: corpus.len(),
        queries: queries.len(),
        top_k,
        scan_span,
        result_batch,
        plan_profiles,
        warmup_rounds,
        measured_rounds,
        index_build_ns,
        query_profiles,
        qdrant_fixed_top_k: qdrant_measurement,
        posting_block_max_v1: v1_measurement,
        posting_block_max_native: native_pbm_measurement,
        range_direct_dense: direct_dense_measurement,
        range_direct_touched: direct_touched_measurement,
        range_direct_sorted: direct_sorted_measurement,
        native_suffix_batch: native_measurement,
    };
    let output = PathBuf::from(env::var("SPECTRA_OUTPUT").unwrap_or_else(|_| {
        "/private/tmp/stratumind-posting-block-max-v2-tournament.json".to_owned()
    }));
    fs_err::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}

fn run_qdrant(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    query: RemappedSparseVector,
    top_k: usize,
    scratch_pool: &SearchScratchPool,
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
) -> PlanResult {
    let query_terms = query.indices.len();
    let mut scratch = scratch_pool.get();
    let started = Instant::now();
    let mut context =
        SearchContext::new(query, top_k, index, &mut scratch, stopped, hardware_counter).unwrap();
    let mut points = context.search(&|_| true);
    points.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.idx.cmp(&right.idx))
    });
    let search = context.telemetry();
    PlanResult {
        points,
        elapsed: started.elapsed(),
        telemetry: PostingBlockStreamTelemetry {
            query_terms,
            posting_lists: search.posting_lists,
            query_posting_elements: search.posting_elements,
            posting_elements_visited: search.posting_elements_visited,
            batches: search.batch_count,
            batches_expanded: search.batch_count,
            ..Default::default()
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn run_posting_block_max(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    query: RemappedSparseVector,
    top_k: usize,
    scan_span: usize,
    result_batch: usize,
    variant: PostingBlockMaxVariant,
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
) -> PlanResult {
    let arena = SearchScratchArena::new_slow();
    let started = Instant::now();
    let mut cursor = PostingBlockMaxCursor::new_with_variant(
        index,
        query,
        scan_span,
        variant,
        &arena,
        hardware_counter,
    )
    .unwrap();
    let mut points = Vec::with_capacity(top_k);
    while points.len() < top_k {
        let remaining = top_k - points.len();
        let batch = cursor
            .next_batch(result_batch.min(remaining), stopped)
            .unwrap();
        if batch.is_empty() {
            break;
        }
        points.extend(batch);
    }
    PlanResult {
        points,
        elapsed: started.elapsed(),
        telemetry: cursor.telemetry(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_native_batch(
    index: &InvertedIndexCompressedImmutableRam<f32>,
    query: RemappedSparseVector,
    top_k: usize,
    scan_span: usize,
    result_batch: usize,
    hardware_counter: &HardwareCounterCell,
    stopped: &AtomicBool,
) -> PlanResult {
    let arena = SearchScratchArena::new_slow();
    let started = Instant::now();
    let mut cursor =
        NativeSearchContextRankStream::new(query, scan_span, index, &arena, hardware_counter)
            .unwrap();
    let mut points = Vec::with_capacity(top_k);
    while points.len() < top_k {
        let remaining = top_k - points.len();
        let batch = cursor
            .next_batch(result_batch.min(remaining), stopped)
            .unwrap();
        if batch.is_empty() {
            break;
        }
        points.extend(batch);
    }
    PlanResult {
        points,
        elapsed: started.elapsed(),
        telemetry: cursor.telemetry(),
    }
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

fn summarize_nanos(values: &[u128]) -> LatencySummary {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    LatencySummary {
        p50_ns: percentile(&sorted, 50),
        p95_ns: percentile(&sorted, 95),
        p99_ns: percentile(&sorted, 99),
        mean_ns: sorted.iter().sum::<u128>() / sorted.len() as u128,
    }
}

fn percentile(values: &[u128], percentile: usize) -> u128 {
    values[(values.len() - 1) * percentile / 100]
}
