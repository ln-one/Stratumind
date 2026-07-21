// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ordered_float::OrderedFloat;
use segment::index::dense_ball::{DenseDocument, DenseScoredPoint};
use segment::index::dense_quantized::{DenseQuantizedIndex, DenseQuantizedTelemetry};
use segment::index::dense_threshold::{DenseThresholdIndex, DenseThresholdTelemetry};
use serde::{Deserialize, Serialize};

const DEFAULT_TOP_K: usize = 20;

#[derive(Debug, Deserialize)]
struct CorpusRow {
    ordinal: u32,
    dense: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct QueryRow {
    dense: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct LatencySummary {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Serialize)]
struct ResultRow {
    schema_version: usize,
    experiment: &'static str,
    dataset_dir: String,
    documents: usize,
    queries: usize,
    dimension: usize,
    top_k: usize,
    original_vector_bytes: usize,
    threshold_build_ns: u128,
    threshold_auxiliary_bytes: usize,
    quantized_build_ns: u128,
    quantized_encoded_bytes: usize,
    threshold_parity_mismatches: usize,
    quantized_parity_mismatches: usize,
    threshold_sequential_accesses: u64,
    threshold_duplicate_accesses: u64,
    threshold_exact_scores: u64,
    threshold_exact_score_ratio: f64,
    threshold_accesses_per_exact_score: f64,
    quantized_scores: u64,
    quantized_exact_scores: u64,
    quantized_exact_score_ratio: f64,
    threshold_latency_ns_by_query: Vec<u128>,
    quantized_latency_ns_by_query: Vec<u128>,
    exhaustive_latency_ns_by_query: Vec<u128>,
    threshold_latency: LatencySummary,
    quantized_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let corpus_rows: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut query_rows: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Ok(limit) = env::var("SPECTRA_QUERY_LIMIT") {
        query_rows.truncate(limit.parse().expect("invalid SPECTRA_QUERY_LIMIT"));
    }
    assert!(!corpus_rows.is_empty());
    assert!(!query_rows.is_empty());
    let dimension = corpus_rows[0].dense.len();
    let documents: Vec<_> = corpus_rows
        .into_iter()
        .map(|row| DenseDocument {
            id: row.ordinal,
            vector: row.dense,
        })
        .collect();

    let threshold_build_started = Instant::now();
    let threshold = DenseThresholdIndex::build(documents.clone()).unwrap();
    let threshold_build_ns = threshold_build_started.elapsed().as_nanos();
    let quantized_build_started = Instant::now();
    let quantized = DenseQuantizedIndex::build(documents.clone()).unwrap();
    let quantized_build_ns = quantized_build_started.elapsed().as_nanos();

    let mut threshold_latencies = Vec::with_capacity(query_rows.len());
    let mut quantized_latencies = Vec::with_capacity(query_rows.len());
    let mut exhaustive_latencies = Vec::with_capacity(query_rows.len());
    let mut threshold_parity_mismatches = 0usize;
    let mut quantized_parity_mismatches = 0usize;
    let mut threshold_telemetry = DenseThresholdTelemetry::default();
    let mut quantized_telemetry = DenseQuantizedTelemetry::default();

    for query in query_rows {
        let exhaustive_started = Instant::now();
        let expected = exhaustive(&documents, &query.dense, top_k);
        exhaustive_latencies.push(exhaustive_started.elapsed());

        let threshold_started = Instant::now();
        let mut threshold_stream = threshold.stream(&query.dense).unwrap();
        let threshold_actual: Vec<_> = threshold_stream.by_ref().take(top_k).collect();
        threshold_latencies.push(threshold_started.elapsed());
        add_threshold_telemetry(&mut threshold_telemetry, threshold_stream.telemetry());
        threshold_parity_mismatches += usize::from(threshold_actual != expected);

        let quantized_started = Instant::now();
        let mut quantized_stream = quantized.stream(&query.dense).unwrap();
        let quantized_actual: Vec<_> = quantized_stream.by_ref().take(top_k).collect();
        quantized_latencies.push(quantized_started.elapsed());
        add_quantized_telemetry(&mut quantized_telemetry, quantized_stream.telemetry());
        quantized_parity_mismatches += usize::from(quantized_actual != expected);
    }

    let corpus_query_pairs = documents.len() * threshold_latencies.len();
    let result = ResultRow {
        schema_version: 1,
        experiment: "spectra-dense-threshold-v1",
        dataset_dir: dataset_dir.display().to_string(),
        documents: documents.len(),
        queries: threshold_latencies.len(),
        dimension,
        top_k,
        original_vector_bytes: documents.len() * dimension * size_of::<f32>(),
        threshold_build_ns,
        threshold_auxiliary_bytes: threshold.auxiliary_bytes(),
        quantized_build_ns,
        quantized_encoded_bytes: quantized.encoded_bytes(),
        threshold_parity_mismatches,
        quantized_parity_mismatches,
        threshold_sequential_accesses: threshold_telemetry.sequential_accesses as u64,
        threshold_duplicate_accesses: threshold_telemetry.duplicate_accesses as u64,
        threshold_exact_scores: threshold_telemetry.exact_scores as u64,
        threshold_exact_score_ratio: threshold_telemetry.exact_scores as f64
            / corpus_query_pairs as f64,
        threshold_accesses_per_exact_score: threshold_telemetry.sequential_accesses as f64
            / threshold_telemetry.exact_scores as f64,
        quantized_scores: quantized_telemetry.quantized_scores as u64,
        quantized_exact_scores: quantized_telemetry.exact_scores as u64,
        quantized_exact_score_ratio: quantized_telemetry.exact_scores as f64
            / quantized_telemetry.quantized_scores as f64,
        threshold_latency_ns_by_query: threshold_latencies.iter().map(Duration::as_nanos).collect(),
        quantized_latency_ns_by_query: quantized_latencies.iter().map(Duration::as_nanos).collect(),
        exhaustive_latency_ns_by_query: exhaustive_latencies
            .iter()
            .map(Duration::as_nanos)
            .collect(),
        threshold_latency: summarize(&mut threshold_latencies),
        quantized_latency: summarize(&mut quantized_latencies),
        exhaustive_latency: summarize(&mut exhaustive_latencies),
    };
    let serialized = serde_json::to_string_pretty(&result).unwrap() + "\n";
    if let Ok(output) = env::var("SPECTRA_OUTPUT") {
        let output = PathBuf::from(output);
        let temporary = output.with_extension("tmp");
        fs::write(&temporary, &serialized).unwrap();
        fs::rename(temporary, output).unwrap();
    }
    print!("{serialized}");
    assert_eq!(threshold_parity_mismatches, 0);
    assert_eq!(quantized_parity_mismatches, 0);
}

fn add_threshold_telemetry(total: &mut DenseThresholdTelemetry, current: DenseThresholdTelemetry) {
    total.sequential_accesses += current.sequential_accesses;
    total.duplicate_accesses += current.duplicate_accesses;
    total.exact_scores += current.exact_scores;
    total.threshold_updates += current.threshold_updates;
    total.points_emitted += current.points_emitted;
}

fn add_quantized_telemetry(total: &mut DenseQuantizedTelemetry, current: DenseQuantizedTelemetry) {
    total.quantized_scores += current.quantized_scores;
    total.exact_scores += current.exact_scores;
    total.points_emitted += current.points_emitted;
}

fn exhaustive(documents: &[DenseDocument], query: &[f32], top_k: usize) -> Vec<DenseScoredPoint> {
    let mut points: Vec<_> = documents
        .iter()
        .map(|document| DenseScoredPoint {
            id: document.id,
            score: document
                .vector
                .iter()
                .zip(query)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum(),
        })
        .collect();
    points.sort_unstable_by(|left, right| {
        OrderedFloat(right.score)
            .cmp(&OrderedFloat(left.score))
            .then_with(|| left.id.cmp(&right.id))
    });
    points.truncate(top_k);
    points
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    BufReader::new(File::open(path).unwrap())
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect()
}

fn summarize(latencies: &mut [Duration]) -> LatencySummary {
    latencies.sort_unstable();
    let percentile = |numerator: usize| {
        let index = (latencies.len() - 1) * numerator / 100;
        latencies[index].as_nanos()
    };
    LatencySummary {
        p50_ns: percentile(50),
        p95_ns: percentile(95),
        p99_ns: percentile(99),
        mean_ns: latencies.iter().map(Duration::as_nanos).sum::<u128>() / latencies.len() as u128,
    }
}
