// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ordered_float::OrderedFloat;
use segment::index::dense_ball::{DenseDocument, DenseScoredPoint};
use segment::index::dense_quantized::{DenseQuantizedIndex, DenseQuantizedTelemetry};
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
    build_ns: u128,
    original_vector_bytes: usize,
    encoded_bytes: usize,
    parity_mismatches: usize,
    quantized_scores: u64,
    exact_scores: u64,
    exact_score_ratio: f64,
    certified_latency_ns_by_query: Vec<u128>,
    exhaustive_latency_ns_by_query: Vec<u128>,
    certified_latency: LatencySummary,
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

    let build_start = Instant::now();
    let index = DenseQuantizedIndex::build(documents.clone()).unwrap();
    let build_ns = build_start.elapsed().as_nanos();
    let mut certified_latencies = Vec::with_capacity(query_rows.len());
    let mut exhaustive_latencies = Vec::with_capacity(query_rows.len());
    let mut parity_mismatches = 0usize;
    let mut telemetry = DenseQuantizedTelemetry::default();

    for query in query_rows {
        let exhaustive_start = Instant::now();
        let expected = exhaustive(&documents, &query.dense, top_k);
        exhaustive_latencies.push(exhaustive_start.elapsed());

        let certified_start = Instant::now();
        let mut stream = index.stream(&query.dense).unwrap();
        let actual: Vec<_> = stream.by_ref().take(top_k).collect();
        certified_latencies.push(certified_start.elapsed());
        let current = stream.telemetry();
        telemetry.quantized_scores += current.quantized_scores;
        telemetry.exact_scores += current.exact_scores;
        telemetry.points_emitted += current.points_emitted;
        parity_mismatches += usize::from(actual != expected);
    }

    let certified_latency_ns_by_query =
        certified_latencies.iter().map(Duration::as_nanos).collect();
    let exhaustive_latency_ns_by_query = exhaustive_latencies
        .iter()
        .map(Duration::as_nanos)
        .collect();
    let result = ResultRow {
        schema_version: 2,
        experiment: "spectra-dense-quantized-certificate-v2",
        dataset_dir: dataset_dir.display().to_string(),
        documents: documents.len(),
        queries: certified_latencies.len(),
        dimension,
        top_k,
        build_ns,
        original_vector_bytes: documents.len() * dimension * size_of::<f32>(),
        encoded_bytes: index.encoded_bytes(),
        parity_mismatches,
        quantized_scores: telemetry.quantized_scores as u64,
        exact_scores: telemetry.exact_scores as u64,
        exact_score_ratio: telemetry.exact_scores as f64 / telemetry.quantized_scores as f64,
        certified_latency_ns_by_query,
        exhaustive_latency_ns_by_query,
        certified_latency: summarize(&mut certified_latencies),
        exhaustive_latency: summarize(&mut exhaustive_latencies),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(parity_mismatches, 0);
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
