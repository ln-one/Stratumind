// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Matched Segment-owned Dense exact-plan comparison.

use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use fs_err::File;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{
    DEFAULT_VECTOR_NAME, QueryVector, VectorInternal, only_default_vector,
};
use segment::entry::{ReadSegmentEntry, SegmentEntry};
use segment::index::native_dense_stream::{NativeDensePlan, NativeDensePolicy};
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Distance, QuantizationConfig, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    SearchParams,
};
use segment::vector_storage::quantized::quantized_vectors::{
    QuantizedVectors, QuantizedVectorsStorageType,
};
use serde::{Deserialize, Serialize};

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
    compact_ordered_mismatches: usize,
    scalar_ordered_mismatches: usize,
    auto_ordered_mismatches: usize,
    auto_exact_prefix_queries: usize,
    auto_compact_queries: usize,
    auto_scalar_queries: usize,
    auto_exact_scan_queries: usize,
    compact_quantized_scores: usize,
    compact_exact_scores: usize,
    scalar_quantized_scores: usize,
    scalar_exact_scores: usize,
    compact_exact_score_ratio: f64,
    scalar_exact_score_ratio: f64,
    quantized_build_ns: u128,
    quantized_total_bytes: u64,
    scalar_certificate_bytes: u64,
    compact_certificate_bytes: u64,
    exact_latency: LatencySummary,
    compact_latency: LatencySummary,
    scalar_latency: LatencySummary,
    auto_latency: LatencySummary,
    exact_latency_ns_by_query: Vec<u128>,
    compact_latency_ns_by_query: Vec<u128>,
    scalar_latency_ns_by_query: Vec<u128>,
    auto_latency_ns_by_query: Vec<u128>,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", 20);
    let query_limit = env::var("SPECTRA_QUERY_LIMIT")
        .ok()
        .map(|value| value.parse::<usize>().expect("invalid SPECTRA_QUERY_LIMIT"));
    let mut corpus: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut queries: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Ok(limit) = env::var("SPECTRA_DOCUMENT_LIMIT") {
        corpus.truncate(limit.parse().expect("invalid SPECTRA_DOCUMENT_LIMIT"));
    }
    if let Some(limit) = query_limit {
        queries.truncate(limit);
    }
    assert!(!corpus.is_empty() && !queries.is_empty());
    let dimension = corpus[0].dense.len();
    assert!(top_k > 0 && top_k <= corpus.len());
    assert!(corpus.iter().all(|row| row.dense.len() == dimension));
    assert!(queries.iter().all(|row| row.dense.len() == dimension));

    let segment_dir = tempfile::Builder::new()
        .prefix("stratumind-dense-segment")
        .tempdir()
        .unwrap();
    let quantized_dir = tempfile::Builder::new()
        .prefix("stratumind-dense-quantized")
        .tempdir()
        .unwrap();
    let hardware_counter = HardwareCounterCell::new();
    let mut segment = build_simple_segment(segment_dir.path(), dimension, Distance::Dot).unwrap();
    for (position, row) in corpus.iter().enumerate() {
        assert_eq!(row.ordinal as usize, position);
        segment
            .upsert_point(
                position as u64,
                u64::from(row.ordinal).into(),
                only_default_vector(&row.dense),
                &hardware_counter,
            )
            .unwrap();
    }

    let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
        scalar: ScalarQuantizationConfig {
            r#type: ScalarType::Int8,
            quantile: None,
            always_ram: Some(true),
        },
    });
    let build_started = Instant::now();
    let quantized = QuantizedVectors::create_with_compact_limit(
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        &scalar_config,
        QuantizedVectorsStorageType::Immutable,
        quantized_dir.path(),
        1,
        usize::MAX,
        &AtomicBool::new(false),
    )
    .unwrap();
    let quantized_build_ns = build_started.elapsed().as_nanos();
    let quantized_total_bytes = quantized
        .files()
        .iter()
        .map(|path| std::fs::metadata(path).unwrap().len())
        .sum();
    let scalar_certificate_bytes = std::fs::metadata(quantized_dir.path().join(
        segment::vector_storage::quantized::quantized_vectors::QUANTIZED_SCALAR_CERTIFICATE_PATH,
    ))
    .unwrap()
    .len();
    let compact_certificate_bytes = std::fs::metadata(quantized_dir.path().join(
        segment::vector_storage::quantized::quantized_vectors::QUANTIZED_COMPACT_CERTIFICATE_PATH,
    ))
    .unwrap()
    .len();
    *segment.vector_data[DEFAULT_VECTOR_NAME]
        .quantized_vectors
        .borrow_mut() = Some(quantized);

    let mut exact_latencies = Vec::with_capacity(queries.len());
    let mut compact_latencies = Vec::with_capacity(queries.len());
    let mut scalar_latencies = Vec::with_capacity(queries.len());
    let mut auto_latencies = Vec::with_capacity(queries.len());
    let mut compact_mismatches = 0;
    let mut scalar_mismatches = 0;
    let mut auto_mismatches = 0;
    let mut auto_plan_queries = [0usize; 4];
    let mut compact_quantized_scores = 0;
    let mut compact_exact_scores = 0;
    let mut scalar_quantized_scores = 0;
    let mut scalar_exact_scores = 0;

    for (query_index, row) in queries.iter().enumerate() {
        let orders = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]];
        let mut exact = None;
        let mut compact = None;
        let mut scalar = None;
        let mut auto = None;
        for condition in orders[query_index % orders.len()] {
            match condition {
                0 => {
                    exact = Some(run_exact(&segment, &row.dense, top_k, &mut exact_latencies));
                }
                1 => {
                    compact = Some(run_native(
                        &segment,
                        &row.dense,
                        top_k,
                        false,
                        &mut compact_latencies,
                    ));
                }
                2 => {
                    scalar = Some(run_native(
                        &segment,
                        &row.dense,
                        top_k,
                        true,
                        &mut scalar_latencies,
                    ));
                }
                3 => {
                    auto = Some(run_auto(&segment, &row.dense, top_k, &mut auto_latencies));
                }
                _ => unreachable!(),
            }
        }
        let exact = exact.unwrap();
        let (compact, compact_telemetry) = compact.unwrap();
        let (scalar, scalar_telemetry) = scalar.unwrap();
        let (auto, auto_telemetry) = auto.unwrap();
        compact_mismatches += usize::from(compact != exact);
        scalar_mismatches += usize::from(scalar != exact);
        auto_mismatches += usize::from(auto != exact);
        assert_eq!(
            compact_telemetry.plan,
            Some(NativeDensePlan::CompactCertificate)
        );
        assert_eq!(
            scalar_telemetry.plan,
            Some(NativeDensePlan::ScalarCertificate)
        );
        match auto_telemetry.plan.expect("auto plan records its executor") {
            NativeDensePlan::ExactPrefix => auto_plan_queries[0] += 1,
            NativeDensePlan::CompactCertificate => auto_plan_queries[1] += 1,
            NativeDensePlan::ScalarCertificate => auto_plan_queries[2] += 1,
            NativeDensePlan::ExactScan => auto_plan_queries[3] += 1,
        }
        compact_quantized_scores += compact_telemetry.native_quantized_scores;
        compact_exact_scores += compact_telemetry.exact_scores;
        scalar_quantized_scores += scalar_telemetry.native_quantized_scores;
        scalar_exact_scores += scalar_telemetry.exact_scores;
    }

    let result = ResultRow {
        schema_version: 1,
        experiment: "stratumind-segment-owned-dense-certificate-v1",
        dataset_dir: dataset_dir.display().to_string(),
        documents: corpus.len(),
        queries: queries.len(),
        dimension,
        top_k,
        compact_ordered_mismatches: compact_mismatches,
        scalar_ordered_mismatches: scalar_mismatches,
        auto_ordered_mismatches: auto_mismatches,
        auto_exact_prefix_queries: auto_plan_queries[0],
        auto_compact_queries: auto_plan_queries[1],
        auto_scalar_queries: auto_plan_queries[2],
        auto_exact_scan_queries: auto_plan_queries[3],
        compact_quantized_scores,
        compact_exact_scores,
        scalar_quantized_scores,
        scalar_exact_scores,
        compact_exact_score_ratio: compact_exact_scores as f64 / compact_quantized_scores as f64,
        scalar_exact_score_ratio: scalar_exact_scores as f64 / scalar_quantized_scores as f64,
        quantized_build_ns,
        quantized_total_bytes,
        scalar_certificate_bytes,
        compact_certificate_bytes,
        exact_latency: summarize(&exact_latencies),
        compact_latency: summarize(&compact_latencies),
        scalar_latency: summarize(&scalar_latencies),
        auto_latency: summarize(&auto_latencies),
        exact_latency_ns_by_query: nanos(&exact_latencies),
        compact_latency_ns_by_query: nanos(&compact_latencies),
        scalar_latency_ns_by_query: nanos(&scalar_latencies),
        auto_latency_ns_by_query: nanos(&auto_latencies),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn run_exact(
    segment: &segment::segment::Segment,
    query: &[f32],
    top_k: usize,
    latencies: &mut Vec<Duration>,
) -> Vec<segment::types::ExtendedPointId> {
    let query: QueryVector = VectorInternal::Dense(query.to_vec()).into();
    let started = Instant::now();
    let points = segment
        .search(
            DEFAULT_VECTOR_NAME,
            &query,
            &Default::default(),
            &Default::default(),
            None,
            top_k,
            Some(&SearchParams {
                exact: true,
                ..Default::default()
            }),
        )
        .unwrap();
    latencies.push(started.elapsed());
    points.into_iter().map(|point| point.id).collect()
}

fn run_native(
    segment: &segment::segment::Segment,
    query: &[f32],
    top_k: usize,
    disable_compact_certificate: bool,
    latencies: &mut Vec<Duration>,
) -> (
    Vec<segment::types::ExtendedPointId>,
    segment::index::native_dense_stream::NativeDenseTelemetry,
) {
    let mut query_context = QueryContext::default();
    segment.fill_query_context(&mut query_context).unwrap();
    let segment_query_context = query_context.get_segment_query_context();
    let started = Instant::now();
    let (points, telemetry) = segment
        .with_view(|view| {
            view.with_native_dense_stream(
                DEFAULT_VECTOR_NAME,
                query,
                None,
                0,
                NativeDensePolicy {
                    scalar_min_points: 0,
                    compact_max_points: usize::MAX,
                    disable_compact_certificate,
                    ..NativeDensePolicy::default()
                },
                &segment_query_context,
                |next| {
                    (0..top_k)
                        .map(|_| next().map(Option::unwrap))
                        .collect::<segment::common::operation_error::OperationResult<Vec<_>>>()
                },
            )
        })
        .unwrap();
    latencies.push(started.elapsed());
    (
        points.into_iter().map(|point| point.id).collect(),
        telemetry,
    )
}

fn run_auto(
    segment: &segment::segment::Segment,
    query: &[f32],
    top_k: usize,
    latencies: &mut Vec<Duration>,
) -> (
    Vec<segment::types::ExtendedPointId>,
    segment::index::native_dense_stream::NativeDenseTelemetry,
) {
    let mut query_context = QueryContext::default();
    segment.fill_query_context(&mut query_context).unwrap();
    let segment_query_context = query_context.get_segment_query_context();
    let started = Instant::now();
    let (points, telemetry) = segment
        .with_view(|view| {
            view.with_native_dense_stream(
                DEFAULT_VECTOR_NAME,
                query,
                None,
                64,
                NativeDensePolicy::default(),
                &segment_query_context,
                |next| {
                    (0..top_k)
                        .map(|_| next().map(Option::unwrap))
                        .collect::<segment::common::operation_error::OperationResult<Vec<_>>>()
                },
            )
        })
        .unwrap();
    latencies.push(started.elapsed());
    (
        points.into_iter().map(|point| point.id).collect(),
        telemetry,
    )
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
