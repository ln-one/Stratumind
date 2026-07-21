// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::budget::ResourcePermit;
use common::counter::hardware_counter::HardwareCounterCell;
use common::flags::FeatureFlags;
use common::progress_tracker::ProgressTracker;
use common::types::PointOffsetType;
use rand::SeedableRng;
use rand::rngs::StdRng;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
use segment::entry::entry_point::SegmentEntry;
use segment::index::VectorIndexRead;
use segment::index::dense_ball::{DenseBallIndex, DenseDocument};
use segment::index::dense_proposal_certified::DenseProposalCertifiedStream;
use segment::index::dense_scalar_certified::DenseScalarCertifiedIndex;
use segment::index::hnsw_index::get_num_indexing_threads;
use segment::index::hnsw_index::hnsw::{HNSWIndex, HnswIndexOpenArgs};
use segment::segment_constructor::VectorIndexBuildArgs;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Distance, HnswConfig, HnswGlobalConfig, QuantizationConfig, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType, SearchParams, SeqNumberType,
};
use segment::vector_storage::quantized::quantized_vectors::{
    QuantizedVectors, QuantizedVectorsStorageType,
};
use serde::{Deserialize, Serialize};
use tempfile::Builder;

const DEFAULT_TOP_K: usize = 100;
const DEFAULT_HNSW_EF: usize = 128;
const DEFAULT_HNSW_M: usize = 16;
const DEFAULT_EF_CONSTRUCT: usize = 100;
const DEFAULT_DENSE_LEAF_SIZE: usize = 64;
const DEFAULT_DENSE_BRANCH_FACTOR: usize = 8;
const DEFAULT_DENSE_CLUSTERING_ITERATIONS: usize = 4;

#[derive(Debug, Deserialize)]
struct CorpusRow {
    ordinal: PointOffsetType,
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
    throughput_queries_per_second: f64,
}

#[derive(Debug, Serialize)]
struct DenseNativeResult {
    schema_version: usize,
    experiment: &'static str,
    qdrant_base_commit: &'static str,
    build_profile: &'static str,
    dataset_dir: String,
    documents: usize,
    queries: usize,
    dimension: usize,
    top_k: usize,
    proposal_k: usize,
    certificate_batch_size: usize,
    dense_leaf_size: usize,
    dense_branch_factor: usize,
    dense_clustering_iterations: usize,
    hnsw_m: usize,
    hnsw_ef_construct: usize,
    hnsw_ef_search: usize,
    hnsw_build_threads: usize,
    segment_load_ns: u128,
    hnsw_build_ns: u128,
    dense_ball_build_ns: u128,
    dense_ball_nodes: usize,
    dense_ball_leaves: usize,
    scalar_quantized_build_ns: u128,
    scalar_quantized_index_bytes: u64,
    vector_segment_bytes: u64,
    vector_segment_files: usize,
    hnsw_index_bytes: u64,
    hnsw_index_files: usize,
    total_index_bytes: u64,
    ordered_top_k_mismatches: usize,
    certified_ordered_top_k_mismatches: usize,
    proposal_certified_ordered_top_k_mismatches: usize,
    mean_recall_at_k: f64,
    certified_exact_scores: u64,
    certified_quantized_scores: u64,
    certified_exact_score_ratio: f64,
    proposal_certified_exact_scores: u64,
    proposal_certified_bound_evaluations: u64,
    proposal_certified_internal_nodes_expanded: u64,
    proposal_certified_leaf_nodes_expanded: u64,
    proposal_certified_exact_score_ratio: f64,
    hnsw_latency_ns_by_query: Vec<u128>,
    exact_scan_latency_ns_by_query: Vec<u128>,
    hnsw_latency: LatencySummary,
    certified_latency: LatencySummary,
    proposal_certificate_only_latency: LatencySummary,
    proposal_certified_total_latency: LatencySummary,
    exact_scan_latency: LatencySummary,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let hnsw_ef = env_usize("SPECTRA_HNSW_EF", DEFAULT_HNSW_EF);
    let hnsw_m = env_usize("SPECTRA_HNSW_M", DEFAULT_HNSW_M);
    let ef_construct = env_usize("SPECTRA_HNSW_EF_CONSTRUCT", DEFAULT_EF_CONSTRUCT);
    let build_threads = env_usize("SPECTRA_HNSW_BUILD_THREADS", 1);
    let proposal_k = env_usize("SPECTRA_PROPOSAL_K", top_k);
    let certificate_batch_size = env_usize("SPECTRA_CERTIFICATE_BATCH_SIZE", top_k);
    let dense_leaf_size = env_usize("SPECTRA_DENSE_LEAF_SIZE", DEFAULT_DENSE_LEAF_SIZE);
    let dense_branch_factor = env_usize("SPECTRA_DENSE_BRANCH_FACTOR", DEFAULT_DENSE_BRANCH_FACTOR);
    let dense_clustering_iterations = env_usize(
        "SPECTRA_DENSE_CLUSTERING_ITERATIONS",
        DEFAULT_DENSE_CLUSTERING_ITERATIONS,
    );
    assert!(top_k > 0);
    assert!(hnsw_ef > 0);
    assert!(hnsw_m > 0);
    assert!(ef_construct > 0);
    assert!(build_threads > 0);
    assert!(proposal_k >= top_k);
    assert!(certificate_batch_size > 0);

    let corpus: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut queries: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Ok(limit) = env::var("SPECTRA_QUERY_LIMIT") {
        queries.truncate(limit.parse().expect("invalid SPECTRA_QUERY_LIMIT"));
    }
    assert!(!corpus.is_empty());
    assert!(!queries.is_empty());
    let dimension = corpus[0].dense.len();
    assert!(dimension > 0);
    assert!(corpus.iter().all(|row| row.dense.len() == dimension));
    assert!(queries.iter().all(|row| row.dense.len() == dimension));
    assert!(top_k <= corpus.len());
    assert!(proposal_k <= corpus.len());

    let segment_dir = Builder::new().prefix("spectra-segment").tempdir().unwrap();
    let hnsw_dir = Builder::new().prefix("spectra-hnsw").tempdir().unwrap();
    let quantized_dir = Builder::new().prefix("spectra-scalar").tempdir().unwrap();
    let hardware_counter = HardwareCounterCell::new();
    let load_started = Instant::now();
    let mut segment = build_simple_segment(segment_dir.path(), dimension, Distance::Dot).unwrap();
    for (position, row) in corpus.iter().enumerate() {
        assert_eq!(
            row.ordinal as usize, position,
            "ordinals must be contiguous"
        );
        segment
            .upsert_point(
                position as SeqNumberType,
                (row.ordinal as u64).into(),
                only_default_vector(&row.dense),
                &hardware_counter,
            )
            .unwrap();
    }
    let segment_load_ns = load_started.elapsed().as_nanos();

    let scalar_config = QuantizationConfig::Scalar(ScalarQuantization {
        scalar: ScalarQuantizationConfig {
            r#type: ScalarType::Int8,
            quantile: None,
            always_ram: Some(true),
        },
    });
    let scalar_build_started = Instant::now();
    let scalar_quantized = QuantizedVectors::create(
        &segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow(),
        &scalar_config,
        QuantizedVectorsStorageType::Immutable,
        quantized_dir.path(),
        1,
        &AtomicBool::new(false),
    )
    .unwrap();
    let scalar_quantized_build_ns = scalar_build_started.elapsed().as_nanos();
    let certified_index = DenseScalarCertifiedIndex::build(
        &scalar_quantized,
        corpus
            .iter()
            .map(|row| DenseDocument {
                id: row.ordinal,
                vector: row.dense.clone(),
            })
            .collect(),
    )
    .unwrap();
    let dense_ball_build_started = Instant::now();
    let dense_ball = DenseBallIndex::build_hierarchical_clustered(
        corpus
            .iter()
            .map(|row| DenseDocument {
                id: row.ordinal,
                vector: row.dense.clone(),
            })
            .collect(),
        dense_leaf_size,
        dense_branch_factor,
        dense_clustering_iterations,
    )
    .unwrap();
    let dense_ball_build_ns = dense_ball_build_started.elapsed().as_nanos();
    let (scalar_quantized_index_bytes, _) = directory_size(quantized_dir.path());

    let hnsw_config = HnswConfig {
        m: hnsw_m,
        ef_construct,
        full_scan_threshold: 0,
        max_indexing_threads: build_threads,
        on_disk: Some(false),
        payload_m: None,
        inline_storage: None,
    };
    let permit_cpu_count = get_num_indexing_threads(hnsw_config.max_indexing_threads);
    let permit = Arc::new(ResourcePermit::dummy(permit_cpu_count as u32));
    let stopped = AtomicBool::new(false);
    let mut rng = StdRng::seed_from_u64(0x5350_4543_5452_4101);
    let build_started = Instant::now();
    let hnsw = HNSWIndex::build(
        HnswIndexOpenArgs {
            path: hnsw_dir.path(),
            id_tracker: segment.id_tracker.clone(),
            vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .clone(),
            quantized_vectors: segment.vector_data[DEFAULT_VECTOR_NAME]
                .quantized_vectors
                .clone(),
            payload_index: segment.payload_index.clone(),
            hnsw_config,
        },
        VectorIndexBuildArgs {
            permit,
            old_indices: &[],
            gpu_device: None,
            rng: &mut rng,
            stopped: &stopped,
            hnsw_global_config: &HnswGlobalConfig::default(),
            feature_flags: FeatureFlags::default(),
            progress: ProgressTracker::new_for_test(),
        },
    )
    .unwrap();
    let hnsw_build_ns = build_started.elapsed().as_nanos();
    let (vector_segment_bytes, vector_segment_files) = directory_size(segment_dir.path());
    let (hnsw_index_bytes, hnsw_index_files) = directory_size(hnsw_dir.path());

    let mut hnsw_latencies = Vec::with_capacity(queries.len());
    let mut certified_latencies = Vec::with_capacity(queries.len());
    let mut proposal_certificate_only_latencies = Vec::with_capacity(queries.len());
    let mut proposal_certified_total_latencies = Vec::with_capacity(queries.len());
    let mut exact_latencies = Vec::with_capacity(queries.len());
    let mut ordered_top_k_mismatches = 0;
    let mut certified_ordered_top_k_mismatches = 0;
    let mut proposal_certified_ordered_top_k_mismatches = 0;
    let mut certified_exact_scores = 0u64;
    let mut certified_quantized_scores = 0u64;
    let mut proposal_certified_exact_scores = 0u64;
    let mut proposal_certified_bound_evaluations = 0u64;
    let mut proposal_certified_internal_nodes_expanded = 0u64;
    let mut proposal_certified_leaf_nodes_expanded = 0u64;
    let mut recall_sum = 0.0;

    for row in &queries {
        let query = row.dense.clone().into();

        let exact_started = Instant::now();
        let exact = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&query], None, top_k, None, &Default::default())
            .unwrap()
            .pop()
            .unwrap();
        exact_latencies.push(exact_started.elapsed());

        let certified_started = Instant::now();
        let mut certified_stream = certified_index
            .stream(&scalar_quantized, &row.dense, HardwareCounterCell::new())
            .unwrap();
        let certified: Vec<_> = certified_stream.by_ref().take(top_k).collect();
        certified_latencies.push(certified_started.elapsed());
        let certified_telemetry = certified_stream.telemetry();
        certified_exact_scores += certified_telemetry.exact_scores as u64;
        certified_quantized_scores += certified_telemetry.native_quantized_scores as u64;

        let proposal_total_started = Instant::now();
        let hnsw_started = Instant::now();
        let approximate = hnsw
            .search(
                &[&query],
                None,
                proposal_k,
                Some(&SearchParams {
                    hnsw_ef: Some(hnsw_ef),
                    exact: false,
                    ..Default::default()
                }),
                &Default::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        hnsw_latencies.push(hnsw_started.elapsed());

        let proposal_ids: Vec<_> = approximate.iter().map(|point| point.idx).collect();
        let proposal_certificate_started = Instant::now();
        let mut proposal_stream = DenseProposalCertifiedStream::new(
            &dense_ball,
            &row.dense,
            &proposal_ids,
            certificate_batch_size,
        )
        .unwrap();
        let proposal_certified: Vec<_> = proposal_stream.by_ref().take(top_k).collect();
        proposal_certificate_only_latencies.push(proposal_certificate_started.elapsed());
        proposal_certified_total_latencies.push(proposal_total_started.elapsed());
        let proposal_telemetry = proposal_stream.telemetry();
        proposal_certified_exact_scores += proposal_telemetry.exact_scores as u64;
        proposal_certified_bound_evaluations += proposal_telemetry.bound_evaluations as u64;
        proposal_certified_internal_nodes_expanded +=
            proposal_telemetry.internal_nodes_expanded as u64;
        proposal_certified_leaf_nodes_expanded += proposal_telemetry.leaf_nodes_expanded as u64;

        ordered_top_k_mismatches += usize::from(approximate.iter().take(top_k).ne(exact.iter()));
        certified_ordered_top_k_mismatches += usize::from(
            certified
                .iter()
                .map(|point| point.id)
                .ne(exact.iter().map(|point| point.idx)),
        );
        proposal_certified_ordered_top_k_mismatches += usize::from(
            proposal_certified
                .iter()
                .map(|point| point.id)
                .ne(exact.iter().map(|point| point.idx)),
        );
        let exact_ids: std::collections::HashSet<_> = exact.iter().map(|point| point.idx).collect();
        let overlap = approximate
            .iter()
            .take(top_k)
            .filter(|point| exact_ids.contains(&point.idx))
            .count();
        recall_sum += overlap as f64 / exact.len() as f64;
    }

    let query_count = queries.len();
    let hnsw_latency_ns_by_query = hnsw_latencies.iter().map(Duration::as_nanos).collect();
    let exact_scan_latency_ns_by_query = exact_latencies.iter().map(Duration::as_nanos).collect();
    let result = DenseNativeResult {
        schema_version: 3,
        experiment: "spectra-dense-qdrant-native-v3",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        dataset_dir: dataset_dir.display().to_string(),
        documents: corpus.len(),
        queries: query_count,
        dimension,
        top_k,
        proposal_k,
        certificate_batch_size,
        dense_leaf_size,
        dense_branch_factor,
        dense_clustering_iterations,
        hnsw_m,
        hnsw_ef_construct: ef_construct,
        hnsw_ef_search: hnsw_ef,
        hnsw_build_threads: build_threads,
        segment_load_ns,
        hnsw_build_ns,
        dense_ball_build_ns,
        dense_ball_nodes: dense_ball.node_count(),
        dense_ball_leaves: dense_ball.block_count(),
        scalar_quantized_build_ns,
        scalar_quantized_index_bytes,
        vector_segment_bytes,
        vector_segment_files,
        hnsw_index_bytes,
        hnsw_index_files,
        total_index_bytes: vector_segment_bytes + hnsw_index_bytes,
        ordered_top_k_mismatches,
        certified_ordered_top_k_mismatches,
        proposal_certified_ordered_top_k_mismatches,
        mean_recall_at_k: recall_sum / query_count as f64,
        certified_exact_scores,
        certified_quantized_scores,
        certified_exact_score_ratio: certified_exact_scores as f64
            / certified_quantized_scores as f64,
        proposal_certified_exact_scores,
        proposal_certified_bound_evaluations,
        proposal_certified_internal_nodes_expanded,
        proposal_certified_leaf_nodes_expanded,
        proposal_certified_exact_score_ratio: proposal_certified_exact_scores as f64
            / (query_count * corpus.len()) as f64,
        hnsw_latency_ns_by_query,
        exact_scan_latency_ns_by_query,
        hnsw_latency: summarize_latency(&mut hnsw_latencies),
        certified_latency: summarize_latency(&mut certified_latencies),
        proposal_certificate_only_latency: summarize_latency(
            &mut proposal_certificate_only_latencies,
        ),
        proposal_certified_total_latency: summarize_latency(
            &mut proposal_certified_total_latencies,
        ),
        exact_scan_latency: summarize_latency(&mut exact_latencies),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
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

fn directory_size(path: &Path) -> (u64, usize) {
    let mut bytes = 0;
    let mut files = 0;
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                bytes += metadata.len();
                files += 1;
            }
        }
    }
    (bytes, files)
}

fn summarize_latency(durations: &mut [Duration]) -> LatencySummary {
    durations.sort_unstable();
    let total_ns: u128 = durations.iter().map(Duration::as_nanos).sum();
    let total_seconds = total_ns as f64 / 1_000_000_000.0;
    LatencySummary {
        p50_ns: percentile(durations, 50).as_nanos(),
        p95_ns: percentile(durations, 95).as_nanos(),
        p99_ns: percentile(durations, 99).as_nanos(),
        mean_ns: total_ns / durations.len() as u128,
        throughput_queries_per_second: durations.len() as f64 / total_seconds,
    }
}

fn percentile(durations: &[Duration], percentile: usize) -> Duration {
    let index = (durations.len() * percentile)
        .div_ceil(100)
        .saturating_sub(1);
    durations[index]
}
