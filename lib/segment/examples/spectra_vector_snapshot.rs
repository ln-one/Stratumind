// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use segment::common::reciprocal_rank_fusion::{DEFAULT_RRF_K, exact_rrf_scoring};
use segment::index::dense_ball::{DenseBallTelemetry, DenseDocument};
use segment::index::hybrid_exact::HybridExactIndex;
use segment::types::{ExtendedPointId, ScoredPoint};
use serde::{Deserialize, Serialize};
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::block_max::{BlockMaxTelemetry, SparseDocument};

const DEFAULT_TOP_K: usize = 20;
const DEFAULT_BLOCK_SIZE: usize = 128;

#[derive(Debug, Deserialize)]
struct CorpusRow {
    ordinal: u32,
    source_id: String,
    dense: Vec<f32>,
    sparse_indices: Vec<u32>,
    sparse_values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct QueryRow {
    source_id: String,
    dense: Vec<f32>,
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
struct AggregatePhysicalTelemetry {
    sparse_blocks_expanded: u64,
    sparse_documents_evaluated: u64,
    dense_blocks_expanded: u64,
    dense_internal_nodes_expanded: u64,
    dense_bound_evaluations: u64,
    dense_documents_evaluated: u64,
}

impl AggregatePhysicalTelemetry {
    fn add_sparse(&mut self, telemetry: BlockMaxTelemetry) {
        self.sparse_blocks_expanded += telemetry.blocks_expanded as u64;
        self.sparse_documents_evaluated += telemetry.documents_evaluated as u64;
    }

    fn add_dense(&mut self, telemetry: DenseBallTelemetry) {
        self.dense_blocks_expanded += telemetry.blocks_expanded as u64;
        self.dense_internal_nodes_expanded += telemetry.internal_nodes_expanded as u64;
        self.dense_bound_evaluations += telemetry.bound_evaluations as u64;
        self.dense_documents_evaluated += telemetry.documents_evaluated as u64;
    }
}

#[derive(Debug, Default, Serialize)]
struct QualitySummary {
    judged_queries: usize,
    mean_recall_at_k: f64,
    mean_ndcg_at_k: f64,
}

#[derive(Debug, Serialize)]
struct SnapshotResult {
    schema_version: usize,
    experiment: &'static str,
    qdrant_base_commit: &'static str,
    build_profile: &'static str,
    dataset_dir: String,
    documents: usize,
    queries: usize,
    top_k: usize,
    rrf_k: usize,
    block_size: usize,
    dense_leaf_size: usize,
    dense_partition: &'static str,
    dense_clustering_iterations: usize,
    dense_branch_factor: usize,
    dense_dimension: usize,
    build_ns: u128,
    sparse_blocks: usize,
    dense_blocks: usize,
    dense_nodes: usize,
    parity_mismatches: usize,
    fixed_before_exhaustion: usize,
    all_sources_exhausted: usize,
    dynamic_source_pulls: [u64; 2],
    certification_checks: u64,
    dynamic_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
    dynamic_physical: AggregatePhysicalTelemetry,
    exhaustive_physical: AggregatePhysicalTelemetry,
    quality: QualitySummary,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let block_size = env_usize("SPECTRA_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    let dense_leaf_size = env_usize("SPECTRA_DENSE_LEAF_SIZE", block_size);
    let dense_clustering_iterations = env_usize("SPECTRA_DENSE_CLUSTERING_ITERATIONS", 0);
    let dense_branch_factor = env_usize("SPECTRA_DENSE_BRANCH_FACTOR", 0);
    let rrf_k = env_usize("SPECTRA_RRF_K", DEFAULT_RRF_K);
    assert!(top_k > 0);
    assert!(block_size > 0);
    assert!(dense_leaf_size > 0);
    assert!(rrf_k > 0);

    let corpus_rows: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut query_rows: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Ok(limit) = env::var("SPECTRA_QUERY_LIMIT") {
        let limit: usize = limit.parse().expect("invalid SPECTRA_QUERY_LIMIT");
        query_rows.truncate(limit);
    }
    assert!(!corpus_rows.is_empty(), "encoded corpus must not be empty");
    assert!(
        !query_rows.is_empty(),
        "encoded query set must not be empty"
    );
    let dense_dimension = corpus_rows[0].dense.len();
    assert!(dense_dimension > 0, "Dense vectors must not be empty");

    let mut source_to_ordinal = HashMap::with_capacity(corpus_rows.len());
    let mut sparse_documents = Vec::with_capacity(corpus_rows.len());
    let mut dense_documents = Vec::with_capacity(corpus_rows.len());
    for (position, row) in corpus_rows.into_iter().enumerate() {
        assert_eq!(
            row.ordinal as usize, position,
            "ordinals must be contiguous"
        );
        assert_eq!(row.dense.len(), dense_dimension, "Dense dimension mismatch");
        assert!(
            source_to_ordinal
                .insert(row.source_id, row.ordinal)
                .is_none(),
            "duplicate corpus source identity"
        );
        sparse_documents.push(SparseDocument {
            id: row.ordinal,
            vector: RemappedSparseVector {
                indices: row.sparse_indices,
                values: row.sparse_values,
            },
        });
        dense_documents.push(DenseDocument {
            id: row.ordinal,
            vector: row.dense,
        });
    }

    let build_start = Instant::now();
    let (index, dense_partition) = if dense_clustering_iterations == 0 {
        (
            HybridExactIndex::build(sparse_documents, dense_documents, block_size, block_size)
                .unwrap(),
            "identity-contiguous",
        )
    } else if dense_branch_factor == 0 {
        (
            HybridExactIndex::build_dense_clustered(
                sparse_documents,
                dense_documents,
                block_size,
                block_size,
                dense_clustering_iterations,
            )
            .unwrap(),
            "balanced-kmeans",
        )
    } else {
        (
            HybridExactIndex::build_dense_hierarchical(
                sparse_documents,
                dense_documents,
                block_size,
                dense_leaf_size,
                dense_branch_factor,
                dense_clustering_iterations,
            )
            .unwrap(),
            "hierarchical-balanced-kmeans",
        )
    };
    let build_elapsed = build_start.elapsed();
    let qrels = read_qrels(&dataset_dir.join("qrels.tsv"), &source_to_ordinal);

    let mut dynamic_latencies = Vec::with_capacity(query_rows.len());
    let mut exhaustive_latencies = Vec::with_capacity(query_rows.len());
    let mut dynamic_physical = AggregatePhysicalTelemetry::default();
    let mut exhaustive_physical = AggregatePhysicalTelemetry::default();
    let mut parity_mismatches = 0;
    let mut fixed_before_exhaustion = 0;
    let mut all_sources_exhausted = 0;
    let mut dynamic_source_pulls = [0u64; 2];
    let mut certification_checks = 0u64;
    let mut quality = QualityAccumulator::default();

    for query in query_rows {
        assert_eq!(query.dense.len(), dense_dimension, "Dense query mismatch");
        let sparse_query = RemappedSparseVector {
            indices: query.sparse_indices,
            values: query.sparse_values,
        };

        let exhaustive_start = Instant::now();
        let (sparse_ranking, sparse_full_telemetry) =
            collect_sparse(index.sparse_index(), sparse_query.clone());
        let (dense_ranking, dense_full_telemetry) =
            collect_dense(index.dense_index(), &query.dense);
        let exhaustive = exhaustive_rrf(&[sparse_ranking, dense_ranking], top_k, rrf_k);
        exhaustive_latencies.push(exhaustive_start.elapsed());
        exhaustive_physical.add_sparse(sparse_full_telemetry);
        exhaustive_physical.add_dense(dense_full_telemetry);

        let dynamic_start = Instant::now();
        let dynamic = index
            .search(sparse_query, &query.dense, top_k, rrf_k, None)
            .unwrap();
        dynamic_latencies.push(dynamic_start.elapsed());

        parity_mismatches += usize::from(dynamic.execution.point_ids != exhaustive);
        match dynamic.execution.stop_reason {
            segment::common::reciprocal_rank_fusion::DynamicRrfStopReason::TopKFixed => {
                fixed_before_exhaustion += 1
            }
            segment::common::reciprocal_rank_fusion::DynamicRrfStopReason::AllSourcesExhausted => {
                all_sources_exhausted += 1
            }
        }
        dynamic_source_pulls[0] += dynamic.execution.source_pulls[0] as u64;
        dynamic_source_pulls[1] += dynamic.execution.source_pulls[1] as u64;
        certification_checks += dynamic.execution.certification_checks as u64;
        dynamic_physical.add_sparse(dynamic.telemetry.sparse);
        dynamic_physical.add_dense(dynamic.telemetry.dense);
        if let Some(relevance) = qrels.get(&query.source_id) {
            quality.add(&dynamic.execution.point_ids, relevance);
        }
    }

    let result = SnapshotResult {
        schema_version: 1,
        experiment: "spectra-vector-snapshot-dynamic-exact-v1",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        dataset_dir: dataset_dir.display().to_string(),
        documents: index.document_count(),
        queries: dynamic_latencies.len(),
        top_k,
        rrf_k,
        block_size,
        dense_leaf_size,
        dense_partition,
        dense_clustering_iterations,
        dense_branch_factor,
        dense_dimension,
        build_ns: build_elapsed.as_nanos(),
        sparse_blocks: index.sparse_index().block_count(),
        dense_blocks: index.dense_index().block_count(),
        dense_nodes: index.dense_index().node_count(),
        parity_mismatches,
        fixed_before_exhaustion,
        all_sources_exhausted,
        dynamic_source_pulls,
        certification_checks,
        dynamic_latency: summarize_latency(&mut dynamic_latencies),
        exhaustive_latency: summarize_latency(&mut exhaustive_latencies),
        dynamic_physical,
        exhaustive_physical,
        quality: quality.finish(),
    };

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(
        parity_mismatches, 0,
        "dynamic snapshot WRRF diverged from exhaustive WRRF"
    );
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

fn read_qrels(
    path: &Path,
    source_to_ordinal: &HashMap<String, u32>,
) -> HashMap<String, HashMap<ExtendedPointId, u32>> {
    if !path.exists() {
        return HashMap::new();
    }
    let mut result: HashMap<String, HashMap<ExtendedPointId, u32>> = HashMap::new();
    for (position, line) in BufReader::new(File::open(path).unwrap())
        .lines()
        .enumerate()
    {
        let line = line.unwrap();
        let columns: Vec<_> = line.split('\t').collect();
        if position == 0 && columns.first() == Some(&"query-id") {
            continue;
        }
        assert_eq!(columns.len(), 3, "invalid qrels row {}", position + 1);
        let Some(ordinal) = source_to_ordinal.get(columns[1]) else {
            continue;
        };
        let relevance: u32 = columns[2].parse().unwrap();
        result
            .entry(columns[0].to_owned())
            .or_default()
            .insert(ExtendedPointId::from(*ordinal as u64), relevance);
    }
    result
}

fn collect_sparse(
    index: &sparse::index::block_max::BlockMaxIndex,
    query: RemappedSparseVector,
) -> (Vec<ExtendedPointId>, BlockMaxTelemetry) {
    let mut stream = index.stream(query).unwrap();
    let ranking = stream
        .by_ref()
        .map(|point| ExtendedPointId::from(point.idx as u64))
        .collect();
    (ranking, stream.telemetry())
}

fn collect_dense(
    index: &segment::index::dense_ball::DenseBallIndex,
    query: &[f32],
) -> (Vec<ExtendedPointId>, DenseBallTelemetry) {
    let mut stream = index.stream(query).unwrap();
    let ranking = stream
        .by_ref()
        .map(|point| ExtendedPointId::from(point.id as u64))
        .collect();
    (ranking, stream.telemetry())
}

fn exhaustive_rrf(
    sources: &[Vec<ExtendedPointId>],
    top_k: usize,
    rrf_k: usize,
) -> Vec<ExtendedPointId> {
    let responses = sources
        .iter()
        .map(|source| source.iter().copied().map(scored_point).collect())
        .collect();
    exact_rrf_scoring(responses, rrf_k, None)
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

#[derive(Default)]
struct QualityAccumulator {
    queries: usize,
    recall_sum: f64,
    ndcg_sum: f64,
}

impl QualityAccumulator {
    fn add(&mut self, ranking: &[ExtendedPointId], relevance: &HashMap<ExtendedPointId, u32>) {
        let relevant_count = relevance.values().filter(|score| **score > 0).count();
        if relevant_count == 0 {
            return;
        }
        self.queries += 1;
        let retrieved = ranking
            .iter()
            .filter(|id| relevance.get(id).is_some_and(|score| *score > 0))
            .count();
        self.recall_sum += retrieved as f64 / relevant_count as f64;

        let dcg: f64 = ranking
            .iter()
            .enumerate()
            .map(|(rank, id)| discounted_gain(*relevance.get(id).unwrap_or(&0), rank))
            .sum();
        let mut ideal: Vec<_> = relevance.values().copied().collect();
        ideal.sort_unstable_by(|left, right| right.cmp(left));
        let idcg: f64 = ideal
            .into_iter()
            .take(ranking.len())
            .enumerate()
            .map(|(rank, score)| discounted_gain(score, rank))
            .sum();
        self.ndcg_sum += if idcg == 0.0 { 0.0 } else { dcg / idcg };
    }

    fn finish(self) -> QualitySummary {
        if self.queries == 0 {
            return QualitySummary::default();
        }
        QualitySummary {
            judged_queries: self.queries,
            mean_recall_at_k: self.recall_sum / self.queries as f64,
            mean_ndcg_at_k: self.ndcg_sum / self.queries as f64,
        }
    }
}

fn discounted_gain(relevance: u32, rank: usize) -> f64 {
    ((2u64.pow(relevance) - 1) as f64) / (rank as f64 + 2.0).log2()
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
