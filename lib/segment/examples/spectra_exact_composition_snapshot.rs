// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact finite-DAG composition over a frozen Dense/Sparse vector snapshot.

use std::collections::{HashMap, HashSet};
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fs_err::File;
use segment::common::reciprocal_rank_fusion::{DynamicRrfPolicy, ExactRrfStream};
use segment::index::dense_ball::DenseDocument;
use segment::index::dense_quantized::DenseQuantizedIndex;
use segment::index::exact_composition::ExactCompositionExecutionPolicy;
use segment::types::ExtendedPointId;
use serde::{Deserialize, Serialize};
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::block_max::{BlockMaxIndex, SparseDocument};

mod support;
use support::exact_composition::{
    BuiltPlan, IsolatedExecutionMode, IsolatedExecutionResult, Topology, build_external_plan,
    build_plan,
};

const DEFAULT_TOP_K: usize = 20;
const DEFAULT_RRF_K: usize = 60;
const DEFAULT_BLOCK_SIZE: usize = 128;
const DEFAULT_QUERY_LIMIT: usize = 20;
const DEFAULT_REPEATS: usize = 3;

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

impl QueryRow {
    fn sparse(&self) -> RemappedSparseVector {
        RemappedSparseVector {
            indices: self.sparse_indices.clone(),
            values: self.sparse_values.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct LatencySummary {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Default, Serialize)]
struct QualitySummary {
    judged_queries: usize,
    mean_recall_at_k: f64,
    mean_ndcg_at_k: f64,
}

#[derive(Debug, Serialize)]
struct SnapshotCompositionResult {
    schema_version: usize,
    experiment: &'static str,
    build_profile: &'static str,
    dataset_dir: String,
    documents: usize,
    queries: usize,
    repeats: usize,
    topology: &'static str,
    leaf_semantics: &'static str,
    physical_leaf_nodes: usize,
    fusion_nodes: usize,
    top_k: usize,
    rrf_k: usize,
    block_size: usize,
    internal_limit: usize,
    exhaustive_after_pulls_per_input: Option<usize>,
    index_build_ns: u128,
    ordered_top_k_mismatches: usize,
    flat_order_drift_queries: usize,
    mean_flat_overlap_at_k: f64,
    dynamic_leaf_pulls: u64,
    exhaustive_leaf_identities: u64,
    leaf_work_ratio: f64,
    dynamic_intermediate_outputs: u64,
    exhaustive_intermediate_identities: u64,
    intermediate_work_ratio: f64,
    certification_checks: u64,
    cancelled_nodes: u64,
    peak_replay_identities: usize,
    sum_node_peak_replay_identities: usize,
    peak_exhaustive_materialized_identities: usize,
    dynamic_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
    quality: QualitySummary,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let rrf_k = env_usize("SPECTRA_RRF_K", DEFAULT_RRF_K);
    let block_size = env_usize("SPECTRA_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    let query_limit = env_usize("SPECTRA_QUERY_LIMIT", DEFAULT_QUERY_LIMIT);
    let judged_only = env_bool("SPECTRA_JUDGED_ONLY", true);
    let repeats = env_usize("SPECTRA_REPEATS", DEFAULT_REPEATS);
    let internal_limit = env_usize("SPECTRA_INTERNAL_LIMIT", top_k.saturating_mul(4));
    let exhaustive_after_pulls_per_input =
        env_optional_usize("SPECTRA_EXHAUSTIVE_AFTER_PULLS_PER_INPUT");
    let topology = Topology::parse(env::var("SPECTRA_TOPOLOGY").as_deref().unwrap_or("flat"));
    assert!(top_k > 0 && rrf_k > 0 && block_size > 0 && repeats > 0);

    let corpus_rows: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut query_rows: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    assert!(!corpus_rows.is_empty());
    let documents = corpus_rows.len();
    let dense_dimension = corpus_rows[0].dense.len();
    let mut source_to_ordinal = HashMap::with_capacity(documents);
    let mut dense_documents = Vec::with_capacity(documents);
    let mut sparse_documents = Vec::with_capacity(documents);
    for (position, row) in corpus_rows.into_iter().enumerate() {
        assert_eq!(row.ordinal as usize, position);
        assert_eq!(row.dense.len(), dense_dimension);
        assert!(
            source_to_ordinal
                .insert(row.source_id, row.ordinal)
                .is_none()
        );
        dense_documents.push(DenseDocument {
            id: row.ordinal,
            vector: row.dense,
        });
        sparse_documents.push(SparseDocument {
            id: row.ordinal,
            vector: RemappedSparseVector {
                indices: row.sparse_indices,
                values: row.sparse_values,
            },
        });
    }
    let qrels = read_qrels(&dataset_dir.join("qrels.tsv"), &source_to_ordinal);
    if judged_only && !qrels.is_empty() {
        query_rows.retain(|query| qrels.contains_key(&query.source_id));
    }
    query_rows.truncate(query_limit);
    assert!(
        query_rows.len() >= 2,
        "composition requires two query variants"
    );

    let build_start = Instant::now();
    let dense = DenseQuantizedIndex::build(dense_documents).unwrap();
    let sparse = BlockMaxIndex::build(sparse_documents, block_size).unwrap();
    let index_build_ns = build_start.elapsed().as_nanos();
    let policy = ExactCompositionExecutionPolicy {
        fusion_policy: DynamicRrfPolicy {
            exhaustive_after_pulls_per_source: exhaustive_after_pulls_per_input,
            ..DynamicRrfPolicy::default()
        },
    };
    if let Some(mode) = IsolatedExecutionMode::from_env() {
        run_isolated(
            mode,
            &query_rows,
            repeats,
            top_k,
            rrf_k,
            internal_limit,
            topology,
            &dense,
            &sparse,
            policy,
        );
        return;
    }

    let mut dynamic_latencies = Vec::with_capacity(query_rows.len() * repeats);
    let mut exhaustive_latencies = Vec::with_capacity(query_rows.len() * repeats);
    let mut ordered_top_k_mismatches = 0;
    let mut flat_order_drift_queries = 0;
    let mut flat_overlap_sum = 0.0;
    let mut dynamic_leaf_pulls = 0u64;
    let mut exhaustive_leaf_identities = 0u64;
    let mut dynamic_intermediate_outputs = 0u64;
    let mut exhaustive_intermediate_identities = 0u64;
    let mut certification_checks = 0u64;
    let mut cancelled_nodes = 0u64;
    let mut peak_replay_identities = 0;
    let mut sum_node_peak_replay_identities = 0;
    let mut peak_exhaustive_materialized_identities = 0;
    let mut quality = QualityAccumulator::default();
    let mut physical_leaf_nodes = 0;
    let mut fusion_nodes = 0;

    for query_index in 0..query_rows.len() {
        let query = &query_rows[query_index];
        let rewrite = &query_rows[(query_index + 1) % query_rows.len()];
        let orders = collect_orders(&dense, &sparse, query, rewrite);
        let flat = build_plan(Topology::Flat, &orders, top_k, rrf_k, internal_limit)
            .plan
            .exhaustive_root(top_k)
            .unwrap();
        let oracle = build_plan(topology, &orders, top_k, rrf_k, internal_limit);
        let built = build_external_plan(topology, orders.len(), top_k, rrf_k, internal_limit);
        physical_leaf_nodes = built.physical_leaf_sources.len();
        fusion_nodes = built.fusion_nodes;
        let semantic_output = oracle.plan.exhaustive_root(top_k).unwrap();
        flat_order_drift_queries += usize::from(semantic_output != flat);
        flat_overlap_sum += overlap_at_k(&semantic_output, &flat, top_k);
        if let Some(relevance) = qrels.get(&query.source_id) {
            quality.add(&semantic_output, relevance);
        }

        for repeat in 0..repeats {
            let (actual, exhaustive) = if (query_index + repeat).is_multiple_of(2) {
                let start = Instant::now();
                let actual = built
                    .plan
                    .execute_with_leaf_streams_and_policy(
                        top_k,
                        physical_streams(&built, &dense, &sparse, query, rewrite),
                        policy,
                    )
                    .unwrap();
                dynamic_latencies.push(start.elapsed());
                let start = Instant::now();
                let exhaustive = built
                    .plan
                    .exhaustive_with_leaf_streams(
                        top_k,
                        physical_streams(&built, &dense, &sparse, query, rewrite),
                    )
                    .unwrap();
                exhaustive_latencies.push(start.elapsed());
                (actual, exhaustive)
            } else {
                let start = Instant::now();
                let exhaustive = built
                    .plan
                    .exhaustive_with_leaf_streams(
                        top_k,
                        physical_streams(&built, &dense, &sparse, query, rewrite),
                    )
                    .unwrap();
                exhaustive_latencies.push(start.elapsed());
                let start = Instant::now();
                let actual = built
                    .plan
                    .execute_with_leaf_streams_and_policy(
                        top_k,
                        physical_streams(&built, &dense, &sparse, query, rewrite),
                        policy,
                    )
                    .unwrap();
                dynamic_latencies.push(start.elapsed());
                (actual, exhaustive)
            };

            ordered_top_k_mismatches += usize::from(actual.point_ids != exhaustive.point_ids);
            dynamic_leaf_pulls += actual.leaf_physical_pulls as u64;
            exhaustive_leaf_identities += exhaustive.leaf_identities_materialized as u64;
            dynamic_intermediate_outputs += actual.intermediate_outputs_produced as u64;
            exhaustive_intermediate_identities +=
                exhaustive.intermediate_identities_materialized as u64;
            certification_checks += actual
                .nodes
                .iter()
                .map(|node| node.certification_checks as u64)
                .sum::<u64>();
            cancelled_nodes += actual.cancelled_nodes as u64;
            peak_replay_identities =
                peak_replay_identities.max(actual.peak_replay_buffered_identities);
            sum_node_peak_replay_identities = sum_node_peak_replay_identities
                .max(actual.sum_node_peak_replay_buffered_identities);
            peak_exhaustive_materialized_identities = peak_exhaustive_materialized_identities
                .max(exhaustive.total_identities_materialized);
        }
    }

    let queries = query_rows.len();
    let result = SnapshotCompositionResult {
        schema_version: 1,
        experiment: "spectra-exact-composition-snapshot-v1",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        dataset_dir: dataset_dir.display().to_string(),
        documents,
        queries,
        repeats,
        topology: topology.name(),
        leaf_semantics: "Dense(q), Sparse(q), Dense(next-query rewrite), Sparse(next-query rewrite)",
        physical_leaf_nodes,
        fusion_nodes,
        top_k,
        rrf_k,
        block_size,
        internal_limit,
        exhaustive_after_pulls_per_input,
        index_build_ns,
        ordered_top_k_mismatches,
        flat_order_drift_queries,
        mean_flat_overlap_at_k: flat_overlap_sum / queries as f64,
        dynamic_leaf_pulls,
        exhaustive_leaf_identities,
        leaf_work_ratio: ratio(dynamic_leaf_pulls, exhaustive_leaf_identities),
        dynamic_intermediate_outputs,
        exhaustive_intermediate_identities,
        intermediate_work_ratio: ratio(
            dynamic_intermediate_outputs,
            exhaustive_intermediate_identities,
        ),
        certification_checks,
        cancelled_nodes,
        peak_replay_identities,
        sum_node_peak_replay_identities,
        peak_exhaustive_materialized_identities,
        dynamic_latency: summarize(&mut dynamic_latencies),
        exhaustive_latency: summarize(&mut exhaustive_latencies),
        quality: quality.finish(),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(ordered_top_k_mismatches, 0);
}

#[allow(clippy::too_many_arguments)]
fn run_isolated(
    mode: IsolatedExecutionMode,
    query_rows: &[QueryRow],
    repeats: usize,
    top_k: usize,
    rrf_k: usize,
    internal_limit: usize,
    topology: Topology,
    dense: &DenseQuantizedIndex,
    sparse: &BlockMaxIndex,
    policy: ExactCompositionExecutionPolicy,
) {
    let start = Instant::now();
    let mut point_ids_returned = 0;
    let mut leaf_identities_consumed = 0;
    let mut intermediate_identities = 0;
    for query_index in 0..query_rows.len() {
        let query = &query_rows[query_index];
        let rewrite = &query_rows[(query_index + 1) % query_rows.len()];
        let built = build_external_plan(topology, 4, top_k, rrf_k, internal_limit);
        for _ in 0..repeats {
            match mode {
                IsolatedExecutionMode::Dynamic => {
                    let execution = built
                        .plan
                        .execute_with_leaf_streams_and_policy(
                            top_k,
                            physical_streams(&built, dense, sparse, query, rewrite),
                            policy,
                        )
                        .unwrap();
                    point_ids_returned += execution.point_ids.len();
                    leaf_identities_consumed += execution.leaf_physical_pulls;
                    intermediate_identities += execution.intermediate_outputs_produced;
                }
                IsolatedExecutionMode::Exhaustive => {
                    let execution = built
                        .plan
                        .exhaustive_with_leaf_streams(
                            top_k,
                            physical_streams(&built, dense, sparse, query, rewrite),
                        )
                        .unwrap();
                    point_ids_returned += execution.point_ids.len();
                    leaf_identities_consumed += execution.leaf_identities_materialized;
                    intermediate_identities += execution.intermediate_identities_materialized;
                }
            }
        }
    }
    let result = IsolatedExecutionResult {
        schema_version: 1,
        experiment: "spectra-exact-composition-snapshot-isolated-v1",
        execution_mode: mode.name(),
        operations: query_rows.len() * repeats,
        point_ids_returned,
        leaf_identities_consumed,
        intermediate_identities,
        elapsed_ns: start.elapsed().as_nanos(),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn collect_orders(
    dense: &DenseQuantizedIndex,
    sparse: &BlockMaxIndex,
    query: &QueryRow,
    rewrite: &QueryRow,
) -> Vec<Vec<ExtendedPointId>> {
    vec![
        dense_order(dense, &query.dense),
        sparse_order(sparse, query.sparse()),
        dense_order(dense, &rewrite.dense),
        sparse_order(sparse, rewrite.sparse()),
    ]
}

fn physical_streams<'a>(
    built: &BuiltPlan,
    dense: &'a DenseQuantizedIndex,
    sparse: &'a BlockMaxIndex,
    query: &'a QueryRow,
    rewrite: &'a QueryRow,
) -> Vec<Option<ExactRrfStream<'a>>> {
    let mut streams = Vec::with_capacity(built.physical_leaf_sources.len() + built.fusion_nodes);
    for &source in &built.physical_leaf_sources {
        let stream: ExactRrfStream<'a> = match source {
            0 => Box::new(
                dense
                    .stream(&query.dense)
                    .unwrap()
                    .map(|point| Ok(ExtendedPointId::from(u64::from(point.id)))),
            ),
            1 => Box::new(
                sparse
                    .stream(query.sparse())
                    .unwrap()
                    .map(|point| Ok(ExtendedPointId::from(u64::from(point.idx)))),
            ),
            2 => Box::new(
                dense
                    .stream(&rewrite.dense)
                    .unwrap()
                    .map(|point| Ok(ExtendedPointId::from(u64::from(point.id)))),
            ),
            3 => Box::new(
                sparse
                    .stream(rewrite.sparse())
                    .unwrap()
                    .map(|point| Ok(ExtendedPointId::from(u64::from(point.idx)))),
            ),
            _ => unreachable!("the real snapshot defines four base leaf streams"),
        };
        streams.push(Some(stream));
    }
    streams.extend(std::iter::repeat_with(|| None).take(built.fusion_nodes));
    streams
}

fn dense_order(index: &DenseQuantizedIndex, query: &[f32]) -> Vec<ExtendedPointId> {
    index
        .stream(query)
        .unwrap()
        .map(|point| ExtendedPointId::from(u64::from(point.id)))
        .collect()
}

fn sparse_order(index: &BlockMaxIndex, query: RemappedSparseVector) -> Vec<ExtendedPointId> {
    index
        .stream(query)
        .unwrap()
        .map(|point| ExtendedPointId::from(u64::from(point.idx)))
        .collect()
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
) -> HashMap<String, HashMap<ExtendedPointId, i32>> {
    if !path.exists() {
        return HashMap::new();
    }
    let mut result: HashMap<String, HashMap<ExtendedPointId, i32>> = HashMap::new();
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
        result.entry(columns[0].to_owned()).or_default().insert(
            ExtendedPointId::from(u64::from(*ordinal)),
            columns[2].parse().unwrap(),
        );
    }
    result
}

#[derive(Default)]
struct QualityAccumulator {
    queries: usize,
    recall_sum: f64,
    ndcg_sum: f64,
}

impl QualityAccumulator {
    fn add(&mut self, ranking: &[ExtendedPointId], relevance: &HashMap<ExtendedPointId, i32>) {
        let relevant = relevance.values().filter(|score| **score > 0).count();
        if relevant == 0 {
            return;
        }
        self.queries += 1;
        let retrieved = ranking
            .iter()
            .filter(|id| relevance.get(id).is_some_and(|score| *score > 0))
            .count();
        self.recall_sum += retrieved as f64 / relevant as f64;
        let dcg: f64 = ranking
            .iter()
            .enumerate()
            .map(|(rank, id)| gain(*relevance.get(id).unwrap_or(&0), rank))
            .sum();
        let mut ideal: Vec<_> = relevance.values().copied().collect();
        ideal.sort_unstable_by(|left, right| right.cmp(left));
        let idcg: f64 = ideal
            .into_iter()
            .take(ranking.len())
            .enumerate()
            .map(|(rank, score)| gain(score, rank))
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

fn gain(relevance: i32, rank: usize) -> f64 {
    if relevance <= 0 {
        return 0.0;
    }
    ((2u64.pow(relevance as u32) - 1) as f64) / (rank as f64 + 2.0).log2()
}

fn overlap_at_k(left: &[ExtendedPointId], right: &[ExtendedPointId], top_k: usize) -> f64 {
    let left: HashSet<_> = left.iter().take(top_k).copied().collect();
    let right: HashSet<_> = right.iter().take(top_k).copied().collect();
    let denominator = left.len().min(right.len());
    if denominator == 0 {
        if left.is_empty() && right.is_empty() {
            1.0
        } else {
            0.0
        }
    } else {
        left.intersection(&right).count() as f64 / denominator as f64
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn summarize(values: &mut [Duration]) -> LatencySummary {
    values.sort_unstable();
    let percentile = |percent: usize| {
        values[(values.len().saturating_sub(1) * percent).div_ceil(100)].as_nanos()
    };
    LatencySummary {
        p50_ns: percentile(50),
        p95_ns: percentile(95),
        p99_ns: percentile(99),
        mean_ns: values.iter().map(Duration::as_nanos).sum::<u128>() / values.len() as u128,
    }
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
    match env::var(name).as_deref() {
        Ok("1" | "true") => true,
        Ok("0" | "false") => false,
        Ok(value) => panic!("invalid {name}: {value}"),
        Err(_) => default,
    }
}
