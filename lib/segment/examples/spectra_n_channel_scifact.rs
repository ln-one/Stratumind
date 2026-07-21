// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use segment::common::reciprocal_rank_fusion::{
    DEFAULT_RRF_K, DynamicRrfPolicy, DynamicRrfScheduler, DynamicRrfStopReason, exact_rrf_scoring,
};
use segment::index::dense_ball::DenseDocument;
use segment::index::n_channel_exact::{
    DenseStreamStrategy, ExactChannelQuery, ExactChannelTelemetry, NChannelExactIndex,
    SparseStreamStrategy,
};
use segment::index::n_channel_router::{PrefixAgreementRouter, PrefixExecutionChoice};
use segment::types::{ExtendedPointId, ScoredPoint};
use serde::{Deserialize, Serialize};
use sparse::SearchScratchPool;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::adaptive_top_k_stream::AdaptiveTopKStream;
use sparse::index::block_max::SparseDocument;

const DEFAULT_TOP_K: usize = 20;
const DEFAULT_SPARSE_BLOCK_SIZE: usize = 128;

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

struct QueryVariants {
    source_id: String,
    dense: Vec<Vec<f32>>,
    sparse: Vec<RemappedSparseVector>,
}

impl QueryVariants {
    fn build(row: QueryRow) -> Self {
        let mut dense = vec![row.dense.clone()];
        for variant in 1..4 {
            let mut vector = row.dense.clone();
            for (coordinate, value) in vector.iter_mut().enumerate() {
                let keep = match variant {
                    1 => coordinate.is_multiple_of(2),
                    2 => !coordinate.is_multiple_of(2),
                    _ => true,
                };
                if !keep {
                    *value = 0.0;
                } else if variant == 3 {
                    *value *= 0.75 + (coordinate % 4) as f32 / 6.0;
                }
            }
            normalize(&mut vector);
            dense.push(vector);
        }

        let original = RemappedSparseVector {
            indices: row.sparse_indices,
            values: row.sparse_values,
        };
        let mut sparse = vec![original.clone()];
        for variant in 1..4 {
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for (position, (&index, &value)) in
                original.indices.iter().zip(&original.values).enumerate()
            {
                let keep = match variant {
                    1 => position.is_multiple_of(2),
                    2 => !position.is_multiple_of(2),
                    _ => true,
                };
                if keep {
                    indices.push(index);
                    values.push(if variant == 3 {
                        value * (0.75 + (position % 3) as f32 / 4.0)
                    } else {
                        value
                    });
                }
            }
            if indices.is_empty() && !original.indices.is_empty() {
                indices.push(original.indices[0]);
                values.push(original.values[0]);
            }
            sparse.push(RemappedSparseVector { indices, values });
        }
        Self {
            source_id: row.source_id,
            dense,
            sparse,
        }
    }

    fn channels(&self, count: usize) -> Vec<ExactChannelQuery<'_>> {
        (0..count)
            .map(|channel| {
                if channel.is_multiple_of(2) {
                    ExactChannelQuery::Dense(&self.dense[(channel / 2) % self.dense.len()])
                } else {
                    ExactChannelQuery::Sparse(
                        self.sparse[(channel / 2) % self.sparse.len()].clone(),
                    )
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
enum SchedulerSelection {
    MaxNext,
    CompetitorCost,
    SafeRouter,
}

impl SchedulerSelection {
    fn from_env() -> Self {
        match env::var("STRATUMIND_SCHEDULER").as_deref() {
            Ok("max-next") => Self::MaxNext,
            Ok("competitor-cost") | Err(_) => Self::CompetitorCost,
            Ok("safe-router") => Self::SafeRouter,
            Ok(value) => panic!(
                "STRATUMIND_SCHEDULER must be max-next, competitor-cost, or safe-router; got {value}"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::MaxNext => "max-next",
            Self::CompetitorCost => "competitor-cost",
            Self::SafeRouter => "safe-router",
        }
    }

    fn policy_scheduler(self) -> DynamicRrfScheduler {
        match self {
            Self::MaxNext => DynamicRrfScheduler::MaxNextContribution,
            Self::CompetitorCost | Self::SafeRouter => DynamicRrfScheduler::CompetitorCostAware,
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

#[derive(Debug, Serialize)]
struct ChannelMatrixRow {
    channels: usize,
    queries: usize,
    ordered_top_k_mismatches: usize,
    top_k_fixed_queries: usize,
    all_sources_exhausted_queries: usize,
    dynamic_source_pulls: Vec<u64>,
    exhaustive_source_pulls: Vec<u64>,
    source_pull_ratio: f64,
    dynamic_dense_quantized_scores: u64,
    dynamic_dense_exact_scores: u64,
    dynamic_dense_document_major_passes: u64,
    dynamic_dense_document_rows_traversed: u64,
    dynamic_dense_logical_dot_products: u64,
    dynamic_dense_physical_channels: u64,
    dynamic_dense_physical_dot_products: u64,
    dynamic_dense_preparation_ns: u128,
    dynamic_sparse_documents_evaluated: u64,
    dynamic_sparse_posting_elements_visited: u64,
    dynamic_sparse_batches_expanded: u64,
    dynamic_sparse_adaptive_refills: u64,
    dynamic_sparse_results_materialized: u64,
    dynamic_sparse_shared_unique_terms: u64,
    dynamic_sparse_shared_logical_term_accesses: u64,
    dynamic_sparse_shared_physical_posting_elements: u64,
    dynamic_sparse_shared_logical_posting_elements: u64,
    dynamic_sparse_shared_preparation_ns: u128,
    dynamic_sparse_auto_selected_shared_queries: usize,
    dynamic_sparse_estimated_unique_posting_elements: u64,
    dynamic_sparse_estimated_logical_posting_elements: u64,
    dynamic_sparse_estimated_deduplicated_posting_elements: u64,
    dynamic_sparse_probe_selected_shared_queries: usize,
    dynamic_sparse_prefix_identities_replayed: u64,
    dynamic_sparse_shared_lazy_unique_batches: u64,
    dynamic_sparse_shared_lazy_reused_batches: u64,
    dynamic_sparse_shared_lazy_physical_posting_elements: u64,
    dynamic_sparse_shared_lazy_logical_posting_elements: u64,
    dynamic_sparse_shared_lazy_physical_score_multiplications: u64,
    dynamic_sparse_shared_lazy_logical_score_multiplications: u64,
    dynamic_sparse_shared_lazy_cached_channel_points: u64,
    router_dynamic_queries: usize,
    router_exhaustive_queries: usize,
    router_oracle_matches: usize,
    dynamic_oracle_queries: usize,
    router_average_prefix_overlap: f64,
    router_minimum_prefix_overlap: f64,
    router_total_regret_ns: u128,
    quality_metric_queries: usize,
    recall_at_k: Option<f64>,
    ndcg_at_k: Option<f64>,
    dynamic_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
    router_simulated_latency: LatencySummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_latency_ns_by_query: Option<Vec<u128>>,
}

#[derive(Debug, Serialize)]
struct RouterObservation {
    channels: usize,
    query_index: usize,
    average_prefix_overlap: f64,
    minimum_prefix_overlap: f64,
    probe_ns: u128,
    dynamic_ns: u128,
    exhaustive_ns: u128,
}

#[derive(Debug, Serialize)]
struct SparseStrategyObservation {
    channels: usize,
    query_index: usize,
    unique_posting_elements: usize,
    logical_posting_elements: usize,
    posting_ns: u128,
    shared_ns: u128,
    ordered_top_k_mismatch: bool,
}

#[derive(Debug, Serialize)]
struct SparseLazyStrategyObservation {
    channels: usize,
    query_index: usize,
    repetition: usize,
    position: usize,
    strategy: &'static str,
    elapsed_ns: u128,
    source_pulls: usize,
    auto_selected_shared: bool,
    shared_lazy_physical_posting_elements: usize,
    shared_lazy_logical_posting_elements: usize,
    shared_lazy_physical_score_multiplications: usize,
    shared_lazy_logical_score_multiplications: usize,
    ordered_top_k_mismatch: bool,
}

#[derive(Debug, Serialize)]
struct ExperimentResult {
    schema_version: usize,
    experiment: &'static str,
    qdrant_base_commit: &'static str,
    dataset_dir: String,
    documents: usize,
    dimension: usize,
    top_k: usize,
    rrf_k: usize,
    warmup_check_interval: Option<usize>,
    exhaustive_after_pulls_per_source: Option<usize>,
    scheduler: &'static str,
    sparse_stream_strategy: String,
    dense_stream_strategy: String,
    router_probe_depth: usize,
    router_minimum_overlap: f64,
    include_router_observations: bool,
    include_sparse_strategy_observations: bool,
    include_sparse_lazy_strategy_observations: bool,
    sparse_lazy_strategy_repetitions: usize,
    include_query_latencies: bool,
    build_ns: u128,
    matrix: Vec<ChannelMatrixRow>,
    router_observations: Vec<RouterObservation>,
    sparse_strategy_observations: Vec<SparseStrategyObservation>,
    sparse_lazy_strategy_observations: Vec<SparseLazyStrategyObservation>,
}

fn main() {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let warmup_check_interval = env::var("SPECTRA_WRRF_WARMUP_CHECK_INTERVAL")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("invalid warmup check interval")
        });
    let exhaustive_after_pulls_per_source = env::var("SPECTRA_WRRF_EXHAUSTIVE_AFTER_PULLS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("invalid exhaustive pull budget")
        });
    let scheduler = SchedulerSelection::from_env();
    let policy = DynamicRrfPolicy {
        warmup_check_interval,
        exhaustive_after_pulls_per_source,
        scheduler: scheduler.policy_scheduler(),
        ..DynamicRrfPolicy::default()
    };
    let sparse_block_size = env_usize("SPECTRA_SPARSE_BLOCK_SIZE", DEFAULT_SPARSE_BLOCK_SIZE);
    let sparse_stream_name =
        env::var("SPECTRA_SPARSE_STREAM").unwrap_or_else(|_| "document-block".to_owned());
    let sparse_batch_size = env_usize("SPECTRA_SPARSE_BATCH_SIZE", 4096);
    let router_probe_depth = env_usize("SPECTRA_ROUTER_PROBE_DEPTH", 64);
    let router_minimum_overlap = env_f64("SPECTRA_ROUTER_MINIMUM_OVERLAP", 0.25);
    let include_router_observations = env_bool("SPECTRA_INCLUDE_ROUTER_OBSERVATIONS", true);
    let include_sparse_strategy_observations =
        env_bool("SPECTRA_INCLUDE_SPARSE_STRATEGY_OBSERVATIONS", false);
    let include_sparse_lazy_strategy_observations =
        env_bool("SPECTRA_INCLUDE_SPARSE_LAZY_STRATEGY_OBSERVATIONS", false);
    let sparse_lazy_strategy_repetitions = env_usize("SPECTRA_SPARSE_LAZY_STRATEGY_REPETITIONS", 1);
    assert!(sparse_lazy_strategy_repetitions > 0);
    let include_query_latencies = env_bool("SPECTRA_INCLUDE_QUERY_LATENCIES", false);
    let router = PrefixAgreementRouter::new(router_minimum_overlap);
    let sparse_stream_strategy = match sparse_stream_name.as_str() {
        "document-block" => SparseStreamStrategy::DocumentBlock,
        "posting-block" => SparseStreamStrategy::PostingBlock {
            batch_size: sparse_batch_size,
        },
        "adaptive-native" => SparseStreamStrategy::AdaptiveNative {
            initial_limit: env_usize("SPECTRA_SPARSE_INITIAL_LIMIT", top_k),
            growth_factor: env_usize("SPECTRA_SPARSE_GROWTH", 2),
            batch_size: sparse_batch_size,
            endpoint_trend_router: env_bool("SPECTRA_SPARSE_ENDPOINT_ROUTER", true),
        },
        "shared-full" => SparseStreamStrategy::SharedFull,
        "auto-shared" => SparseStreamStrategy::AutoShared {
            batch_size: sparse_batch_size,
            max_unique_to_logical_milli: env_usize("SPECTRA_SPARSE_SHARED_MAX_RATIO_MILLI", 750)
                .try_into()
                .expect("SPECTRA_SPARSE_SHARED_MAX_RATIO_MILLI must fit u16"),
        },
        "probe-then-shared" => SparseStreamStrategy::ProbeThenShared {
            batch_size: sparse_batch_size,
            probe_depth: env_usize("SPECTRA_SPARSE_PROBE_DEPTH", top_k),
        },
        "shared-lazy" => SparseStreamStrategy::SharedLazy {
            batch_size: sparse_batch_size,
        },
        "auto-shared-lazy" => SparseStreamStrategy::AutoSharedLazy {
            batch_size: sparse_batch_size,
        },
        value => panic!(
            "SPECTRA_SPARSE_STREAM must be document-block, posting-block, adaptive-native, shared-full, auto-shared, probe-then-shared, shared-lazy, or auto-shared-lazy; got {value}"
        ),
    };
    let dense_stream_name =
        env::var("SPECTRA_DENSE_STREAM").unwrap_or_else(|_| "replay-identical".to_owned());
    let dense_stream_strategy = match dense_stream_name.as_str() {
        "independent" => DenseStreamStrategy::Independent,
        "shared-document-major" => DenseStreamStrategy::SharedDocumentMajor,
        "replay-identical" => DenseStreamStrategy::ReplayIdentical,
        value => {
            panic!(
                "SPECTRA_DENSE_STREAM must be independent, shared-document-major, or replay-identical; got {value}"
            )
        }
    };
    let corpus: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    let mut query_rows: Vec<QueryRow> = read_jsonl(&dataset_dir.join("query-vectors.jsonl"));
    if let Ok(limit) = env::var("SPECTRA_QUERY_LIMIT") {
        query_rows.truncate(limit.parse().expect("invalid SPECTRA_QUERY_LIMIT"));
    }
    let dimension = corpus[0].dense.len();
    let source_ids: Vec<_> = corpus.iter().map(|row| row.source_id.clone()).collect();
    let qrels = read_qrels(&dataset_dir.join("qrels.tsv"));
    let dense_documents = corpus
        .iter()
        .map(|row| DenseDocument {
            id: row.ordinal,
            vector: row.dense.clone(),
        })
        .collect();
    let sparse_documents = corpus
        .into_iter()
        .map(|row| SparseDocument {
            id: row.ordinal,
            vector: RemappedSparseVector {
                indices: row.sparse_indices,
                values: row.sparse_values,
            },
        })
        .collect();
    let build_started = Instant::now();
    let index =
        NChannelExactIndex::build(dense_documents, sparse_documents, sparse_block_size).unwrap();
    let build_ns = build_started.elapsed().as_nanos();
    let query_variants: Vec<_> = query_rows.into_iter().map(QueryVariants::build).collect();
    let channel_counts = env::var("SPECTRA_CHANNEL_COUNTS")
        .unwrap_or_else(|_| "2,4,8".to_owned())
        .split(',')
        .map(|value| value.parse::<usize>().expect("invalid channel count"))
        .collect::<Vec<_>>();
    let mut matrix = Vec::new();
    let mut router_observations = Vec::new();
    let mut sparse_strategy_observations = Vec::new();
    let mut sparse_lazy_strategy_observations = Vec::new();

    for channels in channel_counts {
        assert!((1..=8).contains(&channels));
        let mut dynamic_latencies = Vec::with_capacity(query_variants.len());
        let mut exhaustive_latencies = Vec::with_capacity(query_variants.len());
        let mut router_latencies = Vec::with_capacity(query_variants.len());
        let mut dynamic_pulls = vec![0u64; channels];
        let mut exhaustive_pulls = vec![0u64; channels];
        let mut mismatches = 0;
        let mut top_k_fixed_queries = 0;
        let mut all_sources_exhausted_queries = 0;
        let mut dense_quantized_scores = 0u64;
        let mut dense_exact_scores = 0u64;
        let mut dense_document_major_passes = 0u64;
        let mut dense_document_rows_traversed = 0u64;
        let mut dense_logical_dot_products = 0u64;
        let mut dense_physical_channels = 0u64;
        let mut dense_physical_dot_products = 0u64;
        let mut dense_preparation_ns = 0u128;
        let mut sparse_documents_evaluated = 0u64;
        let mut sparse_posting_elements_visited = 0u64;
        let mut sparse_batches_expanded = 0u64;
        let mut sparse_adaptive_refills = 0u64;
        let mut sparse_results_materialized = 0u64;
        let mut sparse_shared_unique_terms = 0u64;
        let mut sparse_shared_logical_term_accesses = 0u64;
        let mut sparse_shared_physical_posting_elements = 0u64;
        let mut sparse_shared_logical_posting_elements = 0u64;
        let mut sparse_shared_preparation_ns = 0u128;
        let mut sparse_auto_selected_shared_queries = 0usize;
        let mut sparse_estimated_unique_posting_elements = 0u64;
        let mut sparse_estimated_logical_posting_elements = 0u64;
        let mut sparse_estimated_deduplicated_posting_elements = 0u64;
        let mut sparse_probe_selected_shared_queries = 0usize;
        let mut sparse_prefix_identities_replayed = 0u64;
        let mut sparse_shared_lazy_unique_batches = 0u64;
        let mut sparse_shared_lazy_reused_batches = 0u64;
        let mut sparse_shared_lazy_physical_posting_elements = 0u64;
        let mut sparse_shared_lazy_logical_posting_elements = 0u64;
        let mut sparse_shared_lazy_physical_score_multiplications = 0u64;
        let mut sparse_shared_lazy_logical_score_multiplications = 0u64;
        let mut sparse_shared_lazy_cached_channel_points = 0u64;
        let mut router_dynamic_queries = 0usize;
        let mut router_exhaustive_queries = 0usize;
        let mut router_oracle_matches = 0usize;
        let mut dynamic_oracle_queries = 0usize;
        let mut router_average_overlap_sum = 0.0;
        let mut router_minimum_overlap_sum = 0.0;
        let mut router_total_regret_ns = 0u128;
        let mut quality_metric_queries = 0usize;
        let mut recall_sum = 0.0;
        let mut ndcg_sum = 0.0;

        for (query_index, variants) in query_variants.iter().enumerate() {
            if include_sparse_lazy_strategy_observations {
                sparse_lazy_strategy_observations.extend(observe_sparse_lazy_strategies(
                    &index,
                    variants,
                    channels,
                    query_index,
                    sparse_lazy_strategy_repetitions,
                    top_k,
                    policy,
                    dense_stream_strategy,
                    sparse_batch_size,
                ));
            }
            if include_sparse_strategy_observations {
                let run = |strategy| {
                    let started = Instant::now();
                    let result = index
                        .search_with_strategies(
                            variants.channels(channels),
                            top_k,
                            DEFAULT_RRF_K,
                            None,
                            policy,
                            dense_stream_strategy,
                            strategy,
                        )
                        .unwrap();
                    (result, started.elapsed())
                };
                let (posting, posting_elapsed, shared, shared_elapsed) =
                    if query_index.is_multiple_of(2) {
                        let (posting, posting_elapsed) = run(SparseStreamStrategy::PostingBlock {
                            batch_size: sparse_batch_size,
                        });
                        let (shared, shared_elapsed) = run(SparseStreamStrategy::SharedFull);
                        (posting, posting_elapsed, shared, shared_elapsed)
                    } else {
                        let (shared, shared_elapsed) = run(SparseStreamStrategy::SharedFull);
                        let (posting, posting_elapsed) = run(SparseStreamStrategy::PostingBlock {
                            batch_size: sparse_batch_size,
                        });
                        (posting, posting_elapsed, shared, shared_elapsed)
                    };
                sparse_strategy_observations.push(SparseStrategyObservation {
                    channels,
                    query_index,
                    unique_posting_elements: shared
                        .physical
                        .sparse_shared
                        .physical_posting_elements_decoded,
                    logical_posting_elements: shared
                        .physical
                        .sparse_shared
                        .logical_posting_elements,
                    posting_ns: posting_elapsed.as_nanos(),
                    shared_ns: shared_elapsed.as_nanos(),
                    ordered_top_k_mismatch: posting.execution.point_ids
                        != shared.execution.point_ids,
                });
            }
            let probe_started = Instant::now();
            let prefixes = probe_prefixes(
                &index,
                variants,
                channels,
                router_probe_depth,
                sparse_batch_size,
            );
            let decision = router.decide(&prefixes);
            let probe_elapsed = probe_started.elapsed();
            let run_dynamic = || {
                let started = Instant::now();
                let result = match scheduler {
                    SchedulerSelection::SafeRouter => index
                        .search_with_safe_router_policy(
                            variants.channels(channels),
                            top_k,
                            DEFAULT_RRF_K,
                            None,
                            policy,
                        )
                        .unwrap(),
                    SchedulerSelection::MaxNext | SchedulerSelection::CompetitorCost => index
                        .search_with_strategies(
                            variants.channels(channels),
                            top_k,
                            DEFAULT_RRF_K,
                            None,
                            policy,
                            dense_stream_strategy,
                            sparse_stream_strategy,
                        )
                        .unwrap(),
                };
                (result, started.elapsed())
            };
            let run_exhaustive = || exhaustive(&index, variants, channels, top_k);
            let (dynamic, dynamic_elapsed, expected, exhaustive_elapsed, full_pulls) =
                if query_index.is_multiple_of(2) {
                    let (dynamic, elapsed) = run_dynamic();
                    let (expected, full_elapsed, full_pulls) = run_exhaustive();
                    (dynamic, elapsed, expected, full_elapsed, full_pulls)
                } else {
                    let (expected, full_elapsed, full_pulls) = run_exhaustive();
                    let (dynamic, elapsed) = run_dynamic();
                    (dynamic, elapsed, expected, full_elapsed, full_pulls)
                };

            let (selected_elapsed, alternative_elapsed) = match decision.choice {
                PrefixExecutionChoice::Dynamic => {
                    router_dynamic_queries += 1;
                    (dynamic_elapsed, exhaustive_elapsed)
                }
                PrefixExecutionChoice::Exhaustive => {
                    router_exhaustive_queries += 1;
                    (exhaustive_elapsed, dynamic_elapsed)
                }
            };
            let routed_elapsed = probe_elapsed + selected_elapsed;
            router_oracle_matches += usize::from(selected_elapsed <= alternative_elapsed);
            dynamic_oracle_queries += usize::from(dynamic_elapsed <= exhaustive_elapsed);
            router_average_overlap_sum += decision.average_pairwise_overlap;
            router_minimum_overlap_sum += decision.minimum_pairwise_overlap;
            router_total_regret_ns += selected_elapsed
                .saturating_sub(dynamic_elapsed.min(exhaustive_elapsed))
                .as_nanos();
            if include_router_observations {
                router_observations.push(RouterObservation {
                    channels,
                    query_index,
                    average_prefix_overlap: decision.average_pairwise_overlap,
                    minimum_prefix_overlap: decision.minimum_pairwise_overlap,
                    probe_ns: probe_elapsed.as_nanos(),
                    dynamic_ns: dynamic_elapsed.as_nanos(),
                    exhaustive_ns: exhaustive_elapsed.as_nanos(),
                });
            }

            mismatches += usize::from(dynamic.execution.point_ids != expected);
            if let Some(relevance) = qrels.get(&variants.source_id) {
                let (recall, ndcg) =
                    retrieval_quality(&dynamic.execution.point_ids, &source_ids, relevance, top_k);
                quality_metric_queries += 1;
                recall_sum += recall;
                ndcg_sum += ndcg;
            }
            top_k_fixed_queries +=
                usize::from(dynamic.execution.stop_reason == DynamicRrfStopReason::TopKFixed);
            all_sources_exhausted_queries += usize::from(
                dynamic.execution.stop_reason == DynamicRrfStopReason::AllSourcesExhausted,
            );
            for (target, value) in dynamic_pulls
                .iter_mut()
                .zip(&dynamic.execution.source_pulls)
            {
                *target += *value as u64;
            }
            for (target, value) in exhaustive_pulls.iter_mut().zip(full_pulls) {
                *target += value as u64;
            }
            for telemetry in dynamic.channels {
                match telemetry {
                    ExactChannelTelemetry::Dense(telemetry) => {
                        dense_quantized_scores += telemetry.quantized_scores as u64;
                        dense_exact_scores += telemetry.exact_scores as u64;
                    }
                    ExactChannelTelemetry::Sparse(telemetry) => {
                        sparse_documents_evaluated += telemetry.documents_evaluated as u64;
                    }
                    ExactChannelTelemetry::SparsePosting(telemetry) => {
                        sparse_posting_elements_visited +=
                            telemetry.posting_elements_visited as u64;
                        sparse_batches_expanded += telemetry.batches_expanded as u64;
                    }
                    ExactChannelTelemetry::SparseAdaptive(telemetry) => {
                        sparse_posting_elements_visited +=
                            telemetry.posting_elements_visited as u64;
                        sparse_adaptive_refills += telemetry.refills as u64;
                        sparse_results_materialized += telemetry.results_materialized as u64;
                    }
                    ExactChannelTelemetry::SparseShared(_) => {}
                    ExactChannelTelemetry::SparseSharedPosting(_) => {}
                }
            }
            dense_document_major_passes += dynamic.physical.dense_document_major_passes as u64;
            dense_document_rows_traversed += dynamic.physical.dense_document_rows_traversed as u64;
            dense_logical_dot_products += dynamic.physical.dense_logical_dot_products as u64;
            dense_physical_channels += dynamic.physical.dense_physical_channels as u64;
            dense_physical_dot_products += dynamic.physical.dense_physical_dot_products as u64;
            dense_preparation_ns += dynamic.physical.dense_preparation_ns;
            sparse_shared_unique_terms += dynamic.physical.sparse_shared.unique_terms as u64;
            sparse_shared_logical_term_accesses +=
                dynamic.physical.sparse_shared.logical_term_accesses as u64;
            sparse_shared_physical_posting_elements += dynamic
                .physical
                .sparse_shared
                .physical_posting_elements_decoded
                as u64;
            sparse_shared_logical_posting_elements +=
                dynamic.physical.sparse_shared.logical_posting_elements as u64;
            sparse_shared_preparation_ns += dynamic.physical.sparse_shared_preparation_ns;
            sparse_auto_selected_shared_queries +=
                usize::from(dynamic.physical.sparse_auto_selected_shared);
            sparse_estimated_unique_posting_elements +=
                dynamic.physical.sparse_estimated_unique_posting_elements as u64;
            sparse_estimated_logical_posting_elements +=
                dynamic.physical.sparse_estimated_logical_posting_elements as u64;
            sparse_estimated_deduplicated_posting_elements += dynamic
                .physical
                .sparse_estimated_deduplicated_posting_elements
                as u64;
            sparse_probe_selected_shared_queries +=
                usize::from(dynamic.physical.sparse_probe_selected_shared);
            sparse_prefix_identities_replayed +=
                dynamic.physical.sparse_prefix_identities_replayed as u64;
            sparse_shared_lazy_unique_batches += dynamic
                .physical
                .sparse_shared_lazy
                .unique_batch_materializations
                as u64;
            sparse_shared_lazy_reused_batches += dynamic
                .physical
                .sparse_shared_lazy
                .reused_batch_materializations
                as u64;
            sparse_shared_lazy_physical_posting_elements += dynamic
                .physical
                .sparse_shared_lazy
                .physical_posting_elements_decoded
                as u64;
            sparse_shared_lazy_logical_posting_elements += dynamic
                .physical
                .sparse_shared_lazy
                .logical_posting_elements_decoded
                as u64;
            sparse_shared_lazy_physical_score_multiplications += dynamic
                .physical
                .sparse_shared_lazy
                .physical_score_multiplications
                as u64;
            sparse_shared_lazy_logical_score_multiplications += dynamic
                .physical
                .sparse_shared_lazy
                .logical_score_multiplications
                as u64;
            sparse_shared_lazy_cached_channel_points +=
                dynamic.physical.sparse_shared_lazy.cached_channel_points as u64;
            dynamic_latencies.push(dynamic_elapsed);
            exhaustive_latencies.push(exhaustive_elapsed);
            router_latencies.push(routed_elapsed);
        }

        let dynamic_total: u64 = dynamic_pulls.iter().sum();
        let exhaustive_total: u64 = exhaustive_pulls.iter().sum();
        let dynamic_latency_ns_by_query = include_query_latencies
            .then(|| dynamic_latencies.iter().map(Duration::as_nanos).collect());
        matrix.push(ChannelMatrixRow {
            channels,
            queries: query_variants.len(),
            ordered_top_k_mismatches: mismatches,
            top_k_fixed_queries,
            all_sources_exhausted_queries,
            dynamic_source_pulls: dynamic_pulls,
            exhaustive_source_pulls: exhaustive_pulls,
            source_pull_ratio: dynamic_total as f64 / exhaustive_total as f64,
            dynamic_dense_quantized_scores: dense_quantized_scores,
            dynamic_dense_exact_scores: dense_exact_scores,
            dynamic_dense_document_major_passes: dense_document_major_passes,
            dynamic_dense_document_rows_traversed: dense_document_rows_traversed,
            dynamic_dense_logical_dot_products: dense_logical_dot_products,
            dynamic_dense_physical_channels: dense_physical_channels,
            dynamic_dense_physical_dot_products: dense_physical_dot_products,
            dynamic_dense_preparation_ns: dense_preparation_ns,
            dynamic_sparse_documents_evaluated: sparse_documents_evaluated,
            dynamic_sparse_posting_elements_visited: sparse_posting_elements_visited,
            dynamic_sparse_batches_expanded: sparse_batches_expanded,
            dynamic_sparse_adaptive_refills: sparse_adaptive_refills,
            dynamic_sparse_results_materialized: sparse_results_materialized,
            dynamic_sparse_shared_unique_terms: sparse_shared_unique_terms,
            dynamic_sparse_shared_logical_term_accesses: sparse_shared_logical_term_accesses,
            dynamic_sparse_shared_physical_posting_elements:
                sparse_shared_physical_posting_elements,
            dynamic_sparse_shared_logical_posting_elements: sparse_shared_logical_posting_elements,
            dynamic_sparse_shared_preparation_ns: sparse_shared_preparation_ns,
            dynamic_sparse_auto_selected_shared_queries: sparse_auto_selected_shared_queries,
            dynamic_sparse_estimated_unique_posting_elements:
                sparse_estimated_unique_posting_elements,
            dynamic_sparse_estimated_logical_posting_elements:
                sparse_estimated_logical_posting_elements,
            dynamic_sparse_estimated_deduplicated_posting_elements:
                sparse_estimated_deduplicated_posting_elements,
            dynamic_sparse_probe_selected_shared_queries: sparse_probe_selected_shared_queries,
            dynamic_sparse_prefix_identities_replayed: sparse_prefix_identities_replayed,
            dynamic_sparse_shared_lazy_unique_batches: sparse_shared_lazy_unique_batches,
            dynamic_sparse_shared_lazy_reused_batches: sparse_shared_lazy_reused_batches,
            dynamic_sparse_shared_lazy_physical_posting_elements:
                sparse_shared_lazy_physical_posting_elements,
            dynamic_sparse_shared_lazy_logical_posting_elements:
                sparse_shared_lazy_logical_posting_elements,
            dynamic_sparse_shared_lazy_physical_score_multiplications:
                sparse_shared_lazy_physical_score_multiplications,
            dynamic_sparse_shared_lazy_logical_score_multiplications:
                sparse_shared_lazy_logical_score_multiplications,
            dynamic_sparse_shared_lazy_cached_channel_points:
                sparse_shared_lazy_cached_channel_points,
            router_dynamic_queries,
            router_exhaustive_queries,
            router_oracle_matches,
            dynamic_oracle_queries,
            router_average_prefix_overlap: router_average_overlap_sum / query_variants.len() as f64,
            router_minimum_prefix_overlap: router_minimum_overlap_sum / query_variants.len() as f64,
            router_total_regret_ns,
            quality_metric_queries,
            recall_at_k: (quality_metric_queries > 0)
                .then_some(recall_sum / quality_metric_queries as f64),
            ndcg_at_k: (quality_metric_queries > 0)
                .then_some(ndcg_sum / quality_metric_queries as f64),
            dynamic_latency: summarize(&mut dynamic_latencies),
            exhaustive_latency: summarize(&mut exhaustive_latencies),
            router_simulated_latency: summarize(&mut router_latencies),
            dynamic_latency_ns_by_query,
        });
    }

    let result = ExperimentResult {
        schema_version: 2,
        experiment: "spectra-n-channel-scifact-v2",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        dataset_dir: dataset_dir.display().to_string(),
        documents: index.document_count(),
        dimension,
        top_k,
        rrf_k: DEFAULT_RRF_K,
        warmup_check_interval,
        exhaustive_after_pulls_per_source,
        scheduler: scheduler.name(),
        sparse_stream_strategy: sparse_stream_name,
        dense_stream_strategy: dense_stream_name,
        router_probe_depth,
        router_minimum_overlap,
        include_router_observations,
        include_sparse_strategy_observations,
        include_sparse_lazy_strategy_observations,
        sparse_lazy_strategy_repetitions,
        include_query_latencies,
        build_ns,
        matrix,
        router_observations,
        sparse_strategy_observations,
        sparse_lazy_strategy_observations,
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert!(
        result
            .matrix
            .iter()
            .all(|row| row.ordered_top_k_mismatches == 0)
    );
}

#[allow(clippy::too_many_arguments)]
fn observe_sparse_lazy_strategies(
    index: &NChannelExactIndex,
    variants: &QueryVariants,
    channels: usize,
    query_index: usize,
    repetitions: usize,
    top_k: usize,
    policy: DynamicRrfPolicy,
    dense_stream_strategy: DenseStreamStrategy,
    batch_size: usize,
) -> Vec<SparseLazyStrategyObservation> {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let strategies = [
        (
            "posting-block",
            SparseStreamStrategy::PostingBlock { batch_size },
        ),
        (
            "shared-lazy",
            SparseStreamStrategy::SharedLazy { batch_size },
        ),
        (
            "auto-shared-lazy",
            SparseStreamStrategy::AutoSharedLazy { batch_size },
        ),
    ];
    let mut observations = Vec::with_capacity(repetitions * strategies.len());

    for repetition in 0..repetitions {
        let permutation = PERMUTATIONS[(query_index + repetition) % PERMUTATIONS.len()];
        let mut outcomes = Vec::with_capacity(strategies.len());
        for (position, strategy_index) in permutation.into_iter().enumerate() {
            let (_, strategy) = strategies[strategy_index];
            let started = Instant::now();
            let result = index
                .search_with_strategies(
                    variants.channels(channels),
                    top_k,
                    DEFAULT_RRF_K,
                    None,
                    policy,
                    dense_stream_strategy,
                    strategy,
                )
                .unwrap();
            let elapsed = started.elapsed();
            outcomes.push((strategy_index, position, result, elapsed));
        }

        let posting_ids = outcomes
            .iter()
            .find(|(strategy_index, _, _, _)| *strategy_index == 0)
            .expect("posting strategy must run")
            .2
            .execution
            .point_ids
            .clone();
        for (strategy_index, position, result, elapsed) in outcomes {
            let physical = result.physical;
            observations.push(SparseLazyStrategyObservation {
                channels,
                query_index,
                repetition: repetition + 1,
                position: position + 1,
                strategy: strategies[strategy_index].0,
                elapsed_ns: elapsed.as_nanos(),
                source_pulls: result.execution.source_pulls.iter().sum(),
                auto_selected_shared: physical.sparse_auto_selected_shared,
                shared_lazy_physical_posting_elements: physical
                    .sparse_shared_lazy
                    .physical_posting_elements_decoded,
                shared_lazy_logical_posting_elements: physical
                    .sparse_shared_lazy
                    .logical_posting_elements_decoded,
                shared_lazy_physical_score_multiplications: physical
                    .sparse_shared_lazy
                    .physical_score_multiplications,
                shared_lazy_logical_score_multiplications: physical
                    .sparse_shared_lazy
                    .logical_score_multiplications,
                ordered_top_k_mismatch: result.execution.point_ids != posting_ids,
            });
        }
    }

    assert!(
        observations
            .iter()
            .all(|observation| !observation.ordered_top_k_mismatch)
    );
    observations
}

fn probe_prefixes(
    index: &NChannelExactIndex,
    variants: &QueryVariants,
    channels: usize,
    probe_depth: usize,
    batch_size: usize,
) -> Vec<Vec<ExtendedPointId>> {
    let pool = SearchScratchPool::new();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::new();
    variants
        .channels(channels)
        .into_iter()
        .map(|query| match query {
            ExactChannelQuery::Dense(query) => index
                .dense_index()
                .stream(query)
                .unwrap()
                .take(probe_depth)
                .map(|point| ExtendedPointId::from(u64::from(point.id)))
                .collect(),
            ExactChannelQuery::Sparse(query) => AdaptiveTopKStream::new(
                index.sparse_posting_index(),
                query,
                probe_depth,
                2,
                true,
                batch_size,
                true,
                &pool,
                &stopped,
                &hardware_counter,
            )
            .take(probe_depth)
            .map(|point| ExtendedPointId::from(u64::from(point.idx)))
            .collect(),
        })
        .collect()
}

fn exhaustive(
    index: &NChannelExactIndex,
    variants: &QueryVariants,
    channels: usize,
    top_k: usize,
) -> (Vec<ExtendedPointId>, Duration, Vec<usize>) {
    let started = Instant::now();
    let mut rankings = Vec::with_capacity(channels);
    let mut pulls = Vec::with_capacity(channels);
    for query in variants.channels(channels) {
        let ids: Vec<_> = match query {
            ExactChannelQuery::Dense(query) => index
                .dense_index()
                .stream(query)
                .unwrap()
                .map(|point| ExtendedPointId::from(u64::from(point.id)))
                .collect(),
            ExactChannelQuery::Sparse(query) => index
                .sparse_index()
                .stream(query)
                .unwrap()
                .map(|point| ExtendedPointId::from(u64::from(point.idx)))
                .collect(),
        };
        pulls.push(ids.len());
        rankings.push(ids.into_iter().map(scored_point).collect());
    }
    let point_ids = exact_rrf_scoring(rankings, DEFAULT_RRF_K, None)
        .unwrap()
        .into_iter()
        .take(top_k)
        .map(|point| point.id)
        .collect();
    (point_ids, started.elapsed(), pulls)
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

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
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
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect()
}

fn read_qrels(path: &Path) -> HashMap<String, HashMap<String, i32>> {
    if !path.exists() {
        return HashMap::new();
    }
    let mut qrels: HashMap<String, HashMap<String, i32>> = HashMap::new();
    for (line_index, line) in BufReader::new(File::open(path).unwrap())
        .lines()
        .enumerate()
    {
        let line = line.unwrap();
        if line_index == 0 && line.starts_with("query-id\t") {
            continue;
        }
        let mut fields = line.split('\t');
        let query_id = fields.next().expect("qrels query id");
        let corpus_id = fields.next().expect("qrels corpus id");
        let relevance = fields
            .next()
            .expect("qrels relevance")
            .parse::<i32>()
            .expect("invalid qrels relevance");
        qrels
            .entry(query_id.to_owned())
            .or_default()
            .insert(corpus_id.to_owned(), relevance);
    }
    qrels
}

fn retrieval_quality(
    point_ids: &[ExtendedPointId],
    source_ids: &[String],
    relevance: &HashMap<String, i32>,
    top_k: usize,
) -> (f64, f64) {
    let relevant = relevance.values().filter(|&&score| score > 0).count();
    let recall = if relevant == 0 {
        0.0
    } else {
        point_ids
            .iter()
            .take(top_k)
            .filter(|point_id| {
                relevance
                    .get(&source_ids[point_id.as_u64() as usize])
                    .is_some_and(|&score| score > 0)
            })
            .count() as f64
            / relevant as f64
    };
    let dcg = point_ids
        .iter()
        .take(top_k)
        .enumerate()
        .map(|(rank, point_id)| {
            let grade = relevance
                .get(&source_ids[point_id.as_u64() as usize])
                .copied()
                .unwrap_or(0);
            let gain = if grade > 0 {
                (2.0_f64).powi(grade) - 1.0
            } else {
                0.0
            };
            gain / ((rank + 2) as f64).log2()
        })
        .sum::<f64>();
    let mut ideal_grades: Vec<_> = relevance.values().copied().collect();
    ideal_grades.sort_unstable_by(|left, right| right.cmp(left));
    let ideal_dcg = ideal_grades
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(rank, grade)| {
            let gain = if grade > 0 {
                (2.0_f64).powi(grade) - 1.0
            } else {
                0.0
            };
            gain / ((rank + 2) as f64).log2()
        })
        .sum::<f64>();
    let ndcg = if ideal_dcg > 0.0 {
        dcg / ideal_dcg
    } else {
        0.0
    };
    (recall, ndcg)
}

#[cfg(test)]
mod quality_tests {
    use super::*;

    #[test]
    fn retrieval_quality_uses_source_identity_and_graded_ndcg() {
        let source_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let relevance = HashMap::from([("a".to_owned(), 2), ("c".to_owned(), 1)]);
        let ranking = vec![ExtendedPointId::from(2_u64), ExtendedPointId::from(0_u64)];

        let (recall, ndcg) = retrieval_quality(&ranking, &source_ids, &relevance, 2);

        assert_eq!(recall, 1.0);
        assert!(ndcg < 1.0);
        assert!(ndcg > 0.7);
    }

    #[test]
    fn retrieval_quality_treats_non_positive_judgments_as_non_relevant() {
        let source_ids = vec!["negative".to_owned(), "positive".to_owned()];
        let relevance = HashMap::from([("negative".to_owned(), -1), ("positive".to_owned(), 2)]);
        let ranking = vec![ExtendedPointId::from(0_u64), ExtendedPointId::from(1_u64)];

        let (recall, ndcg) = retrieval_quality(&ranking, &source_ids, &relevance, 2);

        assert_eq!(recall, 1.0);
        assert!(ndcg > 0.6);
        assert!(ndcg < 0.7);
    }
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
