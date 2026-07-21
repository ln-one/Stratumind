// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::env;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::counter::hardware_counter::HardwareCounterCell;
use segment::common::reciprocal_rank_fusion::{
    DEFAULT_RRF_K, DynamicRrfPolicy, DynamicRrfScheduler, exact_rrf_scoring,
};
use segment::index::dense_ball::DenseDocument;
use segment::index::n_channel_exact::{
    DenseStreamStrategy, ExactChannelQuery, ExactChannelTelemetry, NChannelExactIndex,
    SparseStreamStrategy,
};
use segment::index::n_channel_router::{
    PrefixAgreementDecision, PrefixAgreementRouter, PrefixExecutionChoice,
};
use segment::types::{ExtendedPointId, ScoredPoint};
use serde::Serialize;
use sparse::SearchScratchPool;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::adaptive_top_k_stream::AdaptiveTopKStream;
use sparse::index::block_max::SparseDocument;

const DIMENSIONS: usize = 8;
const DEFAULT_DOCUMENTS: usize = 100_000;
const DEFAULT_QUERIES: usize = 20;
const DEFAULT_TOP_K: usize = 20;

#[derive(Clone, Copy)]
enum Scenario {
    Correlated,
    Independent,
    AntiCorrelated,
    FlatTies,
}

impl Scenario {
    fn from_env() -> Self {
        match env::var("SPECTRA_SCENARIO").as_deref() {
            Ok("independent") => Self::Independent,
            Ok("anti_correlated") => Self::AntiCorrelated,
            Ok("flat_ties") => Self::FlatTies,
            Ok("correlated") | Err(_) => Self::Correlated,
            Ok(value) => panic!(
                "SPECTRA_SCENARIO must be correlated, independent, anti_correlated, or flat_ties; got {value}"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Correlated => "correlated",
            Self::Independent => "independent",
            Self::AntiCorrelated => "anti_correlated",
            Self::FlatTies => "flat_ties",
        }
    }

    fn score(self, id: u32, documents: usize, coordinate: usize) -> f32 {
        let position = (id as f32 + 0.5) / documents as f32;
        match self {
            Self::Correlated => 1.001 - position,
            Self::AntiCorrelated if coordinate.is_multiple_of(2) => 1.001 - position,
            Self::AntiCorrelated => 0.001 + position,
            Self::FlatTies => 1.0,
            Self::Independent => 0.001 + unit_hash(id, coordinate),
        }
    }
}

#[derive(Clone, Copy)]
enum ChannelFamily {
    Sparse,
    Mixed,
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

impl ChannelFamily {
    fn from_env() -> Self {
        match env::var("SPECTRA_CHANNEL_FAMILY").as_deref() {
            Ok("mixed") => Self::Mixed,
            Ok("sparse") | Err(_) => Self::Sparse,
            Ok(value) => panic!("SPECTRA_CHANNEL_FAMILY must be sparse or mixed; got {value}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Mixed => "mixed",
        }
    }
}

struct QueryVariants {
    dense: Vec<Vec<f32>>,
    sparse: Vec<RemappedSparseVector>,
}

impl QueryVariants {
    fn build(query_index: usize, sparse_term_groups: usize, dense_query_groups: usize) -> Self {
        let dense = (0..DIMENSIONS)
            .map(|channel| {
                let mut query = vec![0.0; DIMENSIONS];
                query[(channel % dense_query_groups + query_index) % DIMENSIONS] = 1.0;
                query
            })
            .collect();
        let sparse = (0..DIMENSIONS)
            .map(|channel| RemappedSparseVector {
                indices: vec![((channel % sparse_term_groups + query_index) % DIMENSIONS) as u32],
                values: vec![1.0],
            })
            .collect();
        Self { dense, sparse }
    }

    fn channels(&self, count: usize, family: ChannelFamily) -> Vec<ExactChannelQuery<'_>> {
        (0..count)
            .map(|channel| match family {
                ChannelFamily::Sparse => ExactChannelQuery::Sparse(self.sparse[channel].clone()),
                ChannelFamily::Mixed if channel.is_multiple_of(2) => {
                    ExactChannelQuery::Dense(&self.dense[channel])
                }
                ChannelFamily::Mixed => ExactChannelQuery::Sparse(self.sparse[channel].clone()),
            })
            .collect()
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
struct RankStreamSummary {
    logical_streams: usize,
    physical_streams: usize,
    shared_groups: usize,
    logical_pulls: usize,
    physical_pulls: usize,
    buffered_identities: usize,
}

#[derive(Debug, Serialize)]
struct MatrixRow {
    channels: usize,
    ordered_top_k_mismatches: usize,
    rank_stream_sharing: RankStreamSummary,
    dynamic_source_pulls: Vec<u64>,
    exhaustive_source_pulls: Vec<u64>,
    source_pull_ratio: f64,
    certification_checks: u64,
    safe_router_queries: usize,
    safe_router_competitor_scheduler_queries: usize,
    safe_router_shared_sparse_queries: usize,
    dense_quantized_scores: u64,
    dense_exact_scores: u64,
    dense_logical_dot_products: u64,
    dense_physical_dot_products: u64,
    sparse_posting_elements_visited: u64,
    sparse_batches_expanded: u64,
    sparse_shared_physical_posting_elements: u64,
    sparse_shared_logical_posting_elements: u64,
    sparse_auto_selected_shared_queries: usize,
    sparse_estimated_unique_posting_elements: u64,
    sparse_estimated_logical_posting_elements: u64,
    sparse_estimated_deduplicated_posting_elements: u64,
    sparse_probe_selected_shared_queries: usize,
    sparse_prefix_identities_replayed: u64,
    sparse_shared_lazy_unique_batches: u64,
    sparse_shared_lazy_reused_batches: u64,
    sparse_shared_lazy_physical_posting_elements: u64,
    sparse_shared_lazy_logical_posting_elements: u64,
    router_dynamic_queries: usize,
    router_exhaustive_queries: usize,
    router_average_prefix_overlap: f64,
    router_minimum_prefix_overlap: f64,
    router_ordered_top_k_mismatches: usize,
    dynamic_latency_ns_by_query: Vec<u128>,
    exhaustive_latency_ns_by_query: Vec<u128>,
    dynamic_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
    routed_latency: LatencySummary,
}

#[derive(Debug, Serialize)]
struct ExperimentResult {
    schema_version: usize,
    experiment: &'static str,
    qdrant_base_commit: &'static str,
    build_profile: &'static str,
    scenario: &'static str,
    channel_family: &'static str,
    documents: usize,
    queries: usize,
    top_k: usize,
    sparse_term_groups: usize,
    dense_query_groups: usize,
    dense_stream_strategy: &'static str,
    batch_size: usize,
    sparse_stream_strategy: &'static str,
    scheduler: &'static str,
    router_probe_depth: usize,
    router_minimum_overlap: f64,
    build_ns: u128,
    matrix: Vec<MatrixRow>,
}

fn main() {
    let documents = env_usize("SPECTRA_DOCUMENTS", DEFAULT_DOCUMENTS);
    let queries = env_usize("SPECTRA_QUERIES", DEFAULT_QUERIES);
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let sparse_term_groups = env_usize("SPECTRA_SPARSE_TERM_GROUPS", DIMENSIONS);
    assert!((1..=DIMENSIONS).contains(&sparse_term_groups));
    let dense_query_groups = env_usize("SPECTRA_DENSE_QUERY_GROUPS", DIMENSIONS);
    assert!((1..=DIMENSIONS).contains(&dense_query_groups));
    let batch_size = env_usize("SPECTRA_SPARSE_BATCH_SIZE", 4096);
    let (sparse_stream_strategy, sparse_stream_name) = match env::var("SPECTRA_SPARSE_STREAM")
        .as_deref()
    {
        Ok("shared-full") => (SparseStreamStrategy::SharedFull, "shared-full"),
        Ok("auto-shared") => (
            SparseStreamStrategy::AutoShared {
                batch_size,
                max_unique_to_logical_milli: 750,
            },
            "auto-shared",
        ),
        Ok("probe-then-shared") => (
            SparseStreamStrategy::ProbeThenShared {
                batch_size,
                probe_depth: env_usize("SPECTRA_SPARSE_PROBE_DEPTH", top_k),
            },
            "probe-then-shared",
        ),
        Ok("shared-lazy") => (
            SparseStreamStrategy::SharedLazy { batch_size },
            "shared-lazy",
        ),
        Ok("auto-shared-lazy") => (
            SparseStreamStrategy::AutoSharedLazy { batch_size },
            "auto-shared-lazy",
        ),
        Ok("posting-block") | Err(_) => (
            SparseStreamStrategy::PostingBlock { batch_size },
            "posting-block",
        ),
        Ok(value) => {
            panic!(
                "SPECTRA_SPARSE_STREAM must be posting-block, shared-full, auto-shared, probe-then-shared, shared-lazy, or auto-shared-lazy; got {value}"
            )
        }
    };
    let (dense_stream_strategy, dense_stream_name) = match env::var("SPECTRA_DENSE_STREAM")
        .as_deref()
    {
        Ok("independent") => (DenseStreamStrategy::Independent, "independent"),
        Ok("shared-document-major") => (
            DenseStreamStrategy::SharedDocumentMajor,
            "shared-document-major",
        ),
        Ok("replay-identical") | Err(_) => {
            (DenseStreamStrategy::ReplayIdentical, "replay-identical")
        }
        Ok(value) => panic!(
            "SPECTRA_DENSE_STREAM must be independent, shared-document-major, or replay-identical; got {value}"
        ),
    };
    let router_probe_depth = env_usize("SPECTRA_ROUTER_PROBE_DEPTH", 64);
    let router_minimum_overlap = env_f64("SPECTRA_ROUTER_MINIMUM_OVERLAP", 0.25);
    let router = PrefixAgreementRouter::new(router_minimum_overlap);
    let scenario = Scenario::from_env();
    let family = ChannelFamily::from_env();
    let scheduler = SchedulerSelection::from_env();
    let channel_counts = env::var("SPECTRA_CHANNEL_COUNTS")
        .unwrap_or_else(|_| "2,4,8".to_owned())
        .split(',')
        .map(|value| value.parse::<usize>().expect("invalid channel count"))
        .collect::<Vec<_>>();
    assert!(documents > 0 && queries > 0 && top_k > 0);
    assert!(channel_counts.iter().all(|count| (1..=8).contains(count)));

    let (dense_documents, sparse_documents) = build_documents(documents, scenario);
    let build_started = Instant::now();
    let index = NChannelExactIndex::build(dense_documents, sparse_documents, 128).unwrap();
    let build_ns = build_started.elapsed().as_nanos();
    let query_variants: Vec<_> = (0..queries)
        .map(|query| QueryVariants::build(query, sparse_term_groups, dense_query_groups))
        .collect();
    let policy = DynamicRrfPolicy {
        scheduler: scheduler.policy_scheduler(),
        exhaustive_after_pulls_per_source: None,
        ..DynamicRrfPolicy::default()
    };
    let mut matrix = Vec::new();

    for channels in channel_counts {
        let mut dynamic_latencies = Vec::with_capacity(queries);
        let mut exhaustive_latencies = Vec::with_capacity(queries);
        let mut routed_latencies = Vec::with_capacity(queries);
        let mut dynamic_pulls = vec![0u64; channels];
        let mut exhaustive_pulls = vec![0u64; channels];
        let mut mismatches = 0;
        let mut rank_logical_streams = 0usize;
        let mut rank_physical_streams = 0usize;
        let mut rank_shared_groups = 0usize;
        let mut rank_logical_pulls = 0usize;
        let mut rank_physical_pulls = 0usize;
        let mut rank_buffered_identities = 0usize;
        let mut dense_quantized_scores = 0u64;
        let mut dense_exact_scores = 0u64;
        let mut dense_logical_dot_products = 0u64;
        let mut dense_physical_dot_products = 0u64;
        let mut posting_elements_visited = 0u64;
        let mut batches_expanded = 0u64;
        let mut shared_physical_posting_elements = 0u64;
        let mut shared_logical_posting_elements = 0u64;
        let mut auto_selected_shared_queries = 0usize;
        let mut estimated_unique_posting_elements = 0u64;
        let mut estimated_logical_posting_elements = 0u64;
        let mut estimated_deduplicated_posting_elements = 0u64;
        let mut probe_selected_shared_queries = 0usize;
        let mut prefix_identities_replayed = 0u64;
        let mut shared_lazy_unique_batches = 0u64;
        let mut shared_lazy_reused_batches = 0u64;
        let mut shared_lazy_physical_posting_elements = 0u64;
        let mut shared_lazy_logical_posting_elements = 0u64;
        let mut router_dynamic_queries = 0usize;
        let mut router_exhaustive_queries = 0usize;
        let mut router_overlap_sum = 0.0;
        let mut router_minimum_overlap_sum = 0.0;
        let mut router_mismatches = 0usize;
        let mut certification_checks = 0u64;
        let mut safe_router_queries = 0usize;
        let mut safe_router_competitor_scheduler_queries = 0usize;
        let mut safe_router_shared_sparse_queries = 0usize;

        for (query_index, variants) in query_variants.iter().enumerate() {
            let run_dynamic = || {
                let started = Instant::now();
                let result = match scheduler {
                    SchedulerSelection::SafeRouter => index
                        .search_with_safe_router_policy(
                            variants.channels(channels, family),
                            top_k,
                            DEFAULT_RRF_K,
                            None,
                            policy,
                        )
                        .unwrap(),
                    SchedulerSelection::MaxNext | SchedulerSelection::CompetitorCost => index
                        .search_with_strategies(
                            variants.channels(channels, family),
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
            let run_exhaustive = || exhaustive(&index, variants, channels, family, top_k);
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
            let (routed_ids, routed_elapsed, decision) = run_routed(
                &index,
                variants,
                channels,
                family,
                top_k,
                batch_size,
                router_probe_depth,
                router,
                policy,
            );
            mismatches += usize::from(dynamic.execution.point_ids != expected);
            router_mismatches += usize::from(routed_ids != expected);
            router_overlap_sum += decision.average_pairwise_overlap;
            router_minimum_overlap_sum += decision.minimum_pairwise_overlap;
            match decision.choice {
                PrefixExecutionChoice::Dynamic => router_dynamic_queries += 1,
                PrefixExecutionChoice::Exhaustive => router_exhaustive_queries += 1,
            }
            for (target, value) in dynamic_pulls
                .iter_mut()
                .zip(&dynamic.execution.source_pulls)
            {
                *target += *value as u64;
            }
            certification_checks += dynamic.execution.certification_checks as u64;
            safe_router_queries += usize::from(dynamic.physical.safe_router_requested);
            safe_router_competitor_scheduler_queries += usize::from(
                dynamic.physical.safe_router_requested
                    && dynamic.physical.scheduler == DynamicRrfScheduler::CompetitorCostAware,
            );
            safe_router_shared_sparse_queries +=
                usize::from(dynamic.physical.safe_router_selected_shared_sparse);
            for (target, value) in exhaustive_pulls.iter_mut().zip(full_pulls) {
                *target += value as u64;
            }
            for telemetry in dynamic.channels {
                match telemetry {
                    ExactChannelTelemetry::Dense(value) => {
                        dense_quantized_scores += value.quantized_scores as u64;
                        dense_exact_scores += value.exact_scores as u64;
                    }
                    ExactChannelTelemetry::SparsePosting(value) => {
                        posting_elements_visited += value.posting_elements_visited as u64;
                        batches_expanded += value.batches_expanded as u64;
                    }
                    ExactChannelTelemetry::Sparse(_) | ExactChannelTelemetry::SparseAdaptive(_) => {
                        unreachable!()
                    }
                    ExactChannelTelemetry::SparseShared(_) => {}
                    ExactChannelTelemetry::SparseSharedPosting(_) => {}
                }
            }
            shared_physical_posting_elements += dynamic
                .physical
                .sparse_shared
                .physical_posting_elements_decoded
                as u64;
            shared_logical_posting_elements +=
                dynamic.physical.sparse_shared.logical_posting_elements as u64;
            auto_selected_shared_queries +=
                usize::from(dynamic.physical.sparse_auto_selected_shared);
            estimated_unique_posting_elements +=
                dynamic.physical.sparse_estimated_unique_posting_elements as u64;
            estimated_logical_posting_elements +=
                dynamic.physical.sparse_estimated_logical_posting_elements as u64;
            estimated_deduplicated_posting_elements += dynamic
                .physical
                .sparse_estimated_deduplicated_posting_elements
                as u64;
            rank_logical_streams += dynamic.physical.logical_rank_streams;
            rank_physical_streams += dynamic.physical.physical_rank_streams;
            rank_shared_groups += dynamic.physical.shared_rank_stream_groups;
            rank_logical_pulls += dynamic.physical.logical_rank_pulls;
            rank_physical_pulls += dynamic.physical.physical_rank_pulls;
            rank_buffered_identities += dynamic.physical.rank_replay_buffered_identities;
            dense_logical_dot_products += dynamic.physical.dense_logical_dot_products as u64;
            dense_physical_dot_products += dynamic.physical.dense_physical_dot_products as u64;
            probe_selected_shared_queries +=
                usize::from(dynamic.physical.sparse_probe_selected_shared);
            prefix_identities_replayed += dynamic.physical.sparse_prefix_identities_replayed as u64;
            shared_lazy_unique_batches += dynamic
                .physical
                .sparse_shared_lazy
                .unique_batch_materializations as u64;
            shared_lazy_reused_batches += dynamic
                .physical
                .sparse_shared_lazy
                .reused_batch_materializations as u64;
            shared_lazy_physical_posting_elements += dynamic
                .physical
                .sparse_shared_lazy
                .physical_posting_elements_decoded
                as u64;
            shared_lazy_logical_posting_elements += dynamic
                .physical
                .sparse_shared_lazy
                .logical_posting_elements_decoded
                as u64;
            dynamic_latencies.push(dynamic_elapsed);
            exhaustive_latencies.push(exhaustive_elapsed);
            routed_latencies.push(routed_elapsed);
        }

        let dynamic_total: u64 = dynamic_pulls.iter().sum();
        let exhaustive_total: u64 = exhaustive_pulls.iter().sum();
        let dynamic_latency_ns_by_query =
            dynamic_latencies.iter().map(Duration::as_nanos).collect();
        let exhaustive_latency_ns_by_query = exhaustive_latencies
            .iter()
            .map(Duration::as_nanos)
            .collect();
        matrix.push(MatrixRow {
            channels,
            ordered_top_k_mismatches: mismatches,
            rank_stream_sharing: RankStreamSummary {
                logical_streams: rank_logical_streams,
                physical_streams: rank_physical_streams,
                shared_groups: rank_shared_groups,
                logical_pulls: rank_logical_pulls,
                physical_pulls: rank_physical_pulls,
                buffered_identities: rank_buffered_identities,
            },
            dynamic_source_pulls: dynamic_pulls,
            exhaustive_source_pulls: exhaustive_pulls,
            source_pull_ratio: dynamic_total as f64 / exhaustive_total as f64,
            certification_checks,
            safe_router_queries,
            safe_router_competitor_scheduler_queries,
            safe_router_shared_sparse_queries,
            dense_quantized_scores,
            dense_exact_scores,
            dense_logical_dot_products,
            dense_physical_dot_products,
            sparse_posting_elements_visited: posting_elements_visited,
            sparse_batches_expanded: batches_expanded,
            sparse_shared_physical_posting_elements: shared_physical_posting_elements,
            sparse_shared_logical_posting_elements: shared_logical_posting_elements,
            sparse_auto_selected_shared_queries: auto_selected_shared_queries,
            sparse_estimated_unique_posting_elements: estimated_unique_posting_elements,
            sparse_estimated_logical_posting_elements: estimated_logical_posting_elements,
            sparse_estimated_deduplicated_posting_elements: estimated_deduplicated_posting_elements,
            sparse_probe_selected_shared_queries: probe_selected_shared_queries,
            sparse_prefix_identities_replayed: prefix_identities_replayed,
            sparse_shared_lazy_unique_batches: shared_lazy_unique_batches,
            sparse_shared_lazy_reused_batches: shared_lazy_reused_batches,
            sparse_shared_lazy_physical_posting_elements: shared_lazy_physical_posting_elements,
            sparse_shared_lazy_logical_posting_elements: shared_lazy_logical_posting_elements,
            router_dynamic_queries,
            router_exhaustive_queries,
            router_average_prefix_overlap: router_overlap_sum / queries as f64,
            router_minimum_prefix_overlap: router_minimum_overlap_sum / queries as f64,
            router_ordered_top_k_mismatches: router_mismatches,
            dynamic_latency_ns_by_query,
            exhaustive_latency_ns_by_query,
            dynamic_latency: summarize(&mut dynamic_latencies),
            exhaustive_latency: summarize(&mut exhaustive_latencies),
            routed_latency: summarize(&mut routed_latencies),
        });
    }

    let result = ExperimentResult {
        schema_version: 1,
        experiment: "spectra-n-channel-synthetic-v1",
        qdrant_base_commit: "44ad62f8cd69642be5afa6441612525e24a0d063",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        scenario: scenario.name(),
        channel_family: family.name(),
        documents,
        queries,
        top_k,
        sparse_term_groups,
        dense_query_groups,
        dense_stream_strategy: dense_stream_name,
        batch_size,
        sparse_stream_strategy: sparse_stream_name,
        scheduler: scheduler.name(),
        router_probe_depth,
        router_minimum_overlap,
        build_ns,
        matrix,
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert!(
        result
            .matrix
            .iter()
            .all(|row| row.ordered_top_k_mismatches == 0)
    );
    assert!(
        result
            .matrix
            .iter()
            .all(|row| row.router_ordered_top_k_mismatches == 0)
    );
}

#[expect(clippy::too_many_arguments)]
fn run_routed(
    index: &NChannelExactIndex,
    variants: &QueryVariants,
    channels: usize,
    family: ChannelFamily,
    top_k: usize,
    batch_size: usize,
    probe_depth: usize,
    router: PrefixAgreementRouter,
    policy: DynamicRrfPolicy,
) -> (Vec<ExtendedPointId>, Duration, PrefixAgreementDecision) {
    let started = Instant::now();
    let prefixes = probe_prefixes(index, variants, channels, family, probe_depth, batch_size);
    let decision = router.decide(&prefixes);
    let ids = match decision.choice {
        PrefixExecutionChoice::Dynamic => {
            index
                .search_with_sparse_strategy(
                    variants.channels(channels, family),
                    top_k,
                    DEFAULT_RRF_K,
                    None,
                    policy,
                    SparseStreamStrategy::PostingBlock { batch_size },
                )
                .unwrap()
                .execution
                .point_ids
        }
        PrefixExecutionChoice::Exhaustive => exhaustive(index, variants, channels, family, top_k).0,
    };
    (ids, started.elapsed(), decision)
}

fn probe_prefixes(
    index: &NChannelExactIndex,
    variants: &QueryVariants,
    channels: usize,
    family: ChannelFamily,
    probe_depth: usize,
    batch_size: usize,
) -> Vec<Vec<ExtendedPointId>> {
    let pool = SearchScratchPool::new();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::new();
    variants
        .channels(channels, family)
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

fn build_documents(
    documents: usize,
    scenario: Scenario,
) -> (Vec<DenseDocument>, Vec<SparseDocument>) {
    let mut dense = Vec::with_capacity(documents);
    let mut sparse = Vec::with_capacity(documents);
    for id in 0..documents as u32 {
        let values: Vec<_> = (0..DIMENSIONS)
            .map(|coordinate| scenario.score(id, documents, coordinate))
            .collect();
        dense.push(DenseDocument {
            id,
            vector: values.clone(),
        });
        sparse.push(SparseDocument {
            id,
            vector: RemappedSparseVector {
                indices: (0..DIMENSIONS as u32).collect(),
                values,
            },
        });
    }
    (dense, sparse)
}

fn exhaustive(
    index: &NChannelExactIndex,
    variants: &QueryVariants,
    channels: usize,
    family: ChannelFamily,
    top_k: usize,
) -> (Vec<ExtendedPointId>, Duration, Vec<usize>) {
    let started = Instant::now();
    let mut rankings = Vec::with_capacity(channels);
    let mut pulls = Vec::with_capacity(channels);
    for query in variants.channels(channels, family) {
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

fn unit_hash(id: u32, coordinate: usize) -> f32 {
    let mut value = u64::from(id) ^ ((coordinate as u64 + 1) * 0x9e37_79b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value as u32) as f32 / u32::MAX as f32
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
