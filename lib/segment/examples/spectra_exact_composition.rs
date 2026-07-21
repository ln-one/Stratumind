// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact finite-DAG composition experiment.
//!
//! Each topology is checked against full materialization of that same logical
//! network. Agreement with a flat network is reported separately as semantic
//! drift, never as an execution-correctness oracle.

use std::cmp::Reverse;
use std::collections::HashSet;
use std::env;
use std::mem::size_of;
use std::time::{Duration, Instant};

use segment::common::reciprocal_rank_fusion::{DynamicRrfPolicy, ExactRrfStream};
use segment::index::exact_composition::ExactCompositionExecutionPolicy;
use segment::types::ExtendedPointId;
use serde::Serialize;

mod support;
use support::exact_composition::{
    BuiltPlan, IsolatedExecutionMode, IsolatedExecutionResult, Topology, build_external_plan,
    build_plan,
};

const DEFAULT_DOCUMENTS: usize = 20_000;
const DEFAULT_QUERIES: usize = 10;
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_LEAVES: usize = 4;
const DEFAULT_TOP_K: usize = 20;
const DEFAULT_RRF_K: usize = 60;

#[derive(Clone, Copy)]
enum Scenario {
    Correlated,
    Independent,
    AntiCorrelated,
    Clustered,
    FlatTies,
}

impl Scenario {
    fn from_env() -> Self {
        match env::var("SPECTRA_SCENARIO").as_deref() {
            Ok("independent") => Self::Independent,
            Ok("anti_correlated") => Self::AntiCorrelated,
            Ok("clustered") => Self::Clustered,
            Ok("flat_ties") => Self::FlatTies,
            Ok("correlated") | Err(_) => Self::Correlated,
            Ok(value) => panic!(
                "SPECTRA_SCENARIO must be correlated, independent, anti_correlated, clustered, or flat_ties; got {value}",
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Correlated => "correlated",
            Self::Independent => "independent",
            Self::AntiCorrelated => "anti_correlated",
            Self::Clustered => "clustered",
            Self::FlatTies => "flat_ties",
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
struct ExperimentResult {
    schema_version: usize,
    experiment: &'static str,
    build_profile: &'static str,
    scenario: &'static str,
    topology: &'static str,
    semantic_oracle: &'static str,
    documents: usize,
    queries: usize,
    repeats: usize,
    leaves: usize,
    physical_leaf_nodes: usize,
    fusion_nodes: usize,
    top_k: usize,
    rrf_k: usize,
    internal_limit: usize,
    exhaustive_after_pulls_per_input: Option<usize>,
    identity_payload_bytes: usize,
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
    peak_replay_payload_bytes: usize,
    peak_exhaustive_materialized_identities: usize,
    peak_exhaustive_payload_bytes: usize,
    plan_build_latency: LatencySummary,
    dynamic_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
}

fn main() {
    let documents = env_usize("SPECTRA_DOCUMENTS", DEFAULT_DOCUMENTS);
    let queries = env_usize("SPECTRA_QUERIES", DEFAULT_QUERIES);
    let repeats = env_usize("SPECTRA_REPEATS", DEFAULT_REPEATS);
    let leaves = env_usize("SPECTRA_LEAVES", DEFAULT_LEAVES);
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let rrf_k = env_usize("SPECTRA_RRF_K", DEFAULT_RRF_K);
    let internal_limit = env_usize("SPECTRA_INTERNAL_LIMIT", top_k.saturating_mul(4));
    let exhaustive_after_pulls_per_input =
        env_optional_usize("SPECTRA_EXHAUSTIVE_AFTER_PULLS_PER_INPUT");
    let execution_policy = ExactCompositionExecutionPolicy {
        fusion_policy: DynamicRrfPolicy {
            exhaustive_after_pulls_per_source: exhaustive_after_pulls_per_input,
            ..DynamicRrfPolicy::default()
        },
    };
    let scenario = Scenario::from_env();
    let topology = Topology::parse(env::var("SPECTRA_TOPOLOGY").as_deref().unwrap_or("flat"));
    assert!(documents > 0 && queries > 0 && repeats > 0);
    assert!(leaves >= 2 && top_k > 0 && rrf_k > 0 && internal_limit > 0);
    if matches!(
        topology,
        Topology::DiamondShared | Topology::DiamondUnfolded
    ) {
        assert!(
            leaves >= 4,
            "diamond experiments require at least four leaves"
        );
    }
    if let Some(mode) = IsolatedExecutionMode::from_env() {
        run_isolated(
            mode,
            documents,
            queries,
            repeats,
            leaves,
            top_k,
            rrf_k,
            internal_limit,
            scenario,
            topology,
            execution_policy,
        );
        return;
    }

    let mut build_latencies = Vec::with_capacity(queries);
    let mut dynamic_latencies = Vec::with_capacity(queries * repeats);
    let mut exhaustive_latencies = Vec::with_capacity(queries * repeats);
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
    let mut physical_leaf_nodes = 0;
    let mut fusion_nodes = 0;

    for query in 0..queries {
        let orders = generate_orders(documents, leaves, query, scenario);
        let flat_reference = build_plan(Topology::Flat, &orders, top_k, rrf_k, internal_limit)
            .plan
            .exhaustive_root(top_k)
            .unwrap();

        let build_start = Instant::now();
        let oracle = build_plan(topology, &orders, top_k, rrf_k, internal_limit);
        let built = build_external_plan(topology, orders.len(), top_k, rrf_k, internal_limit);
        build_latencies.push(build_start.elapsed());
        physical_leaf_nodes = built.physical_leaf_sources.len();
        fusion_nodes = built.fusion_nodes;

        let semantic_output = oracle.plan.exhaustive_root(top_k).unwrap();
        flat_order_drift_queries += usize::from(semantic_output != flat_reference);
        flat_overlap_sum += overlap_at_k(&semantic_output, &flat_reference, top_k);

        for repeat in 0..repeats {
            let (actual, exhaustive) = if (query + repeat).is_multiple_of(2) {
                let dynamic_start = Instant::now();
                let actual = built
                    .plan
                    .execute_with_leaf_streams_and_policy(
                        top_k,
                        materialized_streams(&built, &orders),
                        execution_policy,
                    )
                    .unwrap();
                dynamic_latencies.push(dynamic_start.elapsed());
                let exhaustive_start = Instant::now();
                let exhaustive = built
                    .plan
                    .exhaustive_with_leaf_streams(top_k, materialized_streams(&built, &orders))
                    .unwrap();
                exhaustive_latencies.push(exhaustive_start.elapsed());
                (actual, exhaustive)
            } else {
                let exhaustive_start = Instant::now();
                let exhaustive = built
                    .plan
                    .exhaustive_with_leaf_streams(top_k, materialized_streams(&built, &orders))
                    .unwrap();
                exhaustive_latencies.push(exhaustive_start.elapsed());
                let dynamic_start = Instant::now();
                let actual = built
                    .plan
                    .execute_with_leaf_streams_and_policy(
                        top_k,
                        materialized_streams(&built, &orders),
                        execution_policy,
                    )
                    .unwrap();
                dynamic_latencies.push(dynamic_start.elapsed());
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

    let identity_payload_bytes = size_of::<ExtendedPointId>();
    let result = ExperimentResult {
        schema_version: 1,
        experiment: "spectra-exact-composition-v1",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        scenario: scenario.name(),
        topology: topology.name(),
        semantic_oracle: "full materialization of the same logical network",
        documents,
        queries,
        repeats,
        leaves,
        physical_leaf_nodes,
        fusion_nodes,
        top_k,
        rrf_k,
        internal_limit,
        exhaustive_after_pulls_per_input,
        identity_payload_bytes,
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
        peak_replay_payload_bytes: peak_replay_identities * identity_payload_bytes,
        peak_exhaustive_materialized_identities,
        peak_exhaustive_payload_bytes: peak_exhaustive_materialized_identities
            * identity_payload_bytes,
        plan_build_latency: summarize(&mut build_latencies),
        dynamic_latency: summarize(&mut dynamic_latencies),
        exhaustive_latency: summarize(&mut exhaustive_latencies),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    assert_eq!(ordered_top_k_mismatches, 0);
}

#[allow(clippy::too_many_arguments)]
fn run_isolated(
    mode: IsolatedExecutionMode,
    documents: usize,
    queries: usize,
    repeats: usize,
    leaves: usize,
    top_k: usize,
    rrf_k: usize,
    internal_limit: usize,
    scenario: Scenario,
    topology: Topology,
    policy: ExactCompositionExecutionPolicy,
) {
    let start = Instant::now();
    let mut point_ids_returned = 0;
    let mut leaf_identities_consumed = 0;
    let mut intermediate_identities = 0;
    for query in 0..queries {
        let orders = generate_orders(documents, leaves, query, scenario);
        let built = build_external_plan(topology, orders.len(), top_k, rrf_k, internal_limit);
        for _ in 0..repeats {
            match mode {
                IsolatedExecutionMode::Dynamic => {
                    let execution = built
                        .plan
                        .execute_with_leaf_streams_and_policy(
                            top_k,
                            materialized_streams(&built, &orders),
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
                        .exhaustive_with_leaf_streams(top_k, materialized_streams(&built, &orders))
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
        experiment: "spectra-exact-composition-isolated-v1",
        execution_mode: mode.name(),
        operations: queries * repeats,
        point_ids_returned,
        leaf_identities_consumed,
        intermediate_identities,
        elapsed_ns: start.elapsed().as_nanos(),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn materialized_streams(
    built: &BuiltPlan,
    orders: &[Vec<ExtendedPointId>],
) -> Vec<Option<ExactRrfStream<'static>>> {
    built
        .physical_leaf_sources
        .iter()
        .map(|&source| {
            Some(Box::new(orders[source].clone().into_iter().map(Ok)) as ExactRrfStream<'static>)
        })
        .chain(std::iter::repeat_with(|| None).take(built.fusion_nodes))
        .collect()
}

fn generate_orders(
    documents: usize,
    leaves: usize,
    query: usize,
    scenario: Scenario,
) -> Vec<Vec<ExtendedPointId>> {
    (0..leaves)
        .map(|channel| {
            let mut scores: Vec<_> = (0..documents)
                .map(|document| {
                    let common = mix((document as u64) ^ ((query as u64) << 32));
                    let private =
                        mix((document as u64) ^ ((query as u64) << 24) ^ ((channel as u64) << 48));
                    let clustered = mix((document as u64)
                        ^ ((query as u64) << 28)
                        ^ (((channel / 2) as u64) << 44));
                    let score = match scenario {
                        Scenario::Correlated => common & !0xff | private & 0xff,
                        Scenario::Independent => private,
                        Scenario::AntiCorrelated if channel.is_multiple_of(2) => common,
                        Scenario::AntiCorrelated => u64::MAX - common,
                        Scenario::Clustered => clustered,
                        Scenario::FlatTies => 1,
                    };
                    (Reverse(score), document)
                })
                .collect();
            scores.sort_unstable();
            scores
                .into_iter()
                .map(|(_, document)| ExtendedPointId::from(document as u64))
                .collect()
        })
        .collect()
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

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
