// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Isolates the N-channel fusion and shared-Node layer from channel-specific scoring.
//! The benchmark measures exactness and physical identity materialization, not ANN quality.

use std::env;
use std::time::{Duration, Instant};

use common::types::PointOffsetType;
use ordered_float::OrderedFloat;
use segment::common::reciprocal_rank_fusion::{DEFAULT_RRF_K, exact_rrf_scoring};
use segment::index::shared_node::{
    SharedChannelPlan, SharedNodeCatalog, execute_shared_dynamic_rrf,
};
use segment::types::{ExtendedPointId, ScoredPoint};
use serde::Serialize;

const DEFAULT_DOCUMENTS: usize = 20_000;
const DEFAULT_QUERIES: usize = 50;
const DEFAULT_TOP_K: usize = 20;
const DEFAULT_BLOCK_SIZE: usize = 128;

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
                "SPECTRA_SCENARIO must be correlated, independent, anti_correlated, or flat_ties; got {value}",
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
    scenario: &'static str,
    documents: usize,
    queries: usize,
    channels: usize,
    top_k: usize,
    block_size: usize,
    parity_mismatches: usize,
    fixed_before_exhaustion: usize,
    source_pulls: Vec<u64>,
    logical_node_expansions: u64,
    unique_physical_node_materializations: u64,
    reused_node_materializations: u64,
    unique_identity_materializations: u64,
    dynamic_latency: LatencySummary,
    exhaustive_latency: LatencySummary,
}

fn main() {
    let documents = env_usize("SPECTRA_DOCUMENTS", DEFAULT_DOCUMENTS);
    let queries = env_usize("SPECTRA_QUERIES", DEFAULT_QUERIES);
    let channels = env_usize("SPECTRA_CHANNELS", 4);
    let top_k = env_usize("SPECTRA_TOP_K", DEFAULT_TOP_K);
    let block_size = env_usize("SPECTRA_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    let scenario = Scenario::from_env();
    assert!(documents > 0);
    assert!(queries > 0);
    assert!(channels > 0);
    assert!(top_k > 0);

    let catalog = SharedNodeCatalog::build(
        (0..documents)
            .map(|id| PointOffsetType::try_from(id).expect("document count exceeds u32"))
            .collect(),
        block_size,
    )
    .unwrap();
    let mut dynamic_latencies = Vec::with_capacity(queries);
    let mut exhaustive_latencies = Vec::with_capacity(queries);
    let mut parity_mismatches = 0usize;
    let mut fixed_before_exhaustion = 0usize;
    let mut source_pulls = vec![0u64; channels];
    let mut logical_node_expansions = 0u64;
    let mut unique_physical_node_materializations = 0u64;
    let mut reused_node_materializations = 0u64;
    let mut unique_identity_materializations = 0u64;

    for query in 0..queries {
        let scores = generate_scores(documents, channels, query, scenario);

        let exhaustive_start = Instant::now();
        let expected = exhaustive_rrf(&scores, top_k);
        exhaustive_latencies.push(exhaustive_start.elapsed());

        let dynamic_start = Instant::now();
        let plans = scores
            .iter()
            .map(|channel_scores| {
                let node_upper_bounds = (0..catalog.node_count())
                    .map(|node| {
                        catalog.node_range(node).map(|range| {
                            channel_scores[range]
                                .iter()
                                .copied()
                                .fold(f64::NEG_INFINITY, f64::max)
                        })
                    })
                    .collect();
                SharedChannelPlan {
                    node_upper_bounds,
                    score: Box::new(move |ordinal, _| Ok(Some(channel_scores[ordinal]))),
                }
            })
            .collect();
        let actual =
            execute_shared_dynamic_rrf(&catalog, plans, top_k, DEFAULT_RRF_K, None).unwrap();
        dynamic_latencies.push(dynamic_start.elapsed());

        parity_mismatches += usize::from(actual.fusion.point_ids != expected);
        fixed_before_exhaustion += usize::from(matches!(
            actual.fusion.stop_reason,
            segment::common::reciprocal_rank_fusion::DynamicRrfStopReason::TopKFixed
        ));
        for (total, pulls) in source_pulls.iter_mut().zip(actual.fusion.source_pulls) {
            *total += pulls as u64;
        }
        logical_node_expansions += actual
            .channels
            .iter()
            .map(|channel| channel.nodes_expanded as u64)
            .sum::<u64>();
        unique_physical_node_materializations +=
            actual.physical.unique_node_materializations as u64;
        reused_node_materializations += actual.physical.reused_node_materializations as u64;
        unique_identity_materializations += actual.physical.unique_identity_materializations as u64;
    }

    let row = ResultRow {
        schema_version: 1,
        experiment: "spectra-shared-n-channel-v1",
        scenario: scenario.name(),
        documents,
        queries,
        channels,
        top_k,
        block_size,
        parity_mismatches,
        fixed_before_exhaustion,
        source_pulls,
        logical_node_expansions,
        unique_physical_node_materializations,
        reused_node_materializations,
        unique_identity_materializations,
        dynamic_latency: summarize(&mut dynamic_latencies),
        exhaustive_latency: summarize(&mut exhaustive_latencies),
    };
    println!("{}", serde_json::to_string_pretty(&row).unwrap());
    assert_eq!(parity_mismatches, 0);
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn generate_scores(
    documents: usize,
    channels: usize,
    query: usize,
    scenario: Scenario,
) -> Vec<Vec<f64>> {
    (0..channels)
        .map(|channel| {
            (0..documents)
                .map(|document| {
                    let common = mix((document as u64) ^ ((query as u64) << 32)) % 1_000_003;
                    let private =
                        mix((document as u64) ^ ((query as u64) << 24) ^ ((channel as u64) << 48))
                            % 1_000_003;
                    let score = match scenario {
                        Scenario::Correlated => common * 16 + private % 16,
                        Scenario::Independent => private,
                        Scenario::AntiCorrelated if channel % 2 == 0 => common,
                        Scenario::AntiCorrelated => 1_000_003 - common,
                        Scenario::FlatTies => (common / 50_000) * 50_000,
                    };
                    score as f64
                })
                .collect()
        })
        .collect()
}

fn exhaustive_rrf(scores: &[Vec<f64>], top_k: usize) -> Vec<ExtendedPointId> {
    let rankings = scores
        .iter()
        .map(|channel| {
            let mut ranking: Vec<_> = channel.iter().copied().enumerate().collect();
            ranking.sort_unstable_by(|left, right| {
                OrderedFloat(right.1)
                    .cmp(&OrderedFloat(left.1))
                    .then_with(|| left.0.cmp(&right.0))
            });
            ranking
                .into_iter()
                .map(|(id, _)| ScoredPoint {
                    id: ExtendedPointId::from(id as u64),
                    version: 0,
                    score: 0.0,
                    payload: None,
                    vector: None,
                    shard_key: None,
                    order_value: None,
                })
                .collect()
        })
        .collect();
    exact_rrf_scoring(rankings, DEFAULT_RRF_K, None)
        .unwrap()
        .into_iter()
        .take(top_k)
        .map(|point| point.id)
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
