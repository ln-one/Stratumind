#!/usr/bin/env python3
"""Analyze paired query latency from an executor repetition artifact."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="posting-block")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_721)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def bootstrap_query_cluster_ci(
    query_deltas: dict[int, list[int]], samples: int, seed: int
) -> tuple[int, int]:
    query_means = [statistics.mean(values) for values in query_deltas.values()]
    generator = random.Random(seed)
    bootstrap_means = []
    for _ in range(samples):
        resample = [generator.choice(query_means) for _ in query_means]
        bootstrap_means.append(statistics.mean(resample))
    return (
        round(percentile(bootstrap_means, 0.025)),
        round(percentile(bootstrap_means, 0.975)),
    )


def observations(artifact: dict[str, Any]) -> dict[tuple[str, int, int, int], int]:
    values = {}
    for run in artifact["runs"]:
        strategy = run["strategy"]
        repetition = run["repetition"]
        for row in run["result"]["router_observations"]:
            key = (strategy, row["channels"], repetition, row["query_index"])
            values[key] = row["dynamic_ns"]
    return values


def analyze(
    artifact: dict[str, Any], baseline: str, bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    values = observations(artifact)
    strategies = artifact["configuration"]["strategies"]
    channels = sorted({key[1] for key in values})
    comparisons = []
    for candidate in strategies:
        if candidate == baseline:
            continue
        for channel_count in channels:
            deltas = []
            ratios = []
            query_deltas: dict[int, list[int]] = {}
            candidate_keys = [
                key
                for key in values
                if key[0] == candidate and key[1] == channel_count
            ]
            for _, _, repetition, query_index in candidate_keys:
                candidate_ns = values[(candidate, channel_count, repetition, query_index)]
                baseline_ns = values[(baseline, channel_count, repetition, query_index)]
                delta = candidate_ns - baseline_ns
                deltas.append(delta)
                ratios.append(candidate_ns / baseline_ns)
                query_deltas.setdefault(query_index, []).append(delta)
            ci_low, ci_high = bootstrap_query_cluster_ci(
                query_deltas, bootstrap_samples, seed + channel_count
            )
            comparisons.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "channels": channel_count,
                    "paired_observations": len(deltas),
                    "query_clusters": len(query_deltas),
                    "mean_delta_ns": round(statistics.mean(deltas)),
                    "mean_delta_query_cluster_bootstrap_95_ci_ns": [ci_low, ci_high],
                    "median_delta_ns": round(statistics.median(deltas)),
                    "p95_delta_ns": round(percentile(deltas, 0.95)),
                    "median_latency_ratio": statistics.median(ratios),
                    "candidate_win_rate": sum(delta < 0 for delta in deltas) / len(deltas),
                }
            )
    return comparisons


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    mismatches = sum(
        row["ordered_top_k_mismatches"]
        for run in artifact["runs"]
        for row in run["result"]["matrix"]
    )
    if mismatches:
        raise RuntimeError(f"artifact contains {mismatches} ordered Top-K mismatches")
    artifact["paired_analysis"] = {
        "contract": "paired by repetition, channel count, and query index",
        "bootstrap": "query-cluster bootstrap of mean delta",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "comparisons": analyze(
            artifact, args.baseline, args.bootstrap_samples, args.seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
