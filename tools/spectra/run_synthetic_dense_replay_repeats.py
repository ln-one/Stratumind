#!/usr/bin/env python3
"""Repeat the exact Dense identical-query replay ablation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--dense-query-groups", type=int, default=1)
    parser.add_argument(
        "--strategies", default="shared-document-major,replay-identical"
    )
    parser.add_argument("--candidate", default="replay-identical")
    parser.add_argument("--baseline", default="shared-document-major")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_721)
    return parser.parse_args()


def strategies(args: argparse.Namespace) -> list[str]:
    return [value for value in args.strategies.split(",") if value]


def order(args: argparse.Namespace, repetition: int) -> list[str]:
    values = strategies(args)
    return values if repetition % 2 == 0 else list(reversed(values))


def run_once(args: argparse.Namespace, strategy: str) -> dict[str, Any]:
    environment = os.environ | {
        "SPECTRA_DOCUMENTS": str(args.documents),
        "SPECTRA_QUERIES": str(args.queries),
        "SPECTRA_TOP_K": "20",
        "SPECTRA_CHANNEL_COUNTS": str(args.channels),
        "SPECTRA_CHANNEL_FAMILY": "mixed",
        "SPECTRA_DENSE_QUERY_GROUPS": str(args.dense_query_groups),
        "SPECTRA_DENSE_STREAM": strategy,
        "SPECTRA_SPARSE_TERM_GROUPS": "8",
        "SPECTRA_SPARSE_STREAM": "posting-block",
        "SPECTRA_SPARSE_BATCH_SIZE": "4096",
    }
    completed = subprocess.run(
        [str(args.binary)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    if sum(row["ordered_top_k_mismatches"] for row in result["matrix"]):
        raise RuntimeError(f"{strategy} violated exhaustive WRRF ordering")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def bootstrap_query_cluster_ci(
    query_deltas: dict[int, list[int]], samples: int, seed: int
) -> tuple[int, int]:
    query_means = [statistics.mean(values) for values in query_deltas.values()]
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(generator.choice(query_means) for _ in query_means))
    return round(percentile(means, 0.025)), round(percentile(means, 0.975))


def summary(args: argparse.Namespace, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for strategy in strategies(args):
        samples = [
            run["result"]["matrix"][0]
            for run in runs
            if run["strategy"] == strategy
        ]
        logical = sum(row["dense_logical_dot_products"] for row in samples)
        physical = sum(row["dense_physical_dot_products"] for row in samples)
        rows.append(
            {
                "strategy": strategy,
                "repetitions": len(samples),
                "ordered_top_k_mismatches": sum(
                    row["ordered_top_k_mismatches"] for row in samples
                ),
                "median_run_p50_ns": round(
                    statistics.median(row["dynamic_latency"]["p50_ns"] for row in samples)
                ),
                "median_run_p95_ns": round(
                    statistics.median(row["dynamic_latency"]["p95_ns"] for row in samples)
                ),
                "dense_physical_to_logical_dot_ratio": physical / logical,
                "rank_physical_to_logical_pull_ratio": sum(
                    row["rank_stream_sharing"]["physical_pulls"] for row in samples
                )
                / sum(row["rank_stream_sharing"]["logical_pulls"] for row in samples),
            }
        )
    return rows


def paired_analysis(
    runs: list[dict[str, Any]], candidate_name: str, baseline_name: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    values: dict[tuple[str, int, int], int] = {}
    for run in runs:
        latencies = run["result"]["matrix"][0]["dynamic_latency_ns_by_query"]
        for query, latency in enumerate(latencies):
            values[(run["strategy"], run["repetition"], query)] = latency
    deltas = []
    ratios = []
    query_deltas: dict[int, list[int]] = {}
    for repetition in range(1, max(run["repetition"] for run in runs) + 1):
        for query in range(len(runs[0]["result"]["matrix"][0]["dynamic_latency_ns_by_query"])):
            candidate = values[(candidate_name, repetition, query)]
            baseline = values[(baseline_name, repetition, query)]
            delta = candidate - baseline
            deltas.append(delta)
            ratios.append(candidate / baseline)
            query_deltas.setdefault(query, []).append(delta)
    low, high = bootstrap_query_cluster_ci(query_deltas, bootstrap_samples, seed)
    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "paired_observations": len(deltas),
        "query_clusters": len(query_deltas),
        "mean_delta_ns": round(statistics.mean(deltas)),
        "mean_delta_query_cluster_bootstrap_95_ci_ns": [low, high],
        "median_delta_ns": round(statistics.median(deltas)),
        "p95_delta_ns": round(percentile(deltas, 0.95)),
        "median_latency_ratio": statistics.median(ratios),
        "candidate_win_rate": sum(delta < 0 for delta in deltas) / len(deltas),
    }


def main() -> None:
    args = parse_args()
    if args.candidate not in strategies(args) or args.baseline not in strategies(args):
        raise ValueError("candidate and baseline must be listed in --strategies")
    for repetition in range(args.warmup_repetitions):
        for strategy in order(args, repetition):
            run_once(args, strategy)

    runs = []
    for repetition in range(args.repetitions):
        for position, strategy in enumerate(order(args, repetition)):
            print(
                f"repetition={repetition + 1} position={position + 1} strategy={strategy}",
                flush=True,
            )
            runs.append(
                {
                    "repetition": repetition + 1,
                    "position": position + 1,
                    "strategy": strategy,
                    "result": run_once(args, strategy),
                }
            )

    artifact = {
        "schema_version": 1,
        "experiment": "spectra-synthetic-dense-replay-repeats-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform()},
        "contract": "both strategies must equal exhaustive full-corpus WRRF ordered Top-K",
        "counterbalancing": "alternate strategy order by repetition",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": summary(args, runs),
        "paired_analysis": {
            "bootstrap": "query-cluster bootstrap of mean paired delta",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "comparison": paired_analysis(
                runs,
                args.candidate,
                args.baseline,
                args.bootstrap_samples,
                args.seed,
            ),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
