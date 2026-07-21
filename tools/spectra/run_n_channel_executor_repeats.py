#!/usr/bin/env python3
"""Run counterbalanced, qrels-blind N-channel executor repetitions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STRATEGIES = "posting-block,shared-lazy,auto-shared-lazy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--channel-counts", default="4,8")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dense-stream", default="shared-document-major")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_721)
    return parser.parse_args()


def strategy_order(strategies: list[str], repetition: int) -> list[str]:
    offset = repetition % len(strategies)
    return strategies[offset:] + strategies[:offset]


def run_once(args: argparse.Namespace, strategy: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "SPECTRA_DATASET_DIR": str(args.dataset_dir),
            "SPECTRA_QUERY_LIMIT": str(args.query_limit),
            "SPECTRA_CHANNEL_COUNTS": args.channel_counts,
            "SPECTRA_SPARSE_STREAM": strategy,
            "SPECTRA_SPARSE_BATCH_SIZE": str(args.batch_size),
            "SPECTRA_DENSE_STREAM": args.dense_stream,
            "SPECTRA_INCLUDE_ROUTER_OBSERVATIONS": "true",
            "SPECTRA_INCLUDE_SPARSE_STRATEGY_OBSERVATIONS": "false",
            "SPECTRA_INCLUDE_QUERY_LATENCIES": "true",
        }
    )
    try:
        completed = subprocess.run(
            [str(args.binary)],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{strategy} failed with exit code {error.returncode}\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    result = json.loads(completed.stdout)
    mismatches = sum(row["ordered_top_k_mismatches"] for row in result["matrix"])
    if mismatches:
        raise RuntimeError(f"{strategy} produced {mismatches} ordered Top-K mismatches")
    return result


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        strategy = run["result"]["sparse_stream_strategy"]
        for row in run["result"]["matrix"]:
            groups.setdefault((strategy, row["channels"]), []).append(row)

    summaries = []
    for (strategy, channels), rows in sorted(groups.items()):
        p50_values = [row["dynamic_latency"]["p50_ns"] for row in rows]
        p95_values = [row["dynamic_latency"]["p95_ns"] for row in rows]
        summaries.append(
            {
                "strategy": strategy,
                "channels": channels,
                "measured_repetitions": len(rows),
                "ordered_top_k_mismatches": sum(
                    row["ordered_top_k_mismatches"] for row in rows
                ),
                "median_run_p50_ns": int(statistics.median(p50_values)),
                "median_run_p95_ns": int(statistics.median(p95_values)),
                "mean_run_p50_ns": int(statistics.mean(p50_values)),
                "mean_run_p95_ns": int(statistics.mean(p95_values)),
                "total_auto_selected_shared_queries": sum(
                    row["dynamic_sparse_auto_selected_shared_queries"] for row in rows
                ),
                "total_shared_lazy_physical_posting_elements": sum(
                    row["dynamic_sparse_shared_lazy_physical_posting_elements"]
                    for row in rows
                ),
                "total_shared_lazy_logical_posting_elements": sum(
                    row["dynamic_sparse_shared_lazy_logical_posting_elements"]
                    for row in rows
                ),
            }
        )
    return summaries


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def paired_analysis(
    runs: list[dict[str, Any]],
    strategies: list[str],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(strategies) < 2:
        return []
    baseline = "posting-block" if "posting-block" in strategies else strategies[0]
    values: dict[tuple[str, int, int, int], int] = {}
    channel_counts = set()
    for run in runs:
        for row in run["result"]["matrix"]:
            channels = row["channels"]
            channel_counts.add(channels)
            for query, latency in enumerate(row["dynamic_latency_ns_by_query"]):
                values[(run["strategy"], run["repetition"], channels, query)] = latency

    analyses = []
    repetitions = max(run["repetition"] for run in runs)
    for channels in sorted(channel_counts):
        queries = len(
            next(
                row["dynamic_latency_ns_by_query"]
                for run in runs
                for row in run["result"]["matrix"]
                if row["channels"] == channels
            )
        )
        for candidate in strategies:
            if candidate == baseline:
                continue
            deltas = []
            query_deltas: dict[int, list[int]] = {}
            for repetition in range(1, repetitions + 1):
                for query in range(queries):
                    delta = (
                        values[(candidate, repetition, channels, query)]
                        - values[(baseline, repetition, channels, query)]
                    )
                    deltas.append(delta)
                    query_deltas.setdefault(query, []).append(delta)
            query_means = [statistics.mean(group) for group in query_deltas.values()]
            generator = random.Random(seed + channels + len(analyses))
            bootstrap_means = [
                statistics.mean(generator.choice(query_means) for _ in query_means)
                for _ in range(bootstrap_samples)
            ]
            analyses.append(
                {
                    "channels": channels,
                    "candidate": candidate,
                    "baseline": baseline,
                    "paired_observations": len(deltas),
                    "query_clusters": len(query_deltas),
                    "mean_delta_ns": round(statistics.mean(deltas)),
                    "mean_delta_query_cluster_bootstrap_95_ci_ns": [
                        round(percentile(bootstrap_means, 0.025)),
                        round(percentile(bootstrap_means, 0.975)),
                    ],
                    "median_delta_ns": round(statistics.median(deltas)),
                    "candidate_win_rate": sum(delta < 0 for delta in deltas)
                    / len(deltas),
                }
            )
    return analyses


def main() -> None:
    args = parse_args()
    strategies = [value for value in args.strategies.split(",") if value]
    if not strategies:
        raise ValueError("at least one strategy is required")
    if args.repetitions < 1 or args.warmup_repetitions < 0:
        raise ValueError("repetition counts are invalid")
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)

    for repetition in range(args.warmup_repetitions):
        for strategy in strategy_order(strategies, repetition):
            run_once(args, strategy)

    runs = []
    for repetition in range(args.repetitions):
        order = strategy_order(strategies, repetition)
        for position, strategy in enumerate(order):
            print(
                f"measured repetition={repetition + 1} position={position + 1} "
                f"strategy={strategy}",
                file=sys.stderr,
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
        "schema_version": 2,
        "experiment": "spectra-n-channel-executor-repeats-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_contract": (
            "training-free and qrels-blind; the runner never reads qrels to choose an executor"
        ),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "configuration": {
            "binary": str(args.binary),
            "dataset_dir": str(args.dataset_dir),
            "strategies": strategies,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "query_limit": args.query_limit,
            "channel_counts": args.channel_counts,
            "batch_size": args.batch_size,
            "dense_stream": args.dense_stream,
            "counterbalancing": "cyclic rotation",
        },
        "summary": summarize(runs),
        "paired_analysis": {
            "bootstrap": "query-cluster bootstrap of mean paired delta",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "comparisons": paired_analysis(
                runs,
                strategies,
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
