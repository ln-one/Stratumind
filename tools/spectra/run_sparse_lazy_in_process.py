#!/usr/bin/env python3
"""Compare exact Sparse stream strategies after loading one snapshot once."""

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


STRATEGIES = ("posting-block", "shared-lazy", "auto-shared-lazy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--channel-counts", default="4,8")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dense-stream", default="replay-identical")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_721)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for observation in observations:
        groups.setdefault(
            (observation["channels"], observation["strategy"]), []
        ).append(observation)
    result = []
    for (channels, strategy), rows in sorted(groups.items()):
        latencies = [row["elapsed_ns"] for row in rows]
        result.append(
            {
                "channels": channels,
                "strategy": strategy,
                "observations": len(rows),
                "ordered_top_k_mismatches": sum(
                    row["ordered_top_k_mismatch"] for row in rows
                ),
                "median_ns": round(statistics.median(latencies)),
                "p95_ns": round(percentile(latencies, 0.95)),
                "mean_ns": round(statistics.mean(latencies)),
                "source_pulls": sum(row["source_pulls"] for row in rows),
                "auto_selected_shared": sum(
                    row["auto_selected_shared"] for row in rows
                ),
                "shared_lazy_physical_posting_elements": sum(
                    row["shared_lazy_physical_posting_elements"] for row in rows
                ),
                "shared_lazy_logical_posting_elements": sum(
                    row["shared_lazy_logical_posting_elements"] for row in rows
                ),
                "shared_lazy_physical_score_multiplications": sum(
                    row["shared_lazy_physical_score_multiplications"] for row in rows
                ),
                "shared_lazy_logical_score_multiplications": sum(
                    row["shared_lazy_logical_score_multiplications"] for row in rows
                ),
                "position_counts": {
                    str(position): sum(row["position"] == position for row in rows)
                    for position in range(1, 4)
                },
            }
        )
    return result


def paired_analysis(
    observations: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    values = {
        (
            row["channels"],
            row["query_index"],
            row["repetition"],
            row["strategy"],
        ): row["elapsed_ns"]
        for row in observations
    }
    channels = sorted({row["channels"] for row in observations})
    analyses = []
    for channel_count in channels:
        query_ids = sorted(
            {
                row["query_index"]
                for row in observations
                if row["channels"] == channel_count
            }
        )
        repetitions = sorted(
            {
                row["repetition"]
                for row in observations
                if row["channels"] == channel_count
            }
        )
        for candidate in STRATEGIES[1:]:
            deltas = []
            query_means = []
            for query_id in query_ids:
                query_deltas = []
                for repetition in repetitions:
                    delta = values[
                        (channel_count, query_id, repetition, candidate)
                    ] - values[
                        (channel_count, query_id, repetition, "posting-block")
                    ]
                    deltas.append(delta)
                    query_deltas.append(delta)
                query_means.append(statistics.mean(query_deltas))
            generator = random.Random(seed + channel_count + len(analyses))
            bootstrap_means = [
                statistics.mean(generator.choice(query_means) for _ in query_means)
                for _ in range(bootstrap_samples)
            ]
            analyses.append(
                {
                    "channels": channel_count,
                    "candidate": candidate,
                    "baseline": "posting-block",
                    "paired_observations": len(deltas),
                    "query_clusters": len(query_means),
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
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    environment = os.environ.copy()
    environment.update(
        {
            "SPECTRA_DATASET_DIR": str(args.dataset_dir),
            "SPECTRA_QUERY_LIMIT": str(args.query_limit),
            "SPECTRA_CHANNEL_COUNTS": args.channel_counts,
            "SPECTRA_SPARSE_STREAM": "posting-block",
            "SPECTRA_SPARSE_BATCH_SIZE": str(args.batch_size),
            "SPECTRA_DENSE_STREAM": args.dense_stream,
            "SPECTRA_INCLUDE_ROUTER_OBSERVATIONS": "false",
            "SPECTRA_INCLUDE_SPARSE_STRATEGY_OBSERVATIONS": "false",
            "SPECTRA_INCLUDE_SPARSE_LAZY_STRATEGY_OBSERVATIONS": "true",
            "SPECTRA_SPARSE_LAZY_STRATEGY_REPETITIONS": str(args.repetitions),
            "SPECTRA_INCLUDE_QUERY_LATENCIES": "false",
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
            f"experiment failed with exit code {error.returncode}\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    kernel_result = json.loads(completed.stdout)
    observations = kernel_result["sparse_lazy_strategy_observations"]
    mismatches = sum(row["ordered_top_k_mismatch"] for row in observations)
    if mismatches:
        raise RuntimeError(f"strategy observations produced {mismatches} mismatches")
    artifact = {
        "schema_version": 1,
        "experiment": "spectra-sparse-lazy-in-process-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_contract": "training-free and qrels-blind",
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "configuration": {
            "binary": str(args.binary),
            "dataset_dir": str(args.dataset_dir),
            "strategies": STRATEGIES,
            "repetitions": args.repetitions,
            "query_limit": args.query_limit,
            "channel_counts": args.channel_counts,
            "batch_size": args.batch_size,
            "dense_stream": args.dense_stream,
            "counterbalancing": "six permutations rotated by query and repetition",
            "single_process_single_index_build": True,
        },
        "summary": summarize(observations),
        "paired_analysis": {
            "bootstrap": "query-cluster bootstrap of mean paired delta",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "comparisons": paired_analysis(
                observations, args.bootstrap_samples, args.seed
            ),
        },
        "kernel_matrix": kernel_result["matrix"],
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
