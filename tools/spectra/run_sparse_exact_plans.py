#!/usr/bin/env python3
"""Repeated matched benchmark for exact Sparse physical plans."""

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
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "SPECTRA_DATASET_DIR": str(args.dataset_dir),
            "SPECTRA_QUERY_LIMIT": str(args.query_limit),
            "SPECTRA_TOP_K": str(args.top_k),
            "SPECTRA_BLOCK_BATCH_SIZE": str(args.batch_size),
        }
    )
    completed = subprocess.run(
        [str(args.binary)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    if result["ordered_top_k_mismatches"]:
        raise RuntimeError("the exact Sparse plans produced different ordered Top-K results")
    return result


def percentile(values: list[float], percentile_value: float) -> float:
    values = sorted(values)
    return values[int((len(values) - 1) * percentile_value / 100)]


def cluster_bootstrap_ci(
    rows: list[list[int]], samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    query_count = len(rows[0])
    draws = []
    for _ in range(samples):
        query_ids = [rng.randrange(query_count) for _ in range(query_count)]
        values = [row[query_id] for row in rows for query_id in query_ids]
        draws.append(statistics.fmean(values))
    return percentile(draws, 2.5), percentile(draws, 97.5)


def comparison(
    runs: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    rows = []
    candidate_values = []
    baseline_values = []
    for run in runs:
        candidate = run["stream_latency_ns_by_query"]
        baseline = run["qdrant_latency_ns_by_query"]
        if len(candidate) != len(baseline):
            raise RuntimeError("paired latency arrays have different lengths")
        rows.append([left - right for left, right in zip(candidate, baseline)])
        candidate_values.extend(candidate)
        baseline_values.extend(baseline)
    values = [value for row in rows for value in row]
    ci = cluster_bootstrap_ci(rows, samples, seed)
    return {
        "candidate": "posting-stream",
        "baseline": "qdrant-search-context",
        "paired_observations": len(values),
        "query_clusters": len(rows[0]),
        "mean_delta_ns": statistics.fmean(values),
        "mean_delta_query_cluster_bootstrap_95_ci_ns": list(ci),
        "median_delta_ns": statistics.median(values),
        "p95_delta_ns": percentile(values, 95),
        "median_latency_ratio": statistics.median(candidate_values)
        / statistics.median(baseline_values),
        "candidate_win_rate": sum(value < 0 for value in values) / len(values),
    }


def repository_metadata(repository: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--short")),
    }


def main() -> None:
    args = parse_args()
    if not args.binary.is_file() or not args.dataset_dir.is_dir():
        raise FileNotFoundError("binary and dataset directory must exist")
    if min(
        args.repetitions,
        args.query_limit,
        args.top_k,
        args.batch_size,
        args.bootstrap_samples,
    ) <= 0:
        raise ValueError("experiment sizes must be positive")
    if args.warmup_repetitions < 0:
        raise ValueError("warmup repetitions cannot be negative")

    for repetition in range(args.warmup_repetitions):
        print(f"warmup={repetition + 1}", flush=True)
        run_once(args)

    runs = []
    for repetition in range(args.repetitions):
        print(f"repetition={repetition + 1}", flush=True)
        runs.append(run_once(args))

    repository = Path(__file__).resolve().parents[2]
    artifact = {
        "schema_version": 1,
        "experiment": "stratumind-sparse-exact-plans-repeats-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "both plans must return the same exact score-and-identity ordered Top-K",
        "counterbalancing": "within-query execution order alternates by query index",
        "repository": repository_metadata(repository),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "configuration": {
            "binary": str(args.binary),
            "dataset_dir": str(args.dataset_dir),
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "query_limit": args.query_limit,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "summary": {
            "ordered_top_k_mismatches": sum(
                run["ordered_top_k_mismatches"] for run in runs
            ),
            "accepted_prefixes": sum(run["accepted_prefixes"] for run in runs),
            "tied_boundary_fallbacks": sum(
                run["tied_boundary_fallbacks"] for run in runs
            ),
            "qdrant_median_run_p50_ns": statistics.median(
                run["qdrant_latency"]["p50_ns"] for run in runs
            ),
            "qdrant_median_run_p95_ns": statistics.median(
                run["qdrant_latency"]["p95_ns"] for run in runs
            ),
            "stream_median_run_p50_ns": statistics.median(
                run["stream_latency"]["p50_ns"] for run in runs
            ),
            "stream_median_run_p95_ns": statistics.median(
                run["stream_latency"]["p95_ns"] for run in runs
            ),
        },
        "paired_analysis": comparison(runs, args.bootstrap_samples, args.seed),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
