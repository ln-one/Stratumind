#!/usr/bin/env python3
"""Repeat the compact exact Dense certificate against Qdrant native baselines."""

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
    parser.add_argument("--certificate-binary", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hnsw-ef", type=int, default=256)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construct", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_721)
    return parser.parse_args()


def run_binary(binary: Path, environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary)],
        env=os.environ | environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_certificate(args: argparse.Namespace) -> dict[str, Any]:
    result = run_binary(
        args.certificate_binary,
        {
            "SPECTRA_DATASET_DIR": str(args.dataset_dir),
            "SPECTRA_QUERY_LIMIT": str(args.query_limit),
            "SPECTRA_TOP_K": str(args.top_k),
        },
    )
    if result["parity_mismatches"]:
        raise RuntimeError("compact certificate violated exhaustive ordering")
    return result


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    result = run_binary(
        args.native_binary,
        {
            "SPECTRA_DATASET_DIR": str(args.dataset_dir),
            "SPECTRA_QUERY_LIMIT": str(args.query_limit),
            "SPECTRA_TOP_K": str(args.top_k),
            "SPECTRA_HNSW_EF": str(args.hnsw_ef),
            "SPECTRA_HNSW_M": str(args.hnsw_m),
            "SPECTRA_HNSW_EF_CONSTRUCT": str(args.hnsw_ef_construct),
            "SPECTRA_PROPOSAL_K": str(args.top_k),
            "SPECTRA_CERTIFICATE_BATCH_SIZE": str(args.top_k),
            # One leaf keeps the rejected Ball path from dominating gate setup.
            "SPECTRA_DENSE_LEAF_SIZE": "1000000000",
            "SPECTRA_DENSE_BRANCH_FACTOR": "8",
            "SPECTRA_DENSE_CLUSTERING_ITERATIONS": "1",
        },
    )
    if result["certified_ordered_top_k_mismatches"]:
        raise RuntimeError("native Scalar certificate violated exhaustive ordering")
    return result


def order(repetition: int) -> list[str]:
    values = ["compact-certificate", "qdrant-native"]
    return values if repetition % 2 == 0 else list(reversed(values))


def run_condition(args: argparse.Namespace, condition: str) -> dict[str, Any]:
    if condition == "compact-certificate":
        return run_certificate(args)
    if condition == "qdrant-native":
        return run_native(args)
    raise ValueError(condition)


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
        bootstrap_means.append(
            statistics.mean(generator.choice(query_means) for _ in query_means)
        )
    return (
        round(percentile(bootstrap_means, 0.025)),
        round(percentile(bootstrap_means, 0.975)),
    )


def latency_observations(runs: list[dict[str, Any]]) -> dict[tuple[str, int, int], int]:
    values: dict[tuple[str, int, int], int] = {}
    for run in runs:
        repetition = run["repetition"]
        result = run["result"]
        if run["condition"] == "compact-certificate":
            for query, latency in enumerate(result["certified_latency_ns_by_query"]):
                values[("compact-certificate", repetition, query)] = latency
        else:
            for query, latency in enumerate(result["exact_scan_latency_ns_by_query"]):
                values[("qdrant-exact", repetition, query)] = latency
            for query, latency in enumerate(result["hnsw_latency_ns_by_query"]):
                values[("qdrant-hnsw", repetition, query)] = latency
    return values


def compare(
    values: dict[tuple[str, int, int], int],
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas = []
    ratios = []
    query_deltas: dict[int, list[int]] = {}
    candidate_keys = [key for key in values if key[0] == candidate]
    for _, repetition, query in candidate_keys:
        candidate_ns = values[(candidate, repetition, query)]
        baseline_ns = values[(baseline, repetition, query)]
        delta = candidate_ns - baseline_ns
        deltas.append(delta)
        ratios.append(candidate_ns / baseline_ns)
        query_deltas.setdefault(query, []).append(delta)
    low, high = bootstrap_query_cluster_ci(query_deltas, bootstrap_samples, seed)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "paired_observations": len(deltas),
        "query_clusters": len(query_deltas),
        "mean_delta_ns": round(statistics.mean(deltas)),
        "mean_delta_query_cluster_bootstrap_95_ci_ns": [low, high],
        "median_delta_ns": round(statistics.median(deltas)),
        "p95_delta_ns": round(percentile(deltas, 0.95)),
        "median_latency_ratio": statistics.median(ratios),
        "candidate_win_rate": sum(delta < 0 for delta in deltas) / len(deltas),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    certificate = [
        run["result"] for run in runs if run["condition"] == "compact-certificate"
    ]
    native = [run["result"] for run in runs if run["condition"] == "qdrant-native"]
    return {
        "compact_certificate": {
            "ordered_top_k_mismatches": sum(row["parity_mismatches"] for row in certificate),
            "median_run_p50_ns": round(
                statistics.median(row["certified_latency"]["p50_ns"] for row in certificate)
            ),
            "median_run_p95_ns": round(
                statistics.median(row["certified_latency"]["p95_ns"] for row in certificate)
            ),
            "mean_exact_score_ratio": statistics.mean(
                row["exact_score_ratio"] for row in certificate
            ),
        },
        "qdrant_exact": {
            "ordered_top_k_mismatches": 0,
            "median_run_p50_ns": round(
                statistics.median(row["exact_scan_latency"]["p50_ns"] for row in native)
            ),
            "median_run_p95_ns": round(
                statistics.median(row["exact_scan_latency"]["p95_ns"] for row in native)
            ),
        },
        "qdrant_hnsw": {
            "ordered_top_k_mismatches": sum(row["ordered_top_k_mismatches"] for row in native),
            "mean_recall_at_k": statistics.mean(row["mean_recall_at_k"] for row in native),
            "median_run_p50_ns": round(
                statistics.median(row["hnsw_latency"]["p50_ns"] for row in native)
            ),
            "median_run_p95_ns": round(
                statistics.median(row["hnsw_latency"]["p95_ns"] for row in native)
            ),
        },
    }


def main() -> None:
    args = parse_args()
    for repetition in range(args.warmup_repetitions):
        for condition in order(repetition):
            run_condition(args, condition)

    runs = []
    for repetition in range(args.repetitions):
        for position, condition in enumerate(order(repetition)):
            print(
                f"repetition={repetition + 1} position={position + 1} condition={condition}",
                flush=True,
            )
            runs.append(
                {
                    "repetition": repetition + 1,
                    "position": position + 1,
                    "condition": condition,
                    "result": run_condition(args, condition),
                }
            )

    values = latency_observations(runs)
    artifact = {
        "schema_version": 1,
        "experiment": "spectra-dense-exact-repeats-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform()},
        "contract": "compact certificate and Qdrant exact must be ordered-exact; HNSW is approximate",
        "counterbalancing": "alternate process order by repetition",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": summarize_runs(runs),
        "paired_analysis": {
            "bootstrap": "query-cluster bootstrap of mean paired delta",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "comparisons": [
                compare(
                    values,
                    "compact-certificate",
                    "qdrant-exact",
                    args.bootstrap_samples,
                    args.seed,
                ),
                compare(
                    values,
                    "compact-certificate",
                    "qdrant-hnsw",
                    args.bootstrap_samples,
                    args.seed + 1,
                ),
            ],
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
