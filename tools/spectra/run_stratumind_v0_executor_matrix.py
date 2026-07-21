#!/usr/bin/env python3
"""Counterbalanced Stratumind V0 scheduler matrix over the native executor."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCENARIOS = "correlated,independent,anti_correlated,flat_ties"
DEFAULT_SCHEDULERS = "max-next,competitor-cost,safe-router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--schedulers", default=DEFAULT_SCHEDULERS)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def comma_values(value: str, name: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def rotated(values: list[str], offset: int) -> list[str]:
    pivot = offset % len(values)
    return values[pivot:] + values[:pivot]


def run_once(args: argparse.Namespace, scenario: str, scheduler: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "SPECTRA_SCENARIO": scenario,
            "SPECTRA_CHANNEL_FAMILY": "mixed",
            "SPECTRA_CHANNEL_COUNTS": "2",
            "SPECTRA_DOCUMENTS": str(args.documents),
            "SPECTRA_QUERIES": str(args.queries),
            "SPECTRA_TOP_K": str(args.top_k),
            "SPECTRA_SPARSE_STREAM": "posting-block",
            "SPECTRA_SPARSE_BATCH_SIZE": str(args.batch_size),
            "SPECTRA_DENSE_STREAM": "independent",
            "STRATUMIND_SCHEDULER": scheduler,
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
            f"scenario={scenario} scheduler={scheduler} failed with {error.returncode}\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    result = json.loads(completed.stdout)
    if result.get("scheduler") != scheduler:
        raise RuntimeError("binary did not apply the requested scheduler")
    rows = result.get("matrix", [])
    if len(rows) != 1 or rows[0].get("channels") != 2:
        raise RuntimeError("V0 matrix must contain exactly the Dense + Sparse row")
    row = rows[0]
    mismatches = row["ordered_top_k_mismatches"] + row["router_ordered_top_k_mismatches"]
    if mismatches:
        raise RuntimeError(
            f"scenario={scenario} scheduler={scheduler} produced {mismatches} mismatches"
        )
    return result


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["scenario"], run["scheduler"])
        groups.setdefault(key, []).append(run["result"]["matrix"][0])

    summaries = []
    for (scenario, scheduler), rows in sorted(groups.items()):
        pull_ratios = [row["source_pull_ratio"] for row in rows]
        summaries.append(
            {
                "scenario": scenario,
                "scheduler": scheduler,
                "measured_repetitions": len(rows),
                "ordered_top_k_mismatches": sum(
                    row["ordered_top_k_mismatches"] for row in rows
                ),
                "median_source_pull_ratio": statistics.median(pull_ratios),
                "median_dynamic_source_pull_ratio": statistics.median(pull_ratios),
                "min_source_pull_ratio": min(pull_ratios),
                "max_source_pull_ratio": max(pull_ratios),
                "median_dense_source_pulls": int(
                    statistics.median(row["dynamic_source_pulls"][0] for row in rows)
                ),
                "median_sparse_source_pulls": int(
                    statistics.median(row["dynamic_source_pulls"][1] for row in rows)
                ),
                "median_exhaustive_dense_pulls": int(
                    statistics.median(row["exhaustive_source_pulls"][0] for row in rows)
                ),
                "median_exhaustive_sparse_pulls": int(
                    statistics.median(row["exhaustive_source_pulls"][1] for row in rows)
                ),
                "median_run_p50_ns": int(
                    statistics.median(row["dynamic_latency"]["p50_ns"] for row in rows)
                ),
                "median_dynamic_p50_ns": int(
                    statistics.median(row["dynamic_latency"]["p50_ns"] for row in rows)
                ),
                "median_prefix_routed_p50_ns": int(
                    statistics.median(row["routed_latency"]["p50_ns"] for row in rows)
                ),
                "median_exhaustive_p50_ns": int(
                    statistics.median(row["exhaustive_latency"]["p50_ns"] for row in rows)
                ),
                "median_run_p95_ns": int(
                    statistics.median(row["dynamic_latency"]["p95_ns"] for row in rows)
                ),
                "total_certification_checks": sum(
                    row["certification_checks"] for row in rows
                ),
                "total_dense_physical_dot_products": sum(
                    row["dense_physical_dot_products"] for row in rows
                ),
                "total_sparse_posting_elements_visited": sum(
                    row["sparse_posting_elements_visited"] for row in rows
                ),
                "total_safe_router_queries": sum(row["safe_router_queries"] for row in rows),
                "total_safe_router_competitor_scheduler_queries": sum(
                    row["safe_router_competitor_scheduler_queries"] for row in rows
                ),
                "total_safe_router_shared_sparse_queries": sum(
                    row["safe_router_shared_sparse_queries"] for row in rows
                ),
                "total_prefix_router_dynamic_queries": sum(
                    row["router_dynamic_queries"] for row in rows
                ),
                "total_prefix_router_exhaustive_queries": sum(
                    row["router_exhaustive_queries"] for row in rows
                ),
                "prefix_router_ordered_top_k_mismatches": sum(
                    row["router_ordered_top_k_mismatches"] for row in rows
                ),
            }
        )
    return summaries


def git_metadata(repository: Path) -> dict[str, Any]:
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
    scenarios = comma_values(args.scenarios, "scenarios")
    schedulers = comma_values(args.schedulers, "schedulers")
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if min(args.documents, args.queries, args.top_k, args.repetitions, args.batch_size) <= 0:
        raise ValueError("positive experiment sizes are required")
    if args.warmup_repetitions < 0:
        raise ValueError("warmup repetitions cannot be negative")

    for repetition in range(args.warmup_repetitions):
        for scenario_index, scenario in enumerate(rotated(scenarios, repetition)):
            for scheduler in rotated(schedulers, repetition + scenario_index):
                run_once(args, scenario, scheduler)

    runs = []
    for repetition in range(args.repetitions):
        for scenario_index, scenario in enumerate(rotated(scenarios, repetition)):
            for position, scheduler in enumerate(
                rotated(schedulers, repetition + scenario_index)
            ):
                print(
                    f"repetition={repetition + 1} scenario={scenario} "
                    f"position={position + 1} scheduler={scheduler}",
                    file=sys.stderr,
                    flush=True,
                )
                runs.append(
                    {
                        "repetition": repetition + 1,
                        "scenario": scenario,
                        "position": position + 1,
                        "scheduler": scheduler,
                        "result": run_once(args, scenario, scheduler),
                    }
                )

    repository = Path(__file__).resolve().parents[2]
    artifact = {
        "schema_version": 1,
        "experiment": "stratumind-v0-executor-matrix-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "correctness_contract": (
            "every exact arm must match full materialization plus deterministic WRRF"
        ),
        "primary_metric": "source pulls and physical work; latency is secondary",
        "repository": git_metadata(repository),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "configuration": {
            "binary": str(args.binary),
            "scenarios": scenarios,
            "schedulers": schedulers,
            "documents": args.documents,
            "queries": args.queries,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "batch_size": args.batch_size,
            "channels": ["dense", "sparse-impact"],
        },
        "summary": summarize(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
