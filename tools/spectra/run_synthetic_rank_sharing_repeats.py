#!/usr/bin/env python3
"""Repeat exact synthetic rank-sharing boundaries with counterbalanced order."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
    parser.add_argument("--term-groups", default="1,8")
    parser.add_argument("--strategies", default="posting-block,auto-shared-lazy")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def conditions(args: argparse.Namespace) -> list[tuple[str, int]]:
    strategies = [value for value in args.strategies.split(",") if value]
    term_groups = [int(value) for value in args.term_groups.split(",") if value]
    return [(strategy, groups) for groups in term_groups for strategy in strategies]


def ordered_conditions(
    values: list[tuple[str, int]], repetition: int
) -> list[tuple[str, int]]:
    offset = repetition % len(values)
    rotated = values[offset:] + values[:offset]
    return list(reversed(rotated)) if repetition % 2 else rotated


def run_once(args: argparse.Namespace, strategy: str, term_groups: int) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "SPECTRA_DOCUMENTS": str(args.documents),
            "SPECTRA_QUERIES": str(args.queries),
            "SPECTRA_CHANNEL_COUNTS": str(args.channels),
            "SPECTRA_SPARSE_TERM_GROUPS": str(term_groups),
            "SPECTRA_SPARSE_STREAM": strategy,
            "SPECTRA_SPARSE_BATCH_SIZE": str(args.batch_size),
            "SPECTRA_CHANNEL_FAMILY": "sparse",
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
    mismatches = sum(row["ordered_top_k_mismatches"] for row in result["matrix"])
    if mismatches:
        raise RuntimeError(f"{strategy}/{term_groups} produced {mismatches} mismatches")
    return result


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["strategy"], run["term_groups"]), []).append(
            run["result"]["matrix"][0]
        )
    rows = []
    for (strategy, term_groups), samples in sorted(groups.items()):
        p50 = [sample["dynamic_latency"]["p50_ns"] for sample in samples]
        p95 = [sample["dynamic_latency"]["p95_ns"] for sample in samples]
        physical_pulls = sum(
            sample["rank_stream_sharing"]["physical_pulls"] for sample in samples
        )
        logical_pulls = sum(
            sample["rank_stream_sharing"]["logical_pulls"] for sample in samples
        )
        rows.append(
            {
                "strategy": strategy,
                "term_groups": term_groups,
                "repetitions": len(samples),
                "ordered_top_k_mismatches": sum(
                    sample["ordered_top_k_mismatches"] for sample in samples
                ),
                "median_run_p50_ns": int(statistics.median(p50)),
                "median_run_p95_ns": int(statistics.median(p95)),
                "mean_run_p50_ns": int(statistics.mean(p50)),
                "mean_run_p95_ns": int(statistics.mean(p95)),
                "physical_to_logical_rank_pull_ratio": physical_pulls / logical_pulls,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    values = conditions(args)
    for repetition in range(args.warmup_repetitions):
        for strategy, term_groups in ordered_conditions(values, repetition):
            run_once(args, strategy, term_groups)

    runs = []
    for repetition in range(args.repetitions):
        for position, (strategy, term_groups) in enumerate(
            ordered_conditions(values, repetition)
        ):
            print(
                f"repetition={repetition + 1} position={position + 1} "
                f"strategy={strategy} term_groups={term_groups}",
                flush=True,
            )
            runs.append(
                {
                    "repetition": repetition + 1,
                    "position": position + 1,
                    "strategy": strategy,
                    "term_groups": term_groups,
                    "result": run_once(args, strategy, term_groups),
                }
            )

    artifact = {
        "schema_version": 1,
        "experiment": "spectra-synthetic-rank-sharing-repeats-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_contract": "training-free, qrels-blind U<D structural rule",
        "machine": {"platform": platform.platform()},
        "configuration": vars(args) | {"binary": str(args.binary), "output": str(args.output)},
        "summary": summarize(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
