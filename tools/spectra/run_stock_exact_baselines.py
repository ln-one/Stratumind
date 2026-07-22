#!/usr/bin/env python3
"""Repeat clean-upstream Dense and Sparse exact kernel harnesses."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--dense-binary", type=Path, required=True)
    parser.add_argument("--sparse-binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def run(binary: Path, args: argparse.Namespace) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "SPECTRA_DATASET_DIR": str(args.dataset_dir),
            "SPECTRA_QUERY_LIMIT": str(args.query_limit),
            "SPECTRA_TOP_K": str(args.top_k),
        }
    )
    completed = subprocess.run(
        [str(binary)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "median_run_p50_ns": statistics.median(run["p50_ns"] for run in runs),
        "median_run_p95_ns": statistics.median(run["p95_ns"] for run in runs),
        "median_run_mean_ns": statistics.median(run["mean_ns"] for run in runs),
    }


def main() -> None:
    args = parse_args()
    if min(args.repetitions, args.query_limit, args.top_k) <= 0:
        raise ValueError("experiment sizes must be positive")
    if args.warmup_repetitions < 0:
        raise ValueError("warmup repetitions cannot be negative")
    for path in [args.dense_binary, args.sparse_binary]:
        if not path.is_file():
            raise FileNotFoundError(path)

    for repetition in range(args.warmup_repetitions):
        print(f"warmup={repetition + 1}", flush=True)
        run(args.dense_binary, args)
        run(args.sparse_binary, args)

    dense_runs = []
    sparse_runs = []
    for repetition in range(args.repetitions):
        print(f"repetition={repetition + 1}", flush=True)
        if repetition % 2 == 0:
            dense_runs.append(run(args.dense_binary, args))
            sparse_runs.append(run(args.sparse_binary, args))
        else:
            sparse_runs.append(run(args.sparse_binary, args))
            dense_runs.append(run(args.dense_binary, args))

    harnesses = {
        "dense": args.upstream_repo
        / "lib/segment/examples/stratumind_stock_dense_exact.rs",
        "sparse": args.upstream_repo
        / "lib/sparse/examples/stratumind_stock_sparse_exact.rs",
    }
    artifact = {
        "schema_version": 1,
        "experiment": "qdrant-stock-v1.18.2-exact-kernels-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "unmodified upstream library source; measurement harnesses only",
        "counterbalancing": "Dense/Sparse process order alternates by repetition",
        "upstream": {
            "commit": git(args.upstream_repo, "rev-parse", "HEAD"),
            "library_diff": git(args.upstream_repo, "diff", "--", "lib", ":(exclude)lib/segment/examples/stratumind_stock_dense_exact.rs", ":(exclude)lib/sparse/examples/stratumind_stock_sparse_exact.rs"),
            "harness_sha256": {name: sha256(path) for name, path in harnesses.items()},
        },
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "configuration": {
            "dataset_dir": str(args.dataset_dir),
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "query_limit": args.query_limit,
            "top_k": args.top_k,
        },
        "summary": {
            "dense_exact": summarize(dense_runs),
            "sparse_search_context": summarize(sparse_runs),
        },
        "dense_runs": dense_runs,
        "sparse_runs": sparse_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
