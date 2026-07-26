#!/usr/bin/env python3
"""Counterbalanced HTTP promotion gate for the Exact Retrieval clean break.

Baseline and candidate build separate immutable collections because the clean
break intentionally replaces implicit PVS discovery with an explicit profile.
Index construction and startup are excluded from measured latency.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from exact_clean_break_gate_support import (
    RRF_K,
    TOP_K,
    create_collection,
    execute_round,
    execution_summary,
    latency_summary,
    load_json_lines,
    optimize_collection,
    paired_result,
    qdrant_server,
    sha256,
    upload_corpus,
)


DATASETS = {
    "nfcorpus": Path(
        "/Users/ln1/Projects/.stratumind-datasets/snapshots/nfcorpus-bge-small-bm25-v1"
    ),
    "scifact": Path(
        "/Users/ln1/Projects/.stratumind-datasets/snapshots/scifact-bge-small-bm25-v1"
    ),
    "synthetic-100k-384d": Path(
        "/Users/ln1/Projects/.stratumind-datasets/snapshots/synthetic-100k-384d-v1"
    ),
    "trec-covid-100k": Path(
        "/Users/ln1/Projects/.stratumind-datasets/snapshots/trec-covid-100k-bge-small-bm25-v1"
    ),
}
MAX_P50_REGRESSION = 0.03
MAX_P95_REGRESSION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--warmup-sweeps", type=int, default=1)
    parser.add_argument("--measured-rounds", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--upsert-batch", type=int, default=128)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--port", type=int, default=6533)
    parser.add_argument("--seed", type=int, default=0x455253)
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def flatten(runs: list[dict[str, Any]], field: str) -> list[Any]:
    return [value for run in runs for value in run[field]]


def values_not_increased(
    candidate: list[float] | None,
    baseline: list[float] | None,
) -> bool:
    return (
        candidate is not None
        and baseline is not None
        and len(candidate) == len(baseline)
        and all(
            candidate_value <= baseline_value
            for candidate_value, baseline_value in zip(
                candidate,
                baseline,
            )
        )
    )


def main() -> None:
    args = parse_args()
    if args.shards < 1 or args.concurrency < 1:
        raise ValueError("--shards and --concurrency must be positive")

    dataset = (args.dataset_path or DATASETS[args.dataset]).resolve()
    corpus_path = dataset / "corpus-vectors.jsonl"
    queries = load_json_lines(dataset / "query-vectors.jsonl")
    if args.query_limit is not None:
        queries = queries[: args.query_limit]
    candidate = args.candidate_binary.resolve()
    baseline = args.baseline_binary.resolve()
    for binary in (candidate, baseline):
        if not binary.is_file():
            raise FileNotFoundError(binary)

    work_context: tempfile.TemporaryDirectory[str] | None
    if args.keep_workdir:
        work_context = None
        workdir = Path(tempfile.mkdtemp(prefix="exact-clean-break-gate."))
    else:
        work_context = tempfile.TemporaryDirectory(prefix="exact-clean-break-gate.")
        workdir = Path(work_context.name)
    storages = {
        "candidate": workdir / "candidate",
        "baseline": workdir / "baseline",
    }
    binaries = {"candidate": candidate, "baseline": baseline}
    profiles = {"candidate": "dense_sparse_v1", "baseline": None}

    with corpus_path.open() as source:
        dimension = len(json.loads(source.readline())["dense"])
    collection_info: dict[str, dict[str, Any]] = {}
    documents_by_plan: dict[str, int] = {}
    for plan in ("candidate", "baseline"):
        storages[plan].mkdir()
        with qdrant_server(
            binaries[plan],
            storages[plan],
            args.port,
            workdir / f"{plan}-seed.log",
        ) as base_url:
            create_collection(
                base_url,
                dimension,
                args.shards,
                profiles[plan],
            )
            documents, uploaded_dimension = upload_corpus(
                base_url, corpus_path, args.upsert_batch
            )
            if uploaded_dimension != dimension:
                raise AssertionError("corpus dimension changed during upload")
            documents_by_plan[plan] = documents
            collection_info[plan] = optimize_collection(base_url, documents)
    if len(set(documents_by_plan.values())) != 1:
        raise AssertionError(f"indexed corpus sizes differ: {documents_by_plan}")
    documents = documents_by_plan["candidate"]

    runs: dict[str, list[dict[str, Any]]] = {"candidate": [], "baseline": []}
    rng = random.Random(args.seed)
    for round_index in range(args.measured_rounds):
        order = list(range(len(queries)))
        rng.shuffle(order)
        plans = ["candidate", "baseline"]
        if round_index % 2:
            plans.reverse()
        for plan in plans:
            runs[plan].append(
                execute_round(
                    binaries[plan],
                    storages[plan],
                    args.port,
                    workdir / f"{plan}-{round_index}.log",
                    queries,
                    order,
                    args.warmup_sweeps,
                    args.concurrency,
                )
            )

    candidate_orders = flatten(runs["candidate"], "orders")
    baseline_orders = flatten(runs["baseline"], "orders")
    if len(candidate_orders) != len(baseline_orders):
        raise AssertionError("candidate and baseline returned different query counts")
    comparable_orders = [
        (left, right)
        for left, right in zip(candidate_orders, baseline_orders)
        if left is not None and right is not None
    ]
    mismatches = sum(left != right for left, right in comparable_orders)

    latencies = {
        plan: [value for value in flatten(plan_runs, "latencies_ns") if value is not None]
        for plan, plan_runs in runs.items()
    }
    errors = {
        plan: [value for value in flatten(plan_runs, "errors") if value is not None]
        for plan, plan_runs in runs.items()
    }
    executions = {
        plan: flatten(plan_runs, "executions") for plan, plan_runs in runs.items()
    }

    def plan_result(plan: str) -> dict[str, Any]:
        measured_seconds = (
            sum(run["measured_ns"] for run in runs[plan]) / 1_000_000_000
        )
        return {
            "binary": str(binaries[plan]),
            "sha256": sha256(binaries[plan]),
            "latency": latency_summary(latencies[plan]),
            "throughput_qps": len(latencies[plan]) / measured_seconds,
            "exact_successes": len(latencies[plan]),
            "errors": len(errors[plan]),
            "first_error": errors[plan][0] if errors[plan] else None,
            "execution": execution_summary(executions[plan]),
        }

    metadata_path = dataset / "manifest.json"
    if not metadata_path.exists():
        metadata_path = dataset / "snapshot-metadata.json"

    candidate_result = plan_result("candidate")
    baseline_result = plan_result("baseline")
    expected_exact_results = len(queries) * args.measured_rounds
    candidate_latency = candidate_result["latency"]
    baseline_latency = baseline_result["latency"]
    candidate_execution = candidate_result["execution"]
    baseline_execution = baseline_result["execution"]
    checks = {
        "ordered_top_k_mismatches_zero": mismatches == 0,
        "candidate_errors_zero": not errors["candidate"],
        "baseline_errors_zero": not errors["baseline"],
        "all_exact_results_comparable": (
            len(comparable_orders) == expected_exact_results
        ),
        "candidate_exhaustive_fallback_zero": (
            candidate_execution["exhaustive_fallback_rate"] == 0.0
        ),
        "baseline_exhaustive_fallback_zero": (
            baseline_execution["exhaustive_fallback_rate"] == 0.0
        ),
        "source_points_materialized_not_increased": values_not_increased(
            candidate_execution["mean_source_points_materialized"],
            baseline_execution["mean_source_points_materialized"],
        ),
        "source_pulls_not_increased": values_not_increased(
            candidate_execution["mean_source_pulls"],
            baseline_execution["mean_source_pulls"],
        ),
        "p50_regression_within_limit": (
            candidate_latency is not None
            and baseline_latency is not None
            and candidate_latency["p50_ns"]
            <= baseline_latency["p50_ns"] * (1.0 + MAX_P50_REGRESSION)
        ),
        "p95_regression_within_limit": (
            candidate_latency is not None
            and baseline_latency is not None
            and candidate_latency["p95_ns"]
            <= baseline_latency["p95_ns"] * (1.0 + MAX_P95_REGRESSION)
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    result = {
        "schema_version": 2,
        "experiment": "stratumind-exact-clean-break-dataset-gate-v1",
        "dataset": args.dataset,
        "dataset_manifest_sha256": sha256(metadata_path),
        "documents": documents,
        "queries": len(queries),
        "dimension": dimension,
        "requested_shards": args.shards,
        "observed_segments": {
            plan: info["segments_count"] for plan, info in collection_info.items()
        },
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "warmup_sweeps_per_round": args.warmup_sweeps,
        "measured_rounds": args.measured_rounds,
        "concurrency": args.concurrency,
        "candidate": candidate_result,
        "baseline": baseline_result,
        "paired": (
            paired_result(
                latencies["candidate"],
                latencies["baseline"],
                len(queries),
                args.measured_rounds,
                args.seed,
            )
            if not errors["candidate"] and not errors["baseline"]
            else None
        ),
        "comparable_exact_results": len(comparable_orders),
        "ordered_top_k_mismatches": mismatches,
        "gate": {
            "passed": not failed_checks,
            "failed_checks": failed_checks,
            "checks": checks,
            "limits": {
                "max_p50_regression": MAX_P50_REGRESSION,
                "max_p95_regression": MAX_P95_REGRESSION,
            },
            "expected_exact_results": expected_exact_results,
        },
        "workdir": str(workdir) if args.keep_workdir else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed_checks:
        raise SystemExit(2)
    if work_context is not None:
        work_context.cleanup()


if __name__ == "__main__":
    main()
