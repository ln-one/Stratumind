#!/usr/bin/env python3
"""Run exact-composition experiments as isolated, reproducible processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


TOPOLOGIES = (
    "flat",
    "chain_open",
    "balanced_open",
    "balanced_finite",
    "diamond_shared",
    "diamond_unfolded",
)
SCENARIOS = (
    "correlated",
    "clustered",
    "independent",
    "anti_correlated",
    "flat_ties",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("synthetic", "snapshot"), default="synthetic")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--query-repeats", type=int, default=3)
    parser.add_argument("--matrix-repeats", type=int, default=1)
    parser.add_argument("--leaves", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--internal-limit", type=int, default=80)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--topologies", default=",".join(TOPOLOGIES))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--budgets", default="none,32,64,128,256")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--isolated-memory-probes", action="store_true")
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def split_choices(raw: str, allowed: tuple[str, ...], name: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SystemExit(f"unknown {name}: {unknown}")
    return values


def split_budgets(raw: str) -> list[int | None]:
    budgets: list[int | None] = []
    for value in raw.split(","):
        value = value.strip()
        if value == "none":
            budgets.append(None)
            continue
        budget = int(value)
        if budget <= 0:
            raise SystemExit("budgets must be positive or 'none'")
        budgets.append(budget)
    return budgets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(
    binary: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    require_correctness: bool = True,
) -> tuple[dict, int | None]:
    with tempfile.TemporaryFile("w+", encoding="utf-8") as stdout_handle, tempfile.TemporaryFile(
        "w+", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            [str(binary)],
            env=environment,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        deadline = time.monotonic() + timeout_seconds
        while True:
            child_pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if child_pid != 0:
                break
            if time.monotonic() >= deadline:
                process.kill()
                _, status, usage = os.wait4(process.pid, 0)
                stdout_handle.seek(0)
                stderr_handle.seek(0)
                stdout = stdout_handle.read()
                stderr = stderr_handle.read()
                raise TimeoutError(
                    f"experiment exceeded {timeout_seconds:g}s and was killed\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        process.returncode = os.waitstatus_to_exitcode(status)
        stdout_handle.seek(0)
        stderr_handle.seek(0)
        stdout = stdout_handle.read()
        stderr = stderr_handle.read()
    if process.returncode != 0:
        raise RuntimeError(
            f"experiment failed with {process.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    result = json.loads(stdout)
    if require_correctness and result.get("ordered_top_k_mismatches") != 0:
        raise RuntimeError(f"correctness gate failed: {result}")
    max_rss = int(usage.ru_maxrss)
    if platform.system() == "Linux":
        max_rss *= 1024
    return result, max_rss


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    default_name = (
        "spectra_exact_composition"
        if args.kind == "synthetic"
        else "spectra_exact_composition_snapshot"
    )
    binary = args.binary or Path(
        f"/tmp/spectra-qdrant-target/release/examples/{default_name}"
    )
    if not binary.is_file():
        raise SystemExit(f"missing release binary: {binary}")
    if args.kind == "snapshot" and args.dataset_dir is None:
        raise SystemExit("--dataset-dir is required for snapshot runs")

    topologies = split_choices(args.topologies, TOPOLOGIES, "topologies")
    scenarios = (
        split_choices(args.scenarios, SCENARIOS, "scenarios")
        if args.kind == "synthetic"
        else [None]
    )
    budgets = split_budgets(args.budgets)
    configurations = [
        (repeat, scenario, topology, budget)
        for repeat in range(args.matrix_repeats)
        for scenario in scenarios
        for topology in topologies
        for budget in budgets
    ]
    random.Random(args.seed).shuffle(configurations)

    rows = []
    for matrix_repeat, scenario, topology, budget in configurations:
        environment = os.environ.copy()
        environment.update(
            {
                "SPECTRA_TOPOLOGY": topology,
                "SPECTRA_QUERIES": str(args.queries),
                "SPECTRA_QUERY_LIMIT": str(args.queries),
                "SPECTRA_REPEATS": str(args.query_repeats),
                "SPECTRA_TOP_K": str(args.top_k),
                "SPECTRA_RRF_K": str(args.rrf_k),
                "SPECTRA_INTERNAL_LIMIT": str(args.internal_limit),
                "SPECTRA_BLOCK_SIZE": str(args.block_size),
            }
        )
        if budget is None:
            environment.pop("SPECTRA_EXHAUSTIVE_AFTER_PULLS_PER_INPUT", None)
        else:
            environment["SPECTRA_EXHAUSTIVE_AFTER_PULLS_PER_INPUT"] = str(budget)
        if args.kind == "synthetic":
            environment["SPECTRA_DOCUMENTS"] = str(args.documents)
            environment["SPECTRA_LEAVES"] = str(args.leaves)
            environment["SPECTRA_SCENARIO"] = str(scenario)
        else:
            environment["SPECTRA_DATASET_DIR"] = str(args.dataset_dir)

        result, max_rss_bytes = run_one(binary, environment, args.timeout_seconds)
        isolated = {}
        if args.isolated_memory_probes:
            for mode in ("dynamic", "exhaustive"):
                environment["SPECTRA_ISOLATED_EXECUTION"] = mode
                probe, probe_max_rss_bytes = run_one(
                    binary,
                    environment,
                    args.timeout_seconds,
                    require_correctness=False,
                )
                if probe.get("execution_mode") != mode:
                    raise RuntimeError(f"invalid isolated {mode} probe: {probe}")
                isolated[mode] = {
                    "process_max_rss_bytes": probe_max_rss_bytes,
                    "result": probe,
                }
            environment.pop("SPECTRA_ISOLATED_EXECUTION", None)
        rows.append(
            {
                "matrix_repeat": matrix_repeat,
                "process_max_rss_bytes": max_rss_bytes,
                "isolated_execution": isolated or None,
                "result": result,
            }
        )
        print(
            f"[{len(rows)}/{len(configurations)}] "
            f"scenario={scenario} topology={topology} budget={budget} "
            f"leaf_ratio={result['leaf_work_ratio']:.4f} "
            f"mismatch={result['ordered_top_k_mismatches']}",
            flush=True,
        )

    serialized_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    serialized_arguments["binary"] = str(binary)
    artifact = {
        "schema_version": 1,
        "experiment": "spectra-exact-composition-matrix-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "seed": args.seed,
        "configuration_count": len(configurations),
        "arguments": serialized_arguments,
        "rows": rows,
    }
    atomic_json(args.output, artifact)


if __name__ == "__main__":
    main()
