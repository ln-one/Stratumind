"""HTTP execution and statistics for the Exact Retrieval clean-break gate."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


COLLECTION = "exact_rank_session_gate"
TOP_K = 20
RRF_K = 60


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: Any | None = None,
    timeout: float = 300,
) -> Any:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} failed with {error.code}: {detail}") from error


def wait_ready(base_url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Qdrant exited early with status {process.returncode}")
        try:
            if request_json(base_url, "GET", "/", timeout=2).get("title"):
                return
        except (OSError, RuntimeError, ValueError):
            pass
        time.sleep(0.1)
    raise TimeoutError("Qdrant did not become ready")


@contextmanager
def qdrant_server(
    binary: Path,
    storage: Path,
    port: int,
    log_path: Path,
) -> Iterator[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "QDRANT__STORAGE__STORAGE_PATH": str(storage),
            "QDRANT__SERVICE__HTTP_PORT": str(port),
            "QDRANT__SERVICE__GRPC_PORT": str(port + 1),
            "QDRANT__TELEMETRY_DISABLED": "true",
            "QDRANT__LOG_LEVEL": "ERROR",
        }
    )
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(binary)],
            cwd=storage.parent,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            wait_ready(base_url, process)
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def point_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["ordinal"],
        "vector": {
            "dense": row["dense"],
            "sparse": {
                "indices": row["sparse_indices"],
                "values": row["sparse_values"],
            },
        },
    }


def create_collection(
    base_url: str,
    dimension: int,
    shards: int,
    exact_rank_profile: str | None,
) -> None:
    body = {
        "vectors": {
            "dense": {
                "size": dimension,
                "distance": "Dot",
                "on_disk": False,
            }
        },
        "sparse_vectors": {"sparse": {"index": {"on_disk": False}}},
        "shard_number": shards,
        "quantization_config": {
            "scalar": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": True,
            }
        },
        "optimizers_config": {
            "default_segment_number": 1,
            "indexing_threshold": 1_000_000_000,
            "max_optimization_threads": 2,
        },
    }
    if exact_rank_profile is not None:
        body["exact_rank_config"] = {"profile": exact_rank_profile}
    request_json(base_url, "PUT", f"/collections/{COLLECTION}", body)


def upload_corpus(
    base_url: str,
    corpus_path: Path,
    batch_size: int,
) -> tuple[int, int]:
    count = 0
    dimension = 0
    batch: list[dict[str, Any]] = []
    with corpus_path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            dimension = len(row["dense"])
            batch.append(point_payload(row))
            if len(batch) == batch_size:
                request_json(
                    base_url,
                    "PUT",
                    f"/collections/{COLLECTION}/points?wait=true",
                    {"points": batch},
                )
                count += len(batch)
                batch.clear()
        if batch:
            request_json(
                base_url,
                "PUT",
                f"/collections/{COLLECTION}/points?wait=true",
                {"points": batch},
            )
            count += len(batch)
    return count, dimension


def optimize_collection(base_url: str, expected_points: int) -> dict[str, Any]:
    request_json(
        base_url,
        "PATCH",
        f"/collections/{COLLECTION}",
        {
            "optimizers_config": {
                "default_segment_number": 1,
                "indexing_threshold": 100,
                "max_optimization_threads": 2,
            }
        },
    )
    deadline = time.monotonic() + 1800
    last: dict[str, Any] = {}
    stable = 0
    while time.monotonic() < deadline:
        last = request_json(base_url, "GET", f"/collections/{COLLECTION}")["result"]
        ready = (
            last.get("status") == "green"
            and last.get("optimizer_status") == "ok"
            and last.get("points_count") == expected_points
            and last.get("indexed_vectors_count", 0) >= expected_points
        )
        stable = stable + 1 if ready else 0
        if stable >= 3:
            return last
        time.sleep(1)
    raise TimeoutError(f"collection optimization did not settle: {last}")


def exact_request(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_rrf": {
            "dense": {"query": query["dense"], "using": "dense"},
            "sparse": {
                "query": {
                    "indices": query["sparse_indices"],
                    "values": query["sparse_values"],
                },
                "using": "sparse",
            },
            "k": RRF_K,
            "weights": [1.0, 1.0],
        },
        "limit": TOP_K,
    }


def query_once(
    base_url: str,
    query: dict[str, Any],
) -> tuple[int, tuple[int, ...], dict[str, Any]]:
    started = time.perf_counter_ns()
    response = request_json(
        base_url,
        "POST",
        f"/collections/{COLLECTION}/points/query/exact-rrf",
        exact_request(query),
    )["result"]
    elapsed = time.perf_counter_ns() - started
    if not response["guarantee"]["orderedTopKExact"]:
        raise AssertionError(f"non-exact success returned: {response}")
    return elapsed, tuple(point["id"] for point in response["points"]), response["execution"]


def percentile(values: list[float] | list[int], fraction: float) -> int:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower])
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def latency_summary(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    return {
        "p50_ns": percentile(values, 0.50),
        "p95_ns": percentile(values, 0.95),
        "p99_ns": percentile(values, 0.99),
        "mean_ns": round(statistics.mean(values)),
    }


def execute_round(
    binary: Path,
    storage: Path,
    port: int,
    log_path: Path,
    queries: list[dict[str, Any]],
    order: list[int],
    warmup_sweeps: int,
    concurrency: int,
) -> dict[str, Any]:
    latencies: list[int | None] = [None] * len(queries)
    orders: list[tuple[int, ...] | None] = [None] * len(queries)
    executions: list[dict[str, Any] | None] = [None] * len(queries)
    errors: list[str | None] = [None] * len(queries)

    def record(query_index: int, result: tuple[int, tuple[int, ...], dict[str, Any]]) -> None:
        latency, point_ids, execution = result
        latencies[query_index] = latency
        orders[query_index] = point_ids
        executions[query_index] = execution

    with qdrant_server(binary, storage, port, log_path) as base_url:
        for _ in range(warmup_sweeps):
            for query_index in order:
                query_once(base_url, queries[query_index])
        measured_started = time.perf_counter_ns()
        if concurrency == 1:
            for query_index in order:
                try:
                    record(query_index, query_once(base_url, queries[query_index]))
                except RuntimeError as error:
                    errors[query_index] = str(error)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(query_once, base_url, queries[query_index]): query_index
                    for query_index in order
                }
                for future in concurrent.futures.as_completed(futures):
                    query_index = futures[future]
                    try:
                        record(query_index, future.result())
                    except RuntimeError as error:
                        errors[query_index] = str(error)
        measured_ns = time.perf_counter_ns() - measured_started
    return {
        "latencies_ns": latencies,
        "orders": orders,
        "executions": executions,
        "errors": errors,
        "measured_ns": measured_ns,
    }


def paired_result(
    candidate: list[int],
    baseline: list[int],
    queries: int,
    rounds: int,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise AssertionError("paired latency vectors have different lengths")
    deltas = [left - right for left, right in zip(candidate, baseline)]
    query_deltas = [
        statistics.mean(deltas[round_index * queries + query_index] for round_index in range(rounds))
        for query_index in range(queries)
    ]
    rng = random.Random(seed)
    bootstrap_means = [
        statistics.mean(rng.choice(query_deltas) for _ in query_deltas)
        for _ in range(bootstrap_samples)
    ]
    return {
        "mean_delta_ns": round(statistics.mean(deltas)),
        "median_delta_ns": round(statistics.median(deltas)),
        "win_rate": sum(delta < 0 for delta in deltas) / len(deltas),
        "query_cluster_bootstrap_mean_delta_95_ci_ns": [
            percentile(bootstrap_means, 0.025),
            percentile(bootstrap_means, 0.975),
        ],
    }


def execution_summary(executions: list[dict[str, Any] | None]) -> dict[str, Any]:
    valid = [execution for execution in executions if execution is not None]
    if not valid:
        return {
            "exhaustive_fallback_rate": None,
            "mean_source_points_materialized": None,
            "mean_source_pulls": None,
            "mean_query_rounds": None,
        }
    return {
        "exhaustive_fallback_rate": sum(
            bool(execution["exhaustiveFallback"]) for execution in valid
        )
        / len(valid),
        "mean_source_points_materialized": [
            statistics.mean(
                execution["sourcePointsMaterialized"][channel] for execution in valid
            )
            for channel in range(2)
        ],
        "mean_source_pulls": [
            statistics.mean(execution["sourcePulls"][channel] for execution in valid)
            for channel in range(2)
        ],
        "mean_query_rounds": statistics.mean(
            execution["queryRounds"] for execution in valid
        ),
    }
