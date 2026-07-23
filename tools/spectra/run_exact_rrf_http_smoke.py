#!/usr/bin/env python3
"""Reproducible Stratumind exact-RRF HTTP correctness and persistence smoke."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DOCUMENTS = 512
DIMENSION = 16
RRF_K = 60
TOP_K = 20
WEIGHTS = (1.0, 1.0)
QUERY_DENSE = tuple((((index * 19 + 7) % 37) - 18) / 18 for index in range(DIMENSION))
QUERY_SPARSE = {3: 1.0, 11: 0.75, 29: 1.25, 47: 0.5}


def request_json(base_url: str, method: str, path: str, body: Any | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} failed with {error.code}: {detail}") from error


def dense_vector(point_id: int) -> list[float]:
    return [(((point_id * (index + 3) + index * 17) % 101) - 50) / 50 for index in range(DIMENSION)]


def sparse_vector(point_id: int) -> dict[int, float]:
    impacts: dict[int, float] = {}
    for salt, multiplier in ((1, 5), (3, 11), (7, 17), (13, 23)):
        index = (point_id * multiplier + salt) % 53
        impacts[index] = impacts.get(index, 0.0) + 0.25 + ((point_id + salt) % 9) / 4
    return impacts


def document(point_id: int) -> dict[str, Any]:
    return {
        "dense": dense_vector(point_id),
        "sparse": sparse_vector(point_id),
        "payload": {"visible": point_id % 3 != 1, "bucket": point_id % 7},
    }


def final_documents() -> dict[int, dict[str, Any]]:
    documents = {point_id: document(point_id) for point_id in range(DOCUMENTS)}
    for point_id in (0, 2, 3):
        documents[point_id] = {
            "dense": [coordinate * (100 + point_id) for coordinate in QUERY_DENSE],
            "sparse": {index: value * (100 + point_id) for index, value in QUERY_SPARSE.items()},
            "payload": {"visible": True, "bucket": point_id % 7, "updated": True},
        }
    for point_id in (6, 12, 18):
        documents.pop(point_id)
    return documents


def point_payload(point_id: int, value: dict[str, Any]) -> dict[str, Any]:
    sparse = value["sparse"]
    return {
        "id": point_id,
        "vector": {
            "dense": value["dense"],
            "sparse": {
                "indices": sorted(sparse),
                "values": [sparse[index] for index in sorted(sparse)],
            },
        },
        "payload": value["payload"],
    }


def upsert(base_url: str, collection: str, documents: dict[int, dict[str, Any]]) -> None:
    points = [point_payload(point_id, value) for point_id, value in sorted(documents.items())]
    request_json(
        base_url,
        "PUT",
        f"/collections/{collection}/points?wait=true",
        {"points": points},
    )


def matches_filter(value: dict[str, Any], visible_only: bool) -> bool:
    return not visible_only or value["payload"].get("visible") is True


def exact_channel_orders(
    documents: dict[int, dict[str, Any]], visible_only: bool
) -> tuple[list[int], list[int]]:
    dense_scores = []
    sparse_scores = []
    for point_id, value in documents.items():
        if not matches_filter(value, visible_only):
            continue
        dense_score = math.fsum(
            left * right for left, right in zip(QUERY_DENSE, value["dense"])
        )
        dense_scores.append((point_id, dense_score))
        sparse_score = math.fsum(
            QUERY_SPARSE.get(index, 0.0) * impact
            for index, impact in value["sparse"].items()
        )
        if sparse_score != 0.0:
            sparse_scores.append((point_id, sparse_score))
    dense_scores.sort(key=lambda item: (-item[1], item[0]))
    sparse_scores.sort(key=lambda item: (-item[1], item[0]))
    return [point_id for point_id, _ in dense_scores], [point_id for point_id, _ in sparse_scores]


def exhaustive_wrrf(
    documents: dict[int, dict[str, Any]], visible_only: bool
) -> list[int]:
    orders = exact_channel_orders(documents, visible_only)
    scores: dict[int, float] = {}
    for order, weight in zip(orders, WEIGHTS):
        for position, point_id in enumerate(order):
            contribution = 1.0 / ((position + 1) / weight + RRF_K - 1)
            scores[point_id] = scores.get(point_id, 0.0) + contribution
    return [
        point_id
        for point_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:TOP_K]
    ]


def exact_rrf_request(visible_only: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "exact_rrf": {
            "dense": {"query": QUERY_DENSE, "using": "dense"},
            "sparse": {
                "query": {
                    "indices": sorted(QUERY_SPARSE),
                    "values": [QUERY_SPARSE[index] for index in sorted(QUERY_SPARSE)],
                },
                "using": "sparse",
            },
            "k": RRF_K,
            "weights": WEIGHTS,
        },
        "limit": TOP_K,
    }
    if visible_only:
        body["filter"] = {"must": [{"key": "visible", "match": {"value": True}}]}
    return body


def verify_case(
    base_url: str,
    collection: str,
    documents: dict[int, dict[str, Any]],
    name: str,
    visible_only: bool,
    force_exact_fallback: bool = False,
) -> dict[str, Any]:
    query = "?consistency=1" if force_exact_fallback else ""
    response = request_json(
        base_url,
        "POST",
        f"/collections/{collection}/points/query/exact-rrf{query}",
        exact_rrf_request(visible_only),
    )["result"]
    actual = [point["id"] for point in response["points"]]
    expected = exhaustive_wrrf(documents, visible_only)
    if actual != expected:
        raise AssertionError(f"{name}: ordered Top-K mismatch\nactual={actual}\nexpected={expected}")
    guarantee = response["guarantee"]
    execution = response["execution"]
    expected_plans = (
        {
            "adaptive-exact-prefix-v0",
            "adaptive-exact-prefix-with-exhaustive-fallback-v0",
        }
        if force_exact_fallback
        else {"native-local-dense-sparse-v1"}
    )
    if not guarantee["orderedTopKExact"] or execution["plan"] not in expected_plans:
        raise AssertionError(
            f"{name}: expected one of the exact plans {expected_plans!r}: {response}"
        )
    return {
        "name": name,
        "ordered_top_k_mismatches": 0,
        "point_ids": actual,
        "guarantee": guarantee,
        "execution": execution,
    }


def wait_ready(base_url: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            response = request_json(base_url, "GET", "/")
            if isinstance(response, dict) and response.get("title") == "qdrant - vector search engine":
                return
        except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, AttributeError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Qdrant at {base_url} did not become ready")


def seed_and_verify(base_url: str, collection: str) -> list[dict[str, Any]]:
    try:
        request_json(base_url, "DELETE", f"/collections/{collection}")
    except RuntimeError as error:
        if "404" not in str(error):
            raise
    request_json(
        base_url,
        "PUT",
        f"/collections/{collection}",
        {
            "vectors": {"dense": {"size": DIMENSION, "distance": "Dot"}},
            "sparse_vectors": {"sparse": {"index": {"on_disk": False}}},
            "shard_number": 2,
        },
    )
    documents = {point_id: document(point_id) for point_id in range(DOCUMENTS)}
    upsert(base_url, collection, documents)
    cases = [
        verify_case(base_url, collection, documents, "base", False),
        verify_case(base_url, collection, documents, "payload-filter", True),
        verify_case(
            base_url,
            collection,
            documents,
            "base-explicit-consistency-fallback",
            False,
            force_exact_fallback=True,
        ),
    ]

    updated = final_documents()
    upsert(base_url, collection, {point_id: updated[point_id] for point_id in (0, 2, 3)})
    request_json(
        base_url,
        "POST",
        f"/collections/{collection}/points/delete?wait=true",
        {"points": [6, 12, 18]},
    )
    cases.extend(
        [
            verify_case(base_url, collection, updated, "overwrite-delete", False),
            verify_case(base_url, collection, updated, "overwrite-delete-filter", True),
        ]
    )
    return cases


def benchmark_existing(
    base_url: str, collection: str, warmup: int, samples: int
) -> dict[str, Any]:
    path = f"/collections/{collection}/points/query/exact-rrf"
    body = exact_rrf_request(False)
    for _ in range(warmup):
        request_json(base_url, "POST", path, body)
    latencies = []
    result = None
    for _ in range(samples):
        started = time.perf_counter_ns()
        result = request_json(base_url, "POST", path, body)["result"]
        latencies.append(time.perf_counter_ns() - started)
    assert result is not None
    actual = [point["id"] for point in result["points"]]
    if actual != exhaustive_wrrf(final_documents(), False):
        raise AssertionError("benchmark-existing: ordered Top-K mismatch")
    latencies.sort()

    def percentile(percent: float) -> int:
        index = round((len(latencies) - 1) * percent)
        return latencies[index]

    return {
        "samples": samples,
        "warmup": warmup,
        "p50_ns": statistics.median(latencies),
        "p95_ns": percentile(0.95),
        "p99_ns": percentile(0.99),
        "min_ns": latencies[0],
        "max_ns": latencies[-1],
        "ordered_top_k_mismatches": 0,
        "execution": result["execution"],
    }


def main() -> None:
    global DOCUMENTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:6533")
    parser.add_argument("--collection", default="stratumind_exact_rrf_smoke")
    parser.add_argument(
        "--phase",
        choices=("seed-and-verify", "verify-existing", "benchmark-existing"),
        default="seed-and-verify",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--documents", type=int, default=DOCUMENTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.documents <= 0:
        parser.error("--documents must be positive")
    DOCUMENTS = args.documents

    wait_ready(args.base_url)
    if args.phase == "seed-and-verify":
        cases = seed_and_verify(args.base_url, args.collection)
    elif args.phase == "verify-existing":
        cases = [
            verify_case(
                args.base_url,
                args.collection,
                final_documents(),
                "restart-persistence",
                False,
            ),
            verify_case(
                args.base_url,
                args.collection,
                final_documents(),
                "restart-persistence-filter",
                True,
            ),
            verify_case(
                args.base_url,
                args.collection,
                final_documents(),
                "restart-persistence-explicit-consistency-fallback",
                False,
                force_exact_fallback=True,
            ),
        ]
    else:
        if args.warmup < 0 or args.samples <= 0:
            parser.error("--warmup must be non-negative and --samples must be positive")
        cases = []
    artifact = {
        "schema_version": 1,
        "experiment": "stratumind-exact-rrf-http-smoke-v1",
        "phase": args.phase,
        "documents_before_delete": DOCUMENTS,
        "dense_dimension": DIMENSION,
        "shards": 2,
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "cases": cases,
    }
    if args.phase == "benchmark-existing":
        artifact["benchmark"] = benchmark_existing(
            args.base_url, args.collection, args.warmup, args.samples
        )
    encoded = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
