#!/usr/bin/env python3
"""Exercise the snapshot-backed Spectra exact server without model inference."""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.error
import urllib.request


def read_json(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def first_jsonl(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"{path} contains no rows")


def request_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:6334")
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    manifest = read_json(args.dataset_dir / "manifest.json")
    query = first_jsonl(args.dataset_dir / "query-vectors.jsonl")
    generation = manifest["outputs"]["corpus"]
    base_url = args.base_url.rstrip("/")

    status, health = request_json(f"{base_url}/health")
    assert status == 200 and health["status"] == "ok"
    assert health["snapshot"] == generation

    payload = {
        "indexGenerationId": generation,
        "channels": [
            {
                "channelId": "dense-intent",
                "kind": "dense-f32",
                "vector": query["dense"],
                "weight": 1.0,
            },
            {
                "channelId": "sparse-lexical",
                "kind": "sparse-impact",
                "indices": query["sparse_indices"],
                "values": query["sparse_values"],
                "weight": 1.0,
            },
        ],
        "rerankCandidateK": args.top_k,
        "rrfK": 60,
    }
    status, result = request_json(f"{base_url}/spectra/query", payload)
    assert status == 200
    assert result["indexGenerationId"] == generation
    assert result["guarantee"] == {
        "scope": "full-corpus",
        "orderedTopKExact": True,
        "scoresExact": False,
        "tieBreak": "frozen-point-identity-ascending",
    }
    assert len(result["hits"]) == args.top_k
    assert [hit["rank"] for hit in result["hits"]] == list(range(1, args.top_k + 1))
    assert [channel["channelId"] for channel in result["channels"]] == [
        "dense-intent",
        "sparse-lexical",
    ]
    assert result["physical"]["executor"] == "n-channel-exact"

    stale = dict(payload, indexGenerationId="stale-generation")
    stale_status, stale_result = request_json(f"{base_url}/spectra/query", stale)
    assert stale_status == 400 and "indexGenerationId" in stale_result["error"]

    artifact = {
        "schema_version": 2,
        "experiment": "spectra-n-channel-exact-server-smoke-v2",
        "health": health,
        "query_source_id": query["source_id"],
        "rerank_candidate_k": args.top_k,
        "first_hit": result["hits"][0],
        "guarantee": result["guarantee"],
        "channels": result["channels"],
        "stop": result["stop"],
        "physical": result["physical"],
        "stale_generation_status": stale_status,
    }
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
