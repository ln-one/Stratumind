#!/usr/bin/env python3
"""Create a deterministic Dense + Sparse snapshot for native kernel scale gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MASK64 = (1 << 64) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dense-dimension", type=int, default=64)
    parser.add_argument("--sparse-vocabulary", type=int, default=65_536)
    parser.add_argument("--document-nnz", type=int, default=16)
    parser.add_argument("--query-nnz", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20_260_722)
    return parser.parse_args()


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def unit_float(value: int) -> float:
    return ((mix64(value) >> 40) / float(1 << 24)) * 2.0 - 1.0


def dense_vector(identity: int, dimension: int, seed: int) -> list[float]:
    values = [
        unit_float(seed ^ (identity * 0xD6E8FEB86659FD93) ^ coordinate)
        for coordinate in range(dimension)
    ]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 7) for value in values]


def sparse_vector(
    identity: int, vocabulary: int, nnz: int, seed: int
) -> tuple[list[int], list[float]]:
    impacts: dict[int, float] = {}
    attempt = 0
    while len(impacts) < nnz:
        mixed = mix64(seed ^ (identity * 0xA0761D6478BD642F) ^ attempt)
        term = mixed % vocabulary
        impacts[term] = round(0.1 + ((mixed >> 24) & 0xFFFF) / 32768.0, 7)
        attempt += 1
    indices = sorted(impacts)
    return indices, [impacts[index] for index in indices]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if min(
        args.documents,
        args.queries,
        args.dense_dimension,
        args.sparse_vocabulary,
        args.document_nnz,
        args.query_nnz,
    ) <= 0:
        raise ValueError("snapshot sizes must be positive")
    if args.document_nnz > args.sparse_vocabulary:
        raise ValueError("document nnz exceeds the sparse vocabulary")
    if args.query_nnz > args.document_nnz:
        raise ValueError("query nnz exceeds document nnz")

    args.output.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output / "corpus-vectors.jsonl"
    query_path = args.output / "query-vectors.jsonl"

    with corpus_path.open("w", encoding="utf-8") as stream:
        for identity in range(args.documents):
            indices, values = sparse_vector(
                identity, args.sparse_vocabulary, args.document_nnz, args.seed
            )
            row = {
                "ordinal": identity,
                "dense": dense_vector(identity, args.dense_dimension, args.seed),
                "sparse_indices": indices,
                "sparse_values": values,
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    with query_path.open("w", encoding="utf-8") as stream:
        for query in range(args.queries):
            identity = mix64(args.seed ^ query) % args.documents
            dense = dense_vector(identity, args.dense_dimension, args.seed)
            noise = dense_vector(query, args.dense_dimension, args.seed ^ 0xBAD5EED)
            dense = [left + 0.05 * right for left, right in zip(dense, noise)]
            norm = math.sqrt(sum(value * value for value in dense)) or 1.0
            dense = [round(value / norm, 7) for value in dense]
            indices, values = sparse_vector(
                identity, args.sparse_vocabulary, args.document_nnz, args.seed
            )
            selected = sorted(zip(indices[: args.query_nnz], values[: args.query_nnz]))
            row = {
                "dense": dense,
                "sparse_indices": [index for index, _ in selected],
                "sparse_values": [value for _, value in selected],
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    metadata = {
        "schema_version": 1,
        "generator": Path(__file__).name,
        "documents": args.documents,
        "queries": args.queries,
        "dense_dimension": args.dense_dimension,
        "sparse_vocabulary": args.sparse_vocabulary,
        "document_nnz": args.document_nnz,
        "query_nnz": args.query_nnz,
        "seed": args.seed,
        "checksums": {
            corpus_path.name: sha256(corpus_path),
            query_path.name: sha256(query_path),
        },
    }
    metadata_path = args.output / "snapshot-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
