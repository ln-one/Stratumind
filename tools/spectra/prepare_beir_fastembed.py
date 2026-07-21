#!/usr/bin/env python3
"""Prepare a pinned BEIR corpus as a Spectra hybrid-vector snapshot.

Requires `fastembed`; model inference stays outside the timed Qdrant kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from fastembed import SparseTextEmbedding, TextEmbedding


FORMAT_VERSION = 1
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


@dataclass(frozen=True)
class BeirDocument:
    source_id: str
    text: str


@dataclass(frozen=True)
class BeirQuery:
    source_id: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--dataset-license", required=True)
    parser.add_argument("--qrels-split", default="test")
    parser.add_argument("--queries-from-qrels", action="store_true")
    parser.add_argument("--skip-empty-documents", action="store_true")
    parser.add_argument(
        "--drop-queries-with-empty-positive-documents",
        action="store_true",
    )
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument(
        "--sparse-pretokenizer",
        choices=("raw", "jieba"),
        default="raw",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-documents", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--sample-seed", default="spectra-real-v1")
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit_documents is not None and args.limit_documents <= 0:
        raise ValueError("--limit-documents must be positive")
    if args.limit_queries is not None and args.limit_queries <= 0:
        raise ValueError("--limit-queries must be positive")

    corpus_path = args.dataset_dir / "corpus.jsonl"
    queries_path = args.dataset_dir / "queries.jsonl"
    qrels_path = args.dataset_dir / "qrels" / f"{args.qrels_split}.tsv"
    documents, skipped_empty_document_ids = read_documents(
        corpus_path,
        args.skip_empty_documents,
    )
    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path) if qrels_path.exists() else []
    queries_with_empty_positive_documents = {
        query_id
        for query_id, document_id, score in qrels
        if score > 0 and document_id in skipped_empty_document_ids
    }
    if (
        queries_with_empty_positive_documents
        and not args.drop_queries_with_empty_positive_documents
    ):
        raise ValueError(
            "current qrels split has queries whose positive documents are empty: "
            f"{sorted(queries_with_empty_positive_documents)}"
        )
    if queries_with_empty_positive_documents:
        queries = [
            query
            for query in queries
            if query.source_id not in queries_with_empty_positive_documents
        ]
        qrels = [
            row for row in qrels if row[0] not in queries_with_empty_positive_documents
        ]

    if args.queries_from_qrels:
        if not qrels:
            raise ValueError("--queries-from-qrels requires a non-empty qrels split")
        qrel_query_ids = {query_id for query_id, _, _ in qrels}
        queries = [query for query in queries if query.source_id in qrel_query_ids]
    if args.limit_queries is not None:
        queries = queries[: args.limit_queries]
    selected_query_ids = {query.source_id for query in queries}
    qrels = [row for row in qrels if row[0] in selected_query_ids]
    documents = select_documents(
        documents,
        qrels,
        args.limit_documents,
        args.sample_seed,
    )
    selected_document_ids = {document.source_id for document in documents}
    qrels = [row for row in qrels if row[1] in selected_document_ids]
    log(
        f"selected {len(documents)} documents, {len(queries)} queries, "
        f"and {len(qrels)} qrels"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    document_texts = [document.text for document in documents]
    query_texts = [query.text for query in queries]
    sparse_document_texts = pretokenize(document_texts, args.sparse_pretokenizer)
    sparse_query_texts = pretokenize(query_texts, args.sparse_pretokenizer)

    stage_started_at = time.perf_counter()
    log(f"loading Sparse model {args.sparse_model}")
    sparse_model = SparseTextEmbedding(
        model_name=args.sparse_model,
        cache_dir=cache_dir,
    )
    average_document_length = (
        sum(sparse_model.token_count(text) for text in sparse_document_texts)
        / len(sparse_document_texts)
    )
    sparse_model = SparseTextEmbedding(
        model_name=args.sparse_model,
        cache_dir=cache_dir,
        avg_len=average_document_length,
    )
    sparse_documents = [
        normalized_sparse(embedding)
        for embedding in sparse_model.embed(
            sparse_document_texts,
            batch_size=args.batch_size,
        )
    ]
    document_frequency: Counter[int] = Counter()
    for indices, _ in sparse_documents:
        document_frequency.update(indices)

    sparse_queries = []
    for embedding in sparse_model.query_embed(sparse_query_texts):
        indices, values = normalized_sparse(embedding)
        values = [
            value * fancy_idf(len(documents), document_frequency[index])
            for index, value in zip(indices, values, strict=True)
        ]
        sparse_queries.append((indices, values))
    raw_vocabulary = sorted(document_frequency)
    dimension_mapping = {
        raw_dimension: remapped_dimension
        for remapped_dimension, raw_dimension in enumerate(raw_vocabulary)
    }
    sparse_documents = [
        remap_sparse(indices, values, dimension_mapping)
        for indices, values in sparse_documents
    ]
    sparse_queries = [
        remap_sparse(indices, values, dimension_mapping)
        for indices, values in sparse_queries
    ]
    sparse_seconds = time.perf_counter() - stage_started_at
    log(f"encoded Sparse vectors in {sparse_seconds:.2f}s")

    stage_started_at = time.perf_counter()
    log(f"loading Dense model {args.dense_model}")
    dense_model = TextEmbedding(
        model_name=args.dense_model,
        cache_dir=cache_dir,
    )
    dense_documents = dense_model.passage_embed(
        document_texts,
        batch_size=args.batch_size,
    )
    dense_queries = dense_model.query_embed(
        query_texts,
        batch_size=args.batch_size,
    )

    corpus_output = args.output_dir / "corpus-vectors.jsonl"
    query_output = args.output_dir / "query-vectors.jsonl"
    qrels_output = args.output_dir / "qrels.tsv"
    write_corpus(
        corpus_output,
        documents,
        dense_documents,
        sparse_documents,
    )
    write_queries(
        query_output,
        queries,
        dense_queries,
        sparse_queries,
    )
    write_qrels(qrels_output, qrels)
    dense_and_write_seconds = time.perf_counter() - stage_started_at
    log(f"encoded Dense vectors and wrote the snapshot in {dense_and_write_seconds:.2f}s")

    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": {
            "name": args.dataset_name,
            "version": args.dataset_version,
            "url": args.dataset_url,
            "license_note": args.dataset_license,
            "qrels_split": args.qrels_split,
            "input_checksums": {
                "corpus": sha256_file(corpus_path),
                "queries": sha256_file(queries_path),
                "qrels": sha256_file(qrels_path) if qrels_path.exists() else None,
            },
        },
        "selection": {
            "documents": len(documents),
            "queries": len(queries),
            "queries_from_qrels": args.queries_from_qrels,
            "skipped_empty_document_ids": sorted(skipped_empty_document_ids),
            "dropped_query_ids_with_empty_positive_documents": sorted(
                queries_with_empty_positive_documents
            ),
            "sample_seed": args.sample_seed,
            "all_selected_query_qrels_preserved": all(
                document_id in selected_document_ids
                for _, document_id, _ in qrels
            ),
        },
        "dense": {
            "model": args.dense_model,
            "dimension": dense_dimension(corpus_output),
        },
        "sparse": {
            "model": args.sparse_model,
            "pretokenizer": args.sparse_pretokenizer,
            "average_document_length": average_document_length,
            "query_modifier": "qdrant_fancy_idf",
            "non_negative": True,
            "dimension_mapping": "sorted-corpus-raw-term-id-to-contiguous-u32",
            "dimensions": len(raw_vocabulary),
            "raw_vocabulary_checksum": vocabulary_checksum(raw_vocabulary),
        },
        "generator": {
            "fastembed_version": importlib.metadata.version("fastembed"),
            "git_commit": git_commit(),
            "timing_seconds": {
                "sparse": round(sparse_seconds, 6),
                "dense_and_write": round(dense_and_write_seconds, 6),
                "total": round(time.perf_counter() - started_at, 6),
            },
        },
        "outputs": {
            "corpus": sha256_file(corpus_output),
            "queries": sha256_file(query_output),
            "qrels": sha256_file(qrels_output),
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    log(f"complete: {args.output_dir}")


def log(message: str) -> None:
    print(f"[spectra-dataset] {message}", file=sys.stderr, flush=True)


def pretokenize(texts: list[str], mode: str) -> list[str]:
    if mode == "raw":
        return texts
    if mode == "jieba":
        try:
            import jieba
        except ImportError as error:
            raise RuntimeError(
                "--sparse-pretokenizer jieba requires the pinned evaluation dependencies"
            ) from error
        return [" ".join(jieba.cut(text, cut_all=False)) for text in texts]
    raise ValueError(f"unsupported Sparse pretokenizer: {mode}")


def read_documents(
    path: Path,
    skip_empty: bool,
) -> tuple[list[BeirDocument], set[str]]:
    documents = []
    skipped_empty_ids = set()
    for row in read_jsonl(path):
        source_id = str(row["_id"])
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        combined = "\n\n".join(part for part in (title, text) if part)
        if not combined:
            if not skip_empty:
                raise ValueError(f"empty BEIR document {source_id}")
            skipped_empty_ids.add(source_id)
            continue
        documents.append(BeirDocument(source_id, combined))
    documents.sort(key=lambda document: document.source_id)
    ensure_unique((document.source_id for document in documents), "document")
    if not documents:
        raise ValueError("BEIR corpus is empty")
    return documents, skipped_empty_ids


def read_queries(path: Path) -> list[BeirQuery]:
    queries = []
    for row in read_jsonl(path):
        source_id = str(row["_id"])
        text = str(row["text"]).strip()
        if not text:
            raise ValueError(f"empty BEIR query {source_id}")
        queries.append(BeirQuery(source_id, text))
    queries.sort(key=lambda query: query.source_id)
    ensure_unique((query.source_id for query in queries), "query")
    if not queries:
        raise ValueError("BEIR query set is empty")
    return queries


def read_qrels(path: Path) -> list[tuple[str, str, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for position, row in enumerate(reader):
            if position == 0 and row and row[0].lower() in {"query-id", "query_id"}:
                continue
            if len(row) != 3:
                raise ValueError(f"invalid qrels row {position + 1}: {row}")
            rows.append((str(row[0]), str(row[1]), int(row[2])))
    return rows


def select_documents(
    documents: list[BeirDocument],
    qrels: list[tuple[str, str, int]],
    limit: int | None,
    seed: str,
) -> list[BeirDocument]:
    if limit is None or limit >= len(documents):
        return documents
    required_ids = {document_id for _, document_id, score in qrels if score > 0}
    if len(required_ids) > limit:
        raise ValueError(
            "--limit-documents is smaller than the relevant-document set for selected queries"
        )
    required = [document for document in documents if document.source_id in required_ids]
    remaining = [document for document in documents if document.source_id not in required_ids]
    remaining.sort(key=lambda document: sample_key(seed, document.source_id))
    selected = required + remaining[: limit - len(required)]
    selected.sort(key=lambda document: document.source_id)
    return selected


def normalized_sparse(embedding: Any) -> tuple[list[int], list[float]]:
    pairs = sorted(
        (int(index), float(value))
        for index, value in zip(embedding.indices, embedding.values, strict=True)
        if float(value) != 0.0
    )
    if any(not math.isfinite(value) or value < 0.0 for _, value in pairs):
        raise ValueError("Sparse model emitted a non-finite or negative impact")
    return [index for index, _ in pairs], [value for _, value in pairs]


def remap_sparse(
    indices: list[int],
    values: list[float],
    dimension_mapping: dict[int, int],
) -> tuple[list[int], list[float]]:
    remapped = [
        (dimension_mapping[index], value)
        for index, value in zip(indices, values, strict=True)
        if index in dimension_mapping
    ]
    return [index for index, _ in remapped], [value for _, value in remapped]


def vocabulary_checksum(raw_vocabulary: list[int]) -> str:
    digest = hashlib.sha256()
    for dimension in raw_vocabulary:
        digest.update(dimension.to_bytes(4, byteorder="big", signed=False))
    return digest.hexdigest()


def fancy_idf(document_count: int, document_frequency: int) -> float:
    return math.log(
        (document_count - document_frequency + 0.5)
        / (document_frequency + 0.5)
        + 1.0
    )


def write_corpus(
    path: Path,
    documents: list[BeirDocument],
    dense_embeddings: Iterable[Any],
    sparse_embeddings: list[tuple[list[int], list[float]]],
) -> None:
    rows = (
        {
            "ordinal": ordinal,
            "source_id": document.source_id,
            "dense": finite_dense(dense),
            "sparse_indices": sparse[0],
            "sparse_values": sparse[1],
        }
        for ordinal, (document, dense, sparse) in enumerate(
            zip(documents, dense_embeddings, sparse_embeddings, strict=True)
        )
    )
    atomic_jsonl(path, rows)


def write_queries(
    path: Path,
    queries: list[BeirQuery],
    dense_embeddings: Iterable[Any],
    sparse_embeddings: list[tuple[list[int], list[float]]],
) -> None:
    rows = (
        {
            "source_id": query.source_id,
            "dense": finite_dense(dense),
            "sparse_indices": sparse[0],
            "sparse_values": sparse[1],
        }
        for query, dense, sparse in zip(
            queries, dense_embeddings, sparse_embeddings, strict=True
        )
    )
    atomic_jsonl(path, rows)


def finite_dense(embedding: Any) -> list[float]:
    vector = [float(value) for value in embedding]
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Dense model emitted an empty or non-finite vector")
    return vector


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{position}") from error


def ensure_unique(values: Iterable[str], label: str) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate BEIR {label} identity {value}")
        seen.add(value)


def sample_key(seed: str, source_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{source_id}".encode()).digest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dense_dimension(corpus_path: Path) -> int:
    first = next(read_jsonl(corpus_path), None)
    if first is None:
        raise ValueError("encoded corpus is empty")
    return len(first["dense"])


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
