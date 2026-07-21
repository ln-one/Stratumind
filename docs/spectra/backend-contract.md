# Spectra single-request Chunk retrieval contract

Spectra owns one logical `ChunkRetrievalBackend` port. A request supplies the
frozen index generation, one to N channel Query representations, one WRRF
profile, and the rerank-candidate limit. A response contains one globally
ordered Chunk identity list. Channel-specific intermediate lists never cross
this boundary.

The port does not know that a channel is implemented by HNSW, Posting Block,
native Qdrant Sparse search, or a future executor. In `reference-exact` mode the
only required channel property is a recoverable, cancelable, identity-total exact
ordered stream. In `production-hybrid` mode an adapter may expose an approximate
or fixed-prefix channel, but it must declare the weaker guarantee.

## Execution modes

`reference-exact` uses admissible streams for every channel. Its ordered Top-K
is identical to exhaustive full-corpus WRRF over the same frozen single-shard
snapshot. It is the oracle, audit, and research mode. Full-corpus fallback is
valid. The current V3 Sparse profile uses strict Qdrant native Top-K prefixes
starting at 4096 and doubles only when the fusion proof requests more ranks.
Max-next and block pruning are disabled in this strict producer until their f32
certificates are outward-safe. Posting Block and Shared Lazy remain selectable
exact implementations behind the same port.

`production-hybrid` sends one Qdrant Query request containing Dense and Sparse
prefetches plus weighted RRF. It uses Qdrant's mature Sparse inverted and Dense
HNSW executors. Fusion is exact over the declared candidate streams, but an
approximate or truncated channel stream does not imply full-corpus exactness.

The mode is explicit. An adapter must reject the wrong mode instead of silently
downgrading its guarantee.

## Logical Spectra request

```ts
type ChunkRetrievalRequest = {
  mode: "production-hybrid" | "reference-exact";
  indexGenerationId: string;
  channels: readonly Array<
    | {
        channelId: string;
        kind: "dense-f32";
        vector: readonly number[];
        weight: number;
        candidateK?: number;
      }
    | {
        channelId: string;
        kind: "sparse-impact";
        indices: readonly number[];
        values: readonly number[];
        weight: number;
        candidateK?: number;
      }
    | {
        channelId: string;
        kind: "bm25-text";
        text: string;
        weight: number;
        candidateK: number;
      }
  >;
  rerankCandidateK: number;
  rrfK: number;
  signal?: AbortSignal;
};
```

`channelId` is unique within one request and exists for audit/telemetry only; it
cannot influence rank ties. `weight` is finite and non-negative. At least one
weight must be positive. `rerankCandidateK` is the sole output-size variable of
this retrieval black box; the V3 profile defaults it to 20 but does not hardcode
20 in the kernel.

`production-hybrid` accepts Qdrant BM25 text or already encoded vectors.
`reference-exact` rejects `bm25-text`: every Dense vector must be finite and have
the frozen dimension, while every Sparse impact vector must have sorted unique
indices and finite non-negative values. An empty Sparse vector is valid and
denotes an empty exact stream. Encoder inference and Query rewriting remain
outside the timed kernel.

## Logical Spectra response

```ts
type ChunkRetrievalResponse = {
  indexGenerationId: string;
  hits: Array<{
    pointId: string;
    retrievalChunkId: string;
    sourceId: string;
    sourceRevisionId: string;
    representationId: string;
    contentHash: string;
    rank: number;
    fusionScore: number | null;
  }>;
  guarantee: {
    scope: "full-corpus" | "candidate-streams";
    orderedTopKExact: boolean;
    scoresExact: boolean;
    tieBreak: "frozen-point-identity-ascending";
  };
  channels: Array<{
    channelId: string;
    logicalRankPulls: number;
    exhausted: boolean;
  }>;
  physical: {
    executor: string;
    physicalStreams: number;
    physicalRankPulls: number;
    peakReplayBufferedIdentities: number;
  };
};
```

`fusionScore` is nullable by design. Rank and identity are authoritative. The
exact executor may stop when the ordered identities are fixed even though an
unread channel could still add a bounded contribution to their numeric scores.

## Reference server wire protocol

The target Qdrant-fork wire protocol exposes the snapshot-backed single-shard
oracle at `POST /spectra/query`:

```json
{
  "indexGenerationId": "sha256-of-corpus-vector-snapshot",
  "channels": [
    {
      "channelId": "dense-intent",
      "kind": "dense-f32",
      "vector": [0.01, -0.02],
      "weight": 1.0
    },
    {
      "channelId": "sparse-lexical",
      "kind": "sparse-impact",
      "indices": [12, 91],
      "values": [1.4, 0.7],
      "weight": 1.0
    }
  ],
  "rerankCandidateK": 20,
  "rrfK": 60
}
```

It returns stable snapshot point IDs, ranks, the exactness declaration, stopping
reason, source pulls, certification checks, and physical access telemetry.
Spectra then resolves those point IDs through its authoritative retrieval
catalog. This identity recovery does not alter retrieval order and prevents the
research snapshot from fabricating production Chunk metadata.

The runnable reference server accepts `channels[]`, validates every channel
before execution, and returns per-channel plus physical-stream telemetry. Dense
and Sparse channels are freely interleaved; the request shape does not encode a
fixed two-channel assumption. It uses the same Adaptive Native(4096) Sparse
default as `NChannelExactIndex`.

Run it with:

```bash
SPECTRA_DATASET_DIR=/tmp/spectra-scifact-snapshot \
SPECTRA_BIND=127.0.0.1:6334 \
cargo run --release --bin spectra_exact_server --locked
```

`GET /health` reports the loaded snapshot, corpus size, Dense dimension, and
offline index build time.

With the server running, the reproducible HTTP smoke gate is:

```bash
python tools/spectra/smoke_exact_server.py \
  --dataset-dir /tmp/spectra-scifact-snapshot
```

## Failure contract

The backend never returns a successful exact claim after timeout, cancellation,
snapshot change, duplicate `channelId`, invalid weight, invalid Sparse input,
Dense dimension mismatch, or identity-universe mismatch. A canceled stream is
not silently replaced by a shorter candidate list. Spectra also rejects an exact
response whose guarantee is absent or weaker than
`full-corpus orderedTopKExact`.

Transport abandonment is wired to the same kernel cancellation flag. The HTTP
handler owns an armed drop guard while its blocking retrieval worker is alive;
dropping the handler sets the flag, every rank stream checks it before its next
pull, and the adaptive Sparse scorer receives the same flag. The kernel checks
again before returning, so cancellation can never escape as a successful shorter
Top-K. HTTP/1 half-close support is deliberately disabled so an EOF aborts the
connection handler instead of waiting for its response. Cancellation is
cooperative: latency is bounded by the currently running Dense preparation or
native Sparse batch, not by the whole query. Tests cover guard propagation,
stream stopping, pre-cancelled search, and a real TCP disconnect after the
blocking worker has started; the latter verifies that the worker observes the
flag and exits.

## Shard boundary

The current exact kernel is single-shard. Collection-level exactness requires a
coordinator to merge shard-level next-value bounds before certifying a global
rank. Taking a fixed Top-L from each shard and applying WRRF afterward is not
globally lazy exact retrieval.
