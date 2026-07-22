# Stratumind Production API V1

This document is the authoritative frozen contract for the Stratumind production retrieval API.
It describes the external behavior of `stratumind-api-v1.1.0`, based on Qdrant v1.18.2. V1.1 is a
backward-compatible input extension of V1.0. Research
servers, N-channel experiments, and physical execution plans are not part of this contract.

## Endpoint

```http
POST /collections/{collection_name}/points/query/exact-rrf
Content-Type: application/json
```

V1 accepts exactly one externally encoded Dense channel and one Sparse channel. The Sparse query
is either an externally encoded non-negative impact vector or a local Qdrant BM25 document. Query
rewriting and Dense inference remain outside Stratumind.

```json
{
  "exact_rrf": {
    "dense": { "query": [0.1, 0.2], "using": "dense" },
    "sparse": {
      "query": { "indices": [12, 99], "values": [1.3, 0.5] },
      "using": "sparse"
    },
    "k": 60,
    "weights": [1.0, 1.0]
  },
  "limit": 20,
  "filter": {},
  "shard_key": null
}
```

V1.1 additionally accepts this Sparse query shape without changing any response field or ranking
rule:

```json
{
  "exact_rrf": {
    "dense": { "query": [0.1, 0.2], "using": "dense" },
    "sparse": {
      "query": {
        "text": "exact hybrid retrieval",
        "model": "qdrant/bm25",
        "options": {}
      },
      "using": "sparse"
    },
    "k": 60,
    "weights": [1.0, 1.0]
  },
  "limit": 20
}
```

`k` is the WRRF rank constant. `limit` is the final rerank-candidate count, never a per-channel
candidate window. `weights` defaults to `[1.0, 1.0]`; every weight must be finite and
non-negative, and at least one must be positive.

The Dense vector must be non-empty and finite. Its dimension and named-vector compatibility are
validated by Qdrant. Sparse indices and values must have equal length, indices must be strictly
increasing, and values must be finite and non-negative. An empty Sparse vector is valid and denotes
an empty exact channel. A point whose exact Sparse score is zero is absent from the Sparse rank
stream, matching Qdrant Sparse query semantics. Unknown request fields are rejected.

The document form permits only the exact model name `qdrant/bm25`. Stratumind parses its options
with Qdrant's native BM25 configuration, runs the local deterministic search encoder, and passes the
resulting Sparse vector into the same exact rank stream as the explicit-vector form. Remote
inference and other models are rejected. An empty document is valid and has Qdrant's native BM25
empty-query behavior. Unknown document fields, unknown BM25 options, and invalid option values are
errors.

Standard Qdrant read query parameters, including timeout and consistency, retain their existing
meaning. `filter` and `shard_key` restrict the visible retrieval universe before ranking.

## Response

```json
{
  "result": {
    "points": [{ "id": 42, "rank": 1, "version": 7 }],
    "guarantee": {
      "scope": "selected-local-shards-frozen-segment-view",
      "orderedTopKExact": true,
      "tieBreak": "point-identity-ascending",
      "channelInput": "native-exact-rank-streams"
    },
    "execution": {
      "plan": "native-local-dense-sparse-v1",
      "stopReason": "top-k-fixed",
      "sourcePulls": [31, 28],
      "sourceExhausted": [false, false],
      "certificationChecks": 9,
      "corpusPointsObserved": 1000,
      "queryRounds": 1,
      "sourcePointsMaterialized": [81, 81],
      "exhaustiveFallback": false
    }
  },
  "status": "ok"
}
```

`points` is the authoritative globally ordered identity list. `rank` is one-based and `version` is
the authoritative Qdrant point version observed by the request. `id` retains Qdrant's point-ID
representation and may therefore be an unsigned integer or UUID string. V1 deliberately does not
expose a fusion score: the ordered identities may be fixed before every bounded unread contribution
has been materialized.

`guarantee` is part of the correctness contract. A successful exact response means that `points`
is identical to exhaustively producing each Dense and Sparse channel's complete exact rank stream
over the visible universe, then applying weighted reciprocal-rank fusion and selecting the first
`limit` identities. Equal channel scores and equal fusion states are resolved by ascending frozen
point identity. If fewer than `limit` positive-weight fused identities exist, every available fused
identity is returned.

`execution` is diagnostic telemetry. Its envelope and current fields are retained in V1, but plan
names, counters, stopping paths, batch sizes, Router decisions, and fallback frequency are not
correctness signals. Production clients must make correctness decisions from `points` and
`guarantee`, not from a particular physical plan or counter value.

## Failure and compatibility rules

Timeout, cancellation, producer failure, invalid input, and a request view that cannot support the
reported exact guarantee must return an error rather than a successful shorter prefix. Exact EOF
is the only successful end-of-stream condition.

The native local plan and the adaptive exact fallback implement the same ordered Top-K contract.
Compact, Scalar, exact-prefix, Posting Block, scheduling, sharing, and Router thresholds may change
without changing this API. Stock Qdrant endpoints and storage formats retain their upstream
semantics.

This endpoint and its Dense-plus-Sparse meaning remain backward compatible for the lifetime of V1.
Every valid V1.0 explicit-Sparse request has identical V1.1 behavior. BM25 document inference is
only an input normalization step and does not weaken the exhaustive ordered Top-K guarantee.
An incompatible request shape, including a general N-channel production API, must use a new API
version or endpoint. Research results do not silently alter V1 behavior.
