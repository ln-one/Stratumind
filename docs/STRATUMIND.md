# Stratumind Production API V1.1

Stratumind is a Qdrant v1.18.2 fork for certified Dense + Sparse rank fusion. It keeps the stock
Qdrant storage format, REST/gRPC ports, collection lifecycle, ordinary query APIs, snapshots and
Docker layout. The additional API is deliberately isolated from Qdrant's existing `Rrf` behavior.

The authoritative frozen production contract is [Stratumind Production API
V1](STRATUMIND_API_V1.md). This document explains the current implementation and verification
evidence; if descriptive implementation text differs from the V1.1 contract, the contract wins.

## Build and run

```bash
docker build --build-arg PROFILE=perf . --tag stratumind:api-v1.1.5
docker run --rm \
  --publish 6333:6333 \
  --publish 6334:6334 \
  --volume stratumind-storage:/qdrant/storage \
  stratumind:api-v1.1.5
```

The upstream image entrypoint, health endpoints and configuration environment variables remain
unchanged. `perf` keeps optimization enabled but disables release fat LTO; this is the verified local
Docker Desktop build profile. Use the upstream default release profile on a builder with enough
linker memory when release-artifact parity is required.

## Certified hybrid endpoint

```http
POST /collections/{collection_name}/points/query/exact-rrf
Content-Type: application/json
```

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
  "filter": {}
}
```

The request accepts one externally frozen finite Dense vector and either an externally frozen,
non-negative Sparse impact vector or a local `{ "text": "...", "model": "qdrant/bm25" }`
document. Sparse indices must be strictly increasing. The BM25 form uses Qdrant's native local
search encoder and then enters the same exact Sparse rank stream. `limit` is the final WRRF result
count; it is not a per-channel candidate window.

The response returns identities and ranks plus an explicit guarantee and execution record:

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

## V1.1 correctness boundary

For default-consistency reads whose selected Shards all have a local readable replica, the current
HTTP plan pins one Qdrant Segment generation for the complete exact Query. Dense and Sparse each
merge their exact Segment batches through `ExactShardStream`, then merge globally across Shards
before dynamic WRRF. It never fuses shard-local RRF results. Remote replicas and explicit
consistency requests use the exact adaptive-prefix plan. Router choice may change cost but not
ordered Top-K.

Production V1.1.5 uses the clean-break path
`ExactRrfService → ExactHybridSession → ExactShardStream → DenseRankState /
PostingBlockMaxState`. Dense and Sparse retain owned, reader-independent ranking state. When fusion
requests more results, a batch temporarily borrows a Qdrant `SegmentReadView`, advances, releases
the guard, and later resumes from the same state. There is no resident worker per Segment, no
cursor borrowing a Segment across checkpoints, and no legacy cursor/state adapter. The two
channels share the frozen generation and session lifetime, but never scores or certificates.
Qdrant Proxy rollback propagates updates into wrapped Segments before replacement; failure keeps
the Proxy installed and fails closed.

Dense uses an exact, metadata-only Router over Segment-owned PVS V1, Qdrant Scalar bounds and
authoritative exact scan. PVS is built only when the persisted Collection profile is explicitly
`dense_sparse_v1`; ordinary Qdrant Collections remain `disabled`. Compatible immutable Float32
Dot/Cosine Segments persist a per-vector signed-int8 PVS row and prefer it in `Auto`; missing,
stale or corrupt PVS state is quarantined and falls back safely. Every certificate computes an
outward-safe upper bound and full-precision rescores unresolved competitors. The session owns
ordered continuation and ExactRank caching, so one query never pays for an independent Top-N
prefix and then starts a second Dense truth.

Compact remains an internal comparison plan and is built only below its 16,384-point limit. On the
local Apple Silicon TREC-COVID 100K gate, Production `Auto` selected PVS for all 50 Queries, matched
Compact Top-20 exactly, and reduced p50 from `1.783 ms` to `1.730 ms`; Scalar measured `2.828 ms`.
Router thresholds remain cost choices: selecting a slower exact plan changes work only, never the
ordered result.

Sparse `Auto` uses `PostingBlockMax`: a Qdrant compressed-posting ranking session with a
single-pass metadata planner. It retains posting state, publishes only score groups whose order is
certified against all unread posting contribution, and resumes from the same state when fusion asks
for another identity. Equal-score groups are ordered by frozen external point identity. It does not
issue repeated geometric Top-N queries or prepend an independently materialized prefix.

The planner reads existing compressed-posting block maxima; it adds no sidecar and changes no
persisted index bytes. Unsupported, mutable, plain or Proxy representations use the existing exact
Eager Posting Block fallback. Planner failure is never interpreted as exact EOF. The former Eager
plan remains available as an explicit internal comparison and recovery plan.

The Production promotion was an explicit cost decision rather than a semantic change. A release
Top-20 gate over the existing local snapshots produced zero ordered mismatches and the following
paired p50 reductions relative to Eager Posting Block: NFCorpus `10.42%` (323 Queries), SciFact
`29.15%` (1,109 Queries), and TREC-COVID 100K `5.46%` (50 Queries). Corresponding p95 latency also
improved on all three datasets. Posting visits and bound evaluations were unchanged; the gain comes
from cheaper bound planning, not weaker pruning or approximate results.

`sourcePulls` counts identities consumed by the WRRF state machine. `sourcePointsMaterialized`
counts identities delivered to the fusion layer and `exhaustiveFallback` reports an internal
one-shot Qdrant fallback, including an IDF-configured Sparse source. These fields must not be
confused with Dense quantized dots, exact rescoring, Sparse posting visits, or prefix overfetch;
kernel experiments report those counters separately.

`ExactHybridSession` advances bounded Dense or Sparse work through Qdrant's existing runtime and
the shared `ExactShardStream` merge skeleton. Segment count no longer multiplies resident worker
demand, and correctness fallback remains a physical plan rather than a compatibility adapter.

The Shard update barrier remains held for the complete session, while individual Segment guards
exist only during bounded reads. Producer failure and cancellation are errors, never exact EOF.
The remote adaptive plan still checks visible point count before and after execution. A
cross-replica MVCC snapshot token remains outside the current scope.

The Dense/Sparse exact rank states, fallible exact EOF streams, safe Router,
Segment/Shard/global k-way merges, and dynamic WRRF certificate are wired into the Collection
request hot path.

The production promotion gate compared the reader-independent session with the previous resident
worker implementation over NFCorpus, SciFact and TREC-COVID 100K. Every comparable ordered Top-20
matched. Single-query p50 improved by `2.38%` to `21.57%`; with four concurrent requests over eight
Segments, exact success rose from `64.40%` to `100%`, p50 improved `21.78%`, and throughput improved
`61.61%`. Full evidence is recorded in
[ExactRankSession Production Promotion V1.1.3](spectra/results/exact-rank-session-production-promotion-v1.1.3.md).
The Shard-generation and merge promotion is recorded in
[Exact Shard Merge NFCorpus Gate V1](spectra/results/exact-shard-merge-nfcorpus-gate-v1.md).
The V1.1.5 clean-break gate is recorded in
[Exact Retrieval Clean-Break Promotion Gate V1](spectra/results/exact-retrieval-clean-break-promotion-v1.md).

The named Sparse vector used here must store document impacts compatible with the caller's frozen
Sparse Query profile. External non-negative impacts use `modifier: none`. A collection configured
with `modifier: idf` is detected from Segment configuration and routed through Qdrant's one-shot
exact path after official IDF QueryContext initialization; it is never silently double-weighted.

## Reproducible HTTP gate

`tools/spectra/run_exact_rrf_http_smoke.py` builds a two-Shard synthetic collection and compares the
endpoint against an independent full-corpus Dense/Sparse WRRF implementation. Its two phases cover
payload filtering, overwrite updates, deletes, restart persistence, native execution, and the
explicit-consistency exact fallback. Every case rejects the run on the first ordered Top-K
mismatch and verifies the expected exact plan.

```bash
python3 tools/spectra/run_exact_rrf_http_smoke.py --phase seed-and-verify
# Restart the same container and storage volume.
python3 tools/spectra/run_exact_rrf_http_smoke.py --phase verify-existing
```

## Compatibility rule

All stock Qdrant endpoints retain their stock semantics. Stratumind-specific behavior is used only
through `exact-rrf`; clients can switch back to the official v1.18.2 image without migrating stored
collections, but the additional endpoint will no longer exist.
