# Stratumind V0

Stratumind is a Qdrant v1.18.2 fork for certified Dense + Sparse rank fusion. It keeps the stock
Qdrant storage format, REST/gRPC ports, collection lifecycle, ordinary query APIs, snapshots and
Docker layout. The additional API is deliberately isolated from Qdrant's existing `Rrf` behavior.

The authoritative frozen production contract is [Stratumind Production API
V1](STRATUMIND_API_V1.md). This document explains the current implementation and verification
evidence; if descriptive implementation text differs from the V1 contract, the contract wins.

## Build and run

```bash
docker build --build-arg PROFILE=perf . --tag stratumind:qdrant-v1.18.2
docker run --rm \
  --publish 6333:6333 \
  --publish 6334:6334 \
  --volume stratumind-storage:/qdrant/storage \
  stratumind:qdrant-v1.18.2
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

## V0 correctness boundary

For default-consistency reads whose selected Shards all have a local readable replica, the current
HTTP plan pins the same Segment identities for both channels, opens Segment-owned exact rank
producers, merges every channel globally across Segments and Shards, and only then runs dynamic
WRRF. It never fuses shard-local RRF results. Remote replicas and explicit consistency requests use
the older exact adaptive-prefix plan; Router choice may change cost but not ordered Top-K.

Dense uses an exact, metadata-only Router over three Segment-owned plans. Low-dimensional
Segments use one lazy `B+1` Qdrant exact prefix; high-dimensional Segments with at most 16,384
eligible points use the persisted compact signed-int8 residual certificate; larger
high-dimensional Segments use Qdrant Scalar reconstruction bounds. Every plan computes an
outward-safe upper bound and full-precision-rescores unresolved competitors. A strict prefix
boundary, or exact EOF, is required before a prefix is exposed. Tied boundaries and unsupported
storage fall back before emitting anything. Exact scan remains the universal fallback.

The compact certificate is built only below its 16,384-point production limit; large Segments do
not pay its storage cost. Router thresholds are implementation profile values, not correctness
assumptions: selecting a slower exact plan changes cost only, never ordered Top-K.

Sparse uses a two-stage exact producer. It first asks Qdrant `SearchContext` for `B+1` points. The
first `B` are exposed only when the extra point proves a strict score boundary, so internal tie
order cannot leak past the frozen external-identity rule. If fusion asks for more, or the boundary
is tied, a persisted Posting-Block cursor starts lazily and resumes from its exact certificate
state; identities already emitted by the prefix are suppressed. This is one native prefix plus one
resumable fallback, not repeated geometric Top-N queries.

`sourcePulls` counts identities consumed by the WRRF state machine. `sourcePointsMaterialized`
counts identities delivered to the fusion layer and `exhaustiveFallback` reports an internal
one-shot Qdrant fallback, including an IDF-configured Sparse source. These fields must not be
confused with Dense quantized dots, exact rescoring, Sparse posting visits, or prefix overfetch;
kernel experiments report those counters separately.

The local-native plan holds Segment read views for the stream lifetime. Producer failure and
cancellation are errors, never exact EOF. The remote adaptive plan still checks visible point count
before and after execution. A cross-replica MVCC snapshot token remains outside the current scope.

The native Dense/Sparse producers, fallible exact EOF streams, safe Router, Segment/Shard/global
k-way merges, and dynamic WRRF certificate are wired into the Collection request hot path.

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
