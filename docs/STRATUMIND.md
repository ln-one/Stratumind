# Stratumind Production API V1.1

Stratumind is a Qdrant v1.18.2 fork for certified Dense + Sparse rank fusion. It keeps the stock
Qdrant storage format, REST/gRPC ports, collection lifecycle, ordinary query APIs, snapshots and
Docker layout. The additional API is deliberately isolated from Qdrant's existing `Rrf` behavior.

The authoritative frozen production contract is [Stratumind Production API
V1](STRATUMIND_API_V1.md). This document explains the current implementation and verification
evidence; if descriptive implementation text differs from the V1.1 contract, the contract wins.

## Build and run

```bash
docker build --build-arg PROFILE=perf . --tag stratumind:api-v1.1.0
docker run --rm \
  --publish 6333:6333 \
  --publish 6334:6334 \
  --volume stratumind-storage:/qdrant/storage \
  stratumind:api-v1.1.0
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
HTTP plan acquires one frozen Shard snapshot and shares its authoritative point identity/version
map between both channels. Every channel is merged globally across Segments and Shards before
dynamic WRRF; it never fuses shard-local RRF results. Remote replicas and explicit consistency
requests use the older exact adaptive-prefix plan. Router choice may change cost but not ordered
Top-K.

The paired-worker candidate can host independent canonical Dense and Sparse sessions over one
pinned Segment read view. Its channels share only the Qdrant runtime worker and lifetime, never
scores or certificates. It is disabled in the production profile after its latency gate lost to
the bounded materialized exact plan at both 512 and 20,000 points. Internal experiments may enable
it with `STRATUMIND_EXPERIMENTAL_PAIRED_NATIVE_WORKERS=1`; this switch is not part of the Production
API contract.

Dense uses an exact, metadata-only Router over Segment-owned exact scan, persisted compact
signed-int8 residual certificate, and Qdrant Scalar reconstruction-bound plans. Every plan
computes an outward-safe upper bound and full-precision-rescores unresolved competitors. The
session owns the ordered continuation and ExactRank cache, so one query never pays for an
independent Top-N prefix and then starts a second physical Dense truth. Unsupported storage falls
back before emitting anything. Exact scan remains the universal fallback.

The compact certificate is built only below its 16,384-point production limit; large Segments do
not pay its storage cost. Router thresholds are implementation profile values, not correctness
assumptions: selecting a slower exact plan changes cost only, never ordered Top-K.

Sparse uses one Qdrant `SearchContext`-backed native ranking session. It retains posting state,
publishes only score groups whose order is certified against all unread posting contribution, and
resumes from the same state when fusion asks for another identity. Equal-score groups are ordered
by frozen external point identity. It does not issue repeated geometric Top-N queries or prepend an
independently materialized prefix.

`sourcePulls` counts identities consumed by the WRRF state machine. `sourcePointsMaterialized`
counts identities delivered to the fusion layer and `exhaustiveFallback` reports an internal
one-shot Qdrant fallback, including an IDF-configured Sparse source. These fields must not be
confused with Dense quantized dots, exact rescoring, Sparse posting visits, or prefix overfetch;
kernel experiments report those counters separately.

When enabled, the paired plan atomically reserves Qdrant search-runtime capacity for its coordinator
and one worker per Segment before starting. Workers and coordinator may occupy the two existing
Qdrant search runtimes when neither runtime can hold the complete task set alone. The production
default, or any reservation miss, uses one bounded materialized exact plan over the same frozen
snapshot instead of partially starting a session.
Segment read views remain pinned for the session lifetime. Producer failure and cancellation are
errors, never exact EOF. The remote adaptive plan still checks visible point count before and after
execution. A cross-replica MVCC snapshot token remains outside the current scope.

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
