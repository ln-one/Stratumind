# Stratumind V0

Stratumind is a Qdrant v1.18.2 fork for certified Dense + Sparse rank fusion. It keeps the stock
Qdrant storage format, REST/gRPC ports, collection lifecycle, ordinary query APIs, snapshots and
Docker layout. The additional API is deliberately isolated from Qdrant's existing `Rrf` behavior.

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

The request accepts one externally frozen finite Dense vector and one externally frozen,
non-negative Sparse impact vector. Sparse indices must be strictly increasing. `limit` is the final
WRRF result count; it is not a per-channel candidate window.

The response returns identities and ranks plus an explicit guarantee and execution record:

```json
{
  "result": {
    "points": [{ "id": 42, "rank": 1, "version": 7 }],
    "guarantee": {
      "scope": "selected-shards-request-view",
      "orderedTopKExact": true,
      "tieBreak": "point-identity-ascending",
      "channelInput": "tie-complete-exact-prefixes"
    },
    "execution": {
      "plan": "adaptive-exact-prefix-v0",
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

The current HTTP physical plan asks Qdrant for exact Dense and Sparse prefixes over the same
selected-shard request view, restores each channel's global order, and runs the dynamic WRRF
certificate over tie-complete prefixes. A one-point probe detects when a score tie crosses a prefix
boundary; that incomplete tie group is withheld and the prefix is deepened geometrically. If the
result is still not certified after three adaptive rounds, execution reads exact EOF. This is an
unbounded exact strategy, not a fixed Dense or Sparse candidate window, and it never fuses
shard-local RRF results.

`sourcePulls` counts identities consumed by the WRRF state machine. `sourcePointsMaterialized`
counts all result points returned by the repeated native queries and can be larger than the corpus
when prefixes are rerun. `exhaustiveFallback` makes the worst-case path explicit. These fields must
not be confused with Dense dot products or Sparse posting decodes: the V0 HTTP plan can reduce
fusion consumption and returned prefix size, but Dense `exact=true` may still score every visible
point internally.

The handler checks the exact visible point count before and after execution. A count change or any
producer failure is an error, not an exact success. The scope is intentionally named
`selected-shards-request-view`: this prototype does not yet expose a cross-shard MVCC snapshot token,
so a same-count concurrent replacement is outside the present guarantee. Production callers should
pin an immutable index generation in the filter.

The native resumable Sparse cursor, fallible exact EOF stream, cost-aware scheduler, safe exact
Router, and Segment/Shard k-way merge primitives are implemented below the API. They are not yet
wired into the Collection request hot path. Replacing adaptive reruns with lazy Segment/Shard
cursors is the next physical optimization; it must not change this endpoint's result contract.

The named Sparse vector used here must store document impacts compatible with the caller's frozen
Sparse Query profile. In particular, a Query vector that already contains IDF-weighted impacts must
not be sent to a vector configured to apply Qdrant's `modifier: idf` again. Spectra therefore treats
the vector name and Sparse profile hash as explicit adapter configuration rather than silently
reusing its text-generated BM25 vector.

## Compatibility rule

All stock Qdrant endpoints retain their stock semantics. Stratumind-specific behavior is used only
through `exact-rrf`; clients can switch back to the official v1.18.2 image without migrating stored
collections, but the additional endpoint will no longer exist.
