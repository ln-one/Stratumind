# PostingBlockMax Production V1.1.2 Gate

## Decision

Production Sparse `Auto` uses the compressed-metadata `PostingBlockMax` planner. The previous
Eager Posting Block implementation remains an explicit exact fallback.

This promotion changes only the physical bound-planning path:

- strict ordered Sparse stream semantics are unchanged;
- pause/resume state and fail-closed behavior are unchanged;
- WRRF and Production API V1.1 are unchanged;
- no sidecar or persisted-index format is added;
- unsupported storage representations fall back to the Eager exact planner.

The Shard batching experiment, Touched/Sorted accumulators, BMP sidecars and HNSW research paths are
not part of this promotion.

## Stable-worktree release gate

All runs used Top-20, scan span 4,096, result batch 32, one warmup and three measured
counterbalanced rounds.

| Dataset | Documents | Queries | Eager p50 | PostingBlockMax p50 | p50 gain | Eager p95 | PostingBlockMax p95 | Ordered mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus | 3,633 | 323 | 6.000 µs | 5.375 µs | 10.42% | 24.083 µs | 16.667 µs | 0 |
| SciFact | 5,183 | 1,109 | 33.875 µs | 24.000 µs | 29.15% | 49.333 µs | 45.542 µs | 0 |
| TREC-COVID | 100,000 | 50 | 765.625 µs | 723.833 µs | 5.46% | 873.334 µs | 822.041 µs | 0 |

The geometric mean p50 improvement across the three datasets is approximately `14.4%`.
Posting-element visits and bound evaluations were identical between both exact plans in every
dataset. The measured difference therefore isolates the compressed-metadata planning cost.

## Qualification

An earlier four-depth TREC-COVID research aggregate improved only `1.80%`, below the original
automatic `10%` TREC gate. Production promotion is therefore recorded as an explicit engineering
decision supported by the stable-worktree Top-20 workload, the cross-dataset aggregate, improved
p95, unchanged physical work counts and zero ordered mismatches. No claim is made that every Sparse
depth or workload is at least 10% faster.
