# Exact Shard Merge NFCorpus Gate V1

## Scope

This gate compares:

- baseline: Production ExactRankSession with eager Shard point-version
  resolution;
- candidate: fixed-generation `ExactShardMergeState` with no owner map and no
  per-identity cross-Segment probe.

Both binaries execute the same Dense PVS, Sparse PostingBlockMax, dynamic WRRF,
Top-20, `k=60`, API, and exact tie rules. The only intended difference is the
Shard generation/merge architecture.

The dataset is the local NFCorpus snapshot:

- 3,633 documents;
- 323 Queries;
- Dense 384 dimensions;
- finite non-negative Sparse impacts;
- one warm-up sweep and five counterbalanced measured rounds;
- 1,615 exact responses per binary and topology.

The release binaries were built independently after invalidating the shared
Cargo target artifacts:

```text
baseline SHA-256:
78e47c1b5675d0adf9826c1dd15c432d0813e1fcd9c0de9070779c2bed3bf68f

candidate SHA-256:
aa2df193af44706003cb1561d1e22de5b6060cd6b8724e1f0685a9ed21123811
```

## Correctness

Both tested topologies produced:

```text
ordered Top-20 mismatch: 0
request errors:          0
exhaustive fallback:     0
```

The physical Dense/Sparse work was effectively unchanged. The speedup comes
from removing the failed Shard resolver work, not from weakening the exact
contract or changing the bottom kernels.

## Latency and throughput

### One Shard, two observed Segments

| Metric | Baseline | ExactShardMerge | Change |
|---|---:|---:|---:|
| p50 | 1.365 ms | 1.092 ms | -19.97% |
| p95 | 2.973 ms | 2.598 ms | -12.61% |
| p99 | 3.823 ms | 3.327 ms | -12.97% |
| mean | 1.614 ms | 1.322 ms | -18.07% |
| throughput | 618.7 QPS | 754.9 QPS | +22.02% |

Paired Query win rate was 96.59%. The Query-cluster bootstrap 95% confidence
interval for mean latency delta was `[-303.792us, -280.320us]`.

### Four Shards, eight observed Segments

| Metric | Baseline | ExactShardMerge | Change |
|---|---:|---:|---:|
| p50 | 1.351 ms | 1.162 ms | -13.99% |
| p95 | 2.844 ms | 2.646 ms | -6.97% |
| p99 | 3.453 ms | 3.187 ms | -7.71% |
| mean | 1.571 ms | 1.383 ms | -11.94% |
| throughput | 635.8 QPS | 721.9 QPS | +13.54% |

Paired Query win rate was 91.02%. The Query-cluster bootstrap 95% confidence
interval for mean latency delta was `[-196.188us, -179.446us]`.

### Four Shards, eight observed Segments, four concurrent Queries

| Metric | Baseline | ExactShardMerge | Change |
|---|---:|---:|---:|
| p50 | 3.197 ms | 3.264 ms | +2.10% |
| p95 | 5.890 ms | 6.071 ms | +3.07% |
| p99 | 7.365 ms | 7.563 ms | +2.68% |
| mean | 3.441 ms | 3.472 ms | +0.91% |
| throughput | 1,152.4 QPS | 1,139.0 QPS | -1.16% |

The Query-cluster bootstrap 95% confidence interval for mean latency delta was
`[-25.948us, +89.162us]`, crossing zero. This cohort is therefore statistically
neutral, remains inside the 5% tail-latency guardrail, and provides no evidence
of reader serialization. It does not demonstrate a concurrency throughput
gain. At this load the shared Qdrant execution pools and bottom kernels dominate
the removed resolver work.

## HTTP and persistence smoke

A separate two-Shard, 512-document HTTP smoke passed with zero ordered
mismatches for:

- native exact base Query;
- payload filter;
- explicit-consistency fallback;
- overwrite and delete;
- filtered Query after overwrite/delete;
- process restart and WAL recovery;
- filtered and explicit-consistency Queries after restart.

The native cases reported `exhaustiveFallback=false`.

## Decision

The candidate passes the available real-snapshot correctness gate, the
single-Query latency gate, and the concurrent no-regression guardrail. The
concurrency throughput target is not met and must not be claimed.

The right response is not another scheduler: physical Dense/Sparse work is
unchanged, and the concurrent confidence interval is neutral. The architecture
remains eligible for engineering verification because it materially improves
the unsaturated path, removes a failed layer, and does not exceed the frozen
tail-latency limit.

This is not evidence for retaining two implementations. If final verification
passes, the lower Qdrant lifecycle invariant and `ExactShardMergeState` become
the sole research path; the resolver stays deleted.

SciFact and TREC-COVID were not regenerated in this run. Their former local
snapshots had been intentionally deleted to reclaim disk space. No replacement
large corpus was created merely to satisfy this gate.

## Production promotion verification

The final Production tree additionally passed:

```text
cargo +nightly fmt --all -- --check
cargo +nightly check --workspace --all-targets
Shard tests:             87 / 87
Collection exact tests:   5 / 5
API V1.1 contract tests: 16 / 16
```

A fresh two-Shard HTTP run from the Production worktree again produced zero
ordered mismatches for base, filter, overwrite, delete, explicit-consistency
fallback, process restart, and post-restart persistence. Native cases reported
`exhaustiveFallback=false`.

The promotion changes no HTTP request/response field, guarantee wording,
WRRF policy, Dense/Sparse kernel, or tie rule.
