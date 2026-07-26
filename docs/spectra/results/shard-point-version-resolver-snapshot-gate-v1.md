# Shard Point-Version Resolver Real-Snapshot Gate V1

## Scope

This gate compares the adaptive on-demand resolver against the original eager
materialized owner map. Both plans execute the same Dense and Sparse exact
streams and the same dynamic WRRF policy.

The input snapshots are immutable and shared outside the worktree:

- SciFact: 5,183 documents and 1,109 queries;
- NFCorpus: 3,633 documents and 323 test queries;
- BAAI/bge-small-en-v1.5 Dense vectors, 384 dimensions;
- Qdrant/bm25 finite non-negative Sparse impacts;
- Top-20, WRRF `k=60`, physical result batch 64;
- 2, 4 and 8 Segments;
- three counterbalanced repetitions per query under the `perf` profile.

The test-only runner is
`collection::exact_rrf::snapshot_gate::shard_point_version_resolver_snapshot_gate`.
Production has no resolver-policy switch and no dependency on local snapshots.

## Correctness

Every dataset and Segment count produced zero ordered Top-20 mismatches. The
change preserves the exact result contract.

## Performance

### SciFact

All 1,109 queries required continuation beyond the first exact Top-20 prefix.

| Segments | On-demand p50 change | On-demand p95 change |
|---:|---:|---:|
| 2 | +6.2% | +3.9% |
| 4 | +13.4% | +1.7% |
| 8 | +22.9% | +6.5% |

### NFCorpus

The continuation rate was 86.7%; 13.3% of queries closed at the first exact
Top-20 prefix.

| Segments | On-demand p50 change | On-demand p95 change |
|---:|---:|---:|
| 2 | +6.7% | -2.0% |
| 4 | +15.5% | +4.1% |
| 8 | +17.4% | +7.2% |

Positive percentages are regressions. The generated local JSON files and the
retired test-only runner were removed after the decision was frozen. This
compact report is the retained historical artifact; Git history preserves the
deleted implementation.

## Decision

Do not promote the adaptive resolver to Production.

The synthetic correlated cohort demonstrated that avoiding a full owner-map
scan can be valuable when the first WRRF prefix closes. The real snapshots show
that the current physical rent is too high:

1. the initial prefix performs cross-Segment point probes;
2. most real queries then require continuation;
3. continuation materializes the complete owner map anyway;
4. the probe cost grows with Segment count and is not recovered.

This is a negative result for the current physical policy, not for exact ranking
or dynamic WRRF. Production should retain eager materialization until a cheaper
authoritative-identity primitive exists below the resolver, or a deterministic
zero/near-zero-cost signal can select on-demand resolution without changing
results. TREC-COVID 100K was not generated after both real datasets failed the
gate.
