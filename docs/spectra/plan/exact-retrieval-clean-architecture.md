# Exact Retrieval Clean Architecture

Status: promoted to Production V1.1.5 at `06438875e`.

## Frozen contract

Production exact retrieval remains:

```text
exact Dense rank order
+ exact Sparse rank order
→ dynamic WRRF
→ proof-based stop
```

The ordered Top-K, external-identity tie rule, cancellation behavior and API V1.1
request/response shape do not change.

## Sole production path

```text
HTTP
→ ExactRrfService
→ ExactHybridSession
→ ExactShardStream<SegmentRankPlan>
→ DenseRankState / PostingBlockMaxState
→ PerVectorScalar / Qdrant compressed posting
```

`DenseRankState` and `PostingBlockMaxState` own all resumable query state. A batch
temporarily borrows a `SegmentReadView`; no cursor borrows a Segment across
checkpoints and no Segment owns a resident worker.

`ExactShardStream<P>` contains the one shared Segment merge implementation.
`DenseRankPlan` and `SparseRankPlan` only initialize and advance their channel
state. Correctness fallbacks for unsupported Segment representations remain
part of the plan; historical adapters do not.

## Explicit profile

PVS construction is opt-in:

```json
{
  "exact_rank_config": {
    "profile": "dense_sparse_v1"
  }
}
```

The default is `disabled`. The profile is persisted in Collection and Segment
configuration and participates in Segment compatibility, so changing it causes
optimizer rebuilding. A Collection missing this field is disabled; it is never
classified by inspecting historical files.

## Deleted implementations

The active crate tree no longer contains the retired Dense Ball/Box/proposal
families, N-channel executors, standalone BMP/block-max experiments, HNSW
propagation prototypes, borrowed exact cursors, stream adapters, compatibility
aliases or the standalone research server. Their only archive is Git history
before this clean break, principally the parent `b26cbf157`.

Historical result documents remain evidence of the experiments that selected
PVS, PostingBlockMax, ExactRankSession and ExactShardStream.

## Clean-break verification

The clean architecture passed:

- nightly workspace formatting and `--all-targets` compilation;
- Dense and Sparse pause/resume order tests;
- all 87 Shard unit tests;
- all 27 dynamic WRRF tests;
- Collection exact-WRRF, PVS persistence/profile and HTTP V1.1 contract tests;
- two-Shard HTTP smoke covering filter, explicit consistency, overwrite,
  delete and restart persistence, with zero ordered Top-K mismatches.

The HTTP diagnostic strings remain the frozen V1.1 values even though the
internal types no longer use the historical `native` terminology.

## Lifecycle invariant

Within one frozen Segment generation, each external point identity has exactly
one authoritative visible version. Load deduplication, update serialization and
Proxy rollback propagation establish this before an `ExactShardStream` opens.
Tests must keep covering overwrite, delete, deferred copies, optimizer rollback
and conflicting-version fail-closed behavior; the merge heap is not a substitute
for this lifecycle invariant.
