# Exact Shard Merge V1

## Status

Promoted into the Production implementation after repairing a blocking Qdrant
lifecycle counterexample and passing the frozen correctness, API, and latency
gates. The original owner-invariant assumption was false:

```text
Proxy optimization starts
-> a lower-scoring new version is written to the COW Segment
-> optimizer is cancelled
-> unwrap_proxy restores the wrapped Segment without propagating Proxy deletes
-> old high-scoring and new low-scoring versions are both indexed-visible
```

The Production implementation now:

- pins one Segment generation for the complete exact Query;
- uses Qdrant's existing `ProxySegment::propagate_to_wrapped` before rollback;
- merges exact Segment batches through owned `ExactShardMergeState`;
- removes `ShardPointVersionResolver`, per-identity Segment probes, and owner
  map materialization without retaining compatibility wrappers.

The repaired P0 regression, Shard tests, Collection exact-WRRF tests, and the
first real NFCorpus gates pass. The measured result is recorded in
`docs/spectra/results/exact-shard-merge-nfcorpus-gate-v1.md`.

This design replaces the failed `ShardPointVersionResolver` research path. It
does not change Production API V1.1, Dense PVS, Sparse PostingBlockMax,
ExactRankSession, dynamic WRRF, score ordering, identity tie-breaking, or
fail-closed semantics.

The only new long-lived execution state permitted by this design is:

```text
ExactShardMergeState
```

It is the resumable form that Qdrant's fixed Top-K Segment aggregation does not
currently provide. It is not a new index, ownership table, cache hierarchy,
thread pool, MVCC layer, or compatibility adapter.

## 1. Problem

One channel is already exact inside each Segment:

```text
Dense:
PVS / Scalar / Compact bound state
-> temporary SegmentReadView
-> RawScorer batch refinement
-> exact ordered Segment batches

Sparse:
PostingBlockMax state
-> temporary SegmentReadView
-> Qdrant compressed posting + SearchScratch
-> exact ordered Segment batches
```

The remaining Shard-level job is conceptually small:

```text
N exact Segment orders
-> one exact Shard order
```

The current research branch makes it expensive by inserting
`ShardPointVersionResolver` between the Segment kernels and the merge:

```text
Segment batch
-> candidate identity × every Segment version probe
-> optional full owner-map materialization
-> K-way merge
```

Real-snapshot measurements showed that this layer adds work without reducing
Dense or Sparse physical work:

```text
SciFact: p50 +6.2% to +22.9%
NFCorpus: p50 +6.7% to +17.4%
ordered Top-20 mismatch: 0
```

The failure is architectural rather than a missing micro-optimization. A
shallow query pays cross-Segment membership probes; a continuing query then
pays the complete owner scan as well.

## 2. Existing Qdrant mechanisms to reuse

The implementation must reuse the following mechanisms directly.

| Responsibility | Existing mechanism | Decision |
|---|---|---|
| Ordinary update exclusion | `LocalShard::update_operation_lock` | Reuse unchanged |
| Segment generation transition | `LockedSegmentHolder` update lock | Upgrade to a shared-read/exclusive-write generation barrier |
| Safe Proxy rollback | `ProxySegment::propagate_to_wrapped` | Reuse in optimizer cancellation; do not invent an owner structure |
| Segment reader lifetime | `ReadSegmentHandle` / temporary `SegmentReadView` | Reuse unchanged |
| Dense physical work | `DenseRankState` and Qdrant `RawScorer::score_points` | Reuse unchanged |
| Sparse physical work | `PostingBlockMaxState`, compressed posting, `SearchScratch` | Reuse unchanged |
| Runtime | existing Qdrant search runtime and current reserved coordinator | Reuse unchanged |
| Counters/cancellation | `QueryContext`, hardware counters, `AtomicBool` | Reuse unchanged |
| Fixed Top-K aggregation | `BatchResultAggregator` maximum-version rule | Keep as the Qdrant semantic oracle; its finite container is not used in the hot path |
| Identity suppression | Qdrant `SearchResultAggregator` identity set | Use the same `AHashSet` pattern |
| Exact stream ordering | existing `exact_score_order` used by `KWayExactScoreStream` | Promote one shared comparator; do not add another order definition |

Qdrant documents that a point may physically occur in more than one Segment
and that search performs deduplication. Each Segment also keeps per-point
versions for conflict resolution:

- <https://qdrant.tech/documentation/manage-data/storage/>
- `lib/shard/src/search_result_aggregator.rs`
- `lib/shard/src/segment_holder/mod.rs`
- `lib/shard/src/proxy_segment/segment_entry.rs`

The current ExactRankSession bypasses ordinary fixed Top-K
`SegmentsSearcher`, so it must adapt those aggregation semantics to a
resumable stream. It must not recreate Qdrant storage ownership outside
Qdrant.

## 3. Frozen correctness contract

### 3.1 One fixed Query generation

The current `ExactSegmentReadSet` holds only
`LocalShard::update_operation_lock`. That excludes ordinary point updates, but
optimizer publish and rollback use the independent
`LockedSegmentHolder` update lock. The current read set therefore does **not**
freeze every Segment-generation transition.

The corrected boundary uses two existing Qdrant lifecycle locks in one fixed
order:

```text
exact Query:
  LocalShard update_operation_lock: shared
  -> Segment generation barrier: shared, owned
  -> clone Segment handles

ordinary update:
  LocalShard update_operation_lock: exclusive
  -> Segment generation barrier: exclusive
  -> mutate holder/Segments

optimizer publish or rollback:
  Segment generation barrier: exclusive
  -> replace holder generation
```

`LockedSegmentHolder`'s current exclusive `updates_mutex` becomes a
shared-read/exclusive-write generation barrier:

- existing update, snapshot, optimizer publish and rollback paths take the
  exclusive side;
- each exact session takes one owned shared guard;
- multiple exact sessions remain concurrent;
- the optimizer may build a new Segment in the background, but its short final
  publish/rollback waits until readers of the old generation finish.

The `ExactSegmentReadSet` owns both shared guards and one fixed handle set for
its complete lifetime. Thus:

```text
Q1 starts on generation g
-> ordinary updates wait
-> optimizer may build g+1 but cannot publish or roll back g
-> every Q1 batch reads the same handles from g
-> Q1 finishes
-> optimizer may atomically publish g+1
-> a later Q2 sees either complete g or complete g+1
```

There is no post-query generation validation, rerun, or future-version chase.
A long Query intentionally keeps its starting generation. New indexes and
updates become visible only to later Queries.

### 3.2 Per-Segment order

Every Segment source must produce:

```text
score descending
then external point identity ascending
```

The source is strictly resumable. EOF, cancellation, storage failure, and a
certificate violation are distinct terminal states.

### 3.3 Shard order

The Shard result must equal exhaustive evaluation of every indexed-visible
point in the frozen Shard, followed by the same Qdrant version/identity
semantics and the same total ordering.

Dense and Sparse remain independent. They share neither score nor bound. Each
channel obtains its own `ExactShardMergeState`.

### 3.4 Indexed-visible authority invariant

After the lifecycle repair, the design relies on one narrower Qdrant storage
invariant:

> In one frozen, valid Shard generation, an external identity has at most one
> authoritative non-deferred value. Any simultaneously visible duplicate is a
> physically duplicated representation of that same authoritative value and
> therefore has the same version and channel score.

Physical duplicates may exist. The invariant concerns the indexed-visible
authority observed by an exact Query, not uniqueness of bytes on disk.

The normal update and successful optimizer paths already aim to preserve this
invariant:

- `SegmentHolder::apply_points` deletes obsolete non-deferred copies before
  applying an update;
- `find_points_to_update_and_delete` preserves the old non-deferred copy only
  while the newer copy is deferred and therefore excluded from indexed
  search;
- `ProxySegment` masks wrapped points changed during optimization;
- optimizer finalization publishes a replacement under the Segment update
  lock.

The current optimizer cancellation path violates it because `unwrap_proxy`
restores wrapped Segments without first calling
`ProxySegment::propagate_to_wrapped`. Snapshot unproxy already uses that
mature Qdrant primitive. The optimizer rollback path must use the same
propagation rule before the invariant can be claimed.

## 4. Qdrant lifecycle repair and P0 gate

### 4.1 Safe optimizer rollback

Replace `unwrap_proxy` with one rollback operation; do not leave the old name
as an alias.

The operation is:

```text
acquire Segment generation barrier exclusively
-> collect the live Proxy handles
-> for every Proxy, call propagate_to_wrapped()
-> if any propagation fails: leave every Proxy installed and fail closed
-> after all propagation succeeds, replace Proxies with wrapped Segments
-> release barrier
```

Important lock discipline:

- never hold the Segment holder write lock while
  `propagate_to_wrapped()` performs Segment work;
- keep the generation barrier for the complete propagation-and-replacement
  transaction;
- replacement acquires the holder write lock only for the short final swap;
- a partially propagated failure must not expose the wrapped Segments;
- deferred updates must preserve the existing old indexed-visible copy until
  the new copy is indexed;
- ordinary update flush dependencies and WAL/restart behavior remain Qdrant's
  existing durability mechanism.

This moves work to the rare cancellation/failure path. Healthy query execution
does not scan identities, probe versions, or rebuild ownership.

### 4.2 P0 mandatory invariant gate

P0 is an implementation precondition for the merge, not an optional test
phase. First add a deterministic test that fails on the current implementation:

```text
old version v has a high channel score
-> start optimization and install Proxy + COW
-> overwrite with lower-scoring version v+1
-> confirm v+1 is non-deferred and Proxy masks v
-> cancel optimization
-> freeze an ExactSegmentReadSet
-> enumerate indexed-visible copies
```

Before the repair, both versions are expected to be visible. After the repair,
only the authoritative new version may be visible.

Build a Shard-level invariant harness using Qdrant's real update and optimizer
paths. At every frozen read boundary, enumerate indexed-visible copies and
assert:

```text
for every external identity:
  authoritative non-deferred values <= 1

if several visible physical copies exist:
  version is identical
  Dense value is identical
  Sparse value is identical
  payload/filter visibility is identical
```

Required transitions and interleavings:

1. ordinary insert and overwrite in an appendable Segment;
2. move from immutable to appendable Segment;
3. newer deferred copy while the older indexed copy remains searchable;
4. deferred indexing completion and old-copy retirement;
5. update and delete while Proxy optimization is active;
6. optimizer final swap;
7. high-score old version, low-score new version, optimizer cancellation;
8. low-score old version, high-score new version, optimizer cancellation;
9. exact session starts before optimizer start, then spans finalization;
10. exact session starts while Proxy+COW is installed, then spans
    cancellation;
11. snapshot proxy/unproxy;
12. propagation failure without partial unproxy;
13. delete and overwrite by identity;
14. delete and overwrite by filter;
15. update partial failure;
16. restart/WAL recovery before and after optimization completion or
    cancellation;
17. interrupted optimization recovery;
18. same-version physical duplicate;
19. several concurrent exact sessions while publish waits.

The harness must compare:

```text
frozen visible universe
==
ordinary exhaustive Qdrant search semantics with DeferredBehavior::Exclude
```

Decision:

- The pre-repair counterexample must fail deterministically.
- The same test and all lifecycle cases must pass after the repair.
- Gate passes: implement the merge below.
- Gate still fails for any valid lifecycle: stop. Do not restore the old
  resolver under a new name. Report the remaining counterexample before
  introducing any authoritative ownership structure.
- Gate fails only for deliberately corrupted storage: production behavior may
  fail closed when the conflict is observed; no full-query corruption scan is
  added to the hot path.

## 5. ExactShardMergeState

### 5.1 Responsibility

`ExactShardMergeState` does exactly four things:

1. retain one buffered batch per Segment source;
2. retain one current head per non-exhausted source;
3. merge heads by Qdrant's exact total order;
4. suppress equivalent physical duplicates and detect contradictions.

It does not:

- calculate Dense or Sparse scores;
- inspect arbitrary point identities in other Segments;
- own Segment read guards;
- schedule one task per point;
- build an identity-to-owner map;
- compare Dense and Sparse scores;
- run WRRF;
- choose a runtime plan.

### 5.2 State

```text
ExactShardMergeState
  sources: Vec<SegmentRankSource>
  heads: BinaryHeap<SourceHead>
  emitted: AHashSet<PointId>
  output: Vec<ScoredPoint>
  batch_size
  cancellation
  telemetry
  terminal_error
```

`SegmentRankSource` retains owned rank state indirectly through the existing
Dense/Sparse closures and stores the latest returned `Vec<ScoredPoint>` plus a
read position. It must not convert each reply into another intermediate
collection merely to consume it.

No Segment guard, posting iterator, `RawScorer`, or scratch buffer survives a
batch call.

### 5.3 Opening

Opening a merge:

1. validates positive batch size;
2. creates one Dense or Sparse source per frozen Segment;
3. obtains one logical head from every source;
4. validates each source's first batch;
5. heapifies all heads once.

Only the minimum information necessary to establish a global head is required
from every Segment. Initial physical pull depth remains an internal,
benchmarked constant; it is not an API field and cannot affect results.

### 5.4 Batch advance

`next_batch(max_results)` is the primary API:

```text
while output.len < max_results:
  take greatest global head
  collect equivalent heads for the same score + identity
  validate that equivalent copies agree on version and score
  emit identity once
  advance only the contributing sources
  refill a source only when its local buffer is empty
```

A batch may be shorter than requested when:

- exact EOF is reached;
- cancellation/error occurs, in which case no successful batch is returned;
- the caller requested zero, which is rejected before execution.

The old `next_result()`-driven production loop is removed. If the existing
iterator-only WRRF boundary still needs individual identities, one private
zero-allocation adapter consumes the already-produced Shard batch in memory.
It must never create one runtime task or one channel message per identity.

### 5.5 Duplicate and version rules

For equivalent heap heads with the same score and external identity:

```text
same version + same score
-> emit one, suppress the rest

different version
-> inconsistent frozen snapshot, fail closed

same version + different score
-> contradictory physical representation, fail closed
```

If an already emitted identity appears later, fail closed. Under the P0
invariant this cannot occur in valid storage; the check is a cheap corruption
guard, not an ownership algorithm.

Versions never participate in user-visible ranking. Source index is only an
internal deterministic tie-breaker and never changes semantic order.

### 5.6 Source validation

Every source batch is validated once:

- all scores finite;
- descending score and ascending external identity;
- version available;
- no point after source EOF;
- no success after sticky error;
- batch state belongs to the frozen read set.

Validation is performed when the batch enters the source buffer, not once per
consumer read.

## 6. Scheduling and Qdrant kernel use

### 6.1 Production execution

Production keeps the current ExactRankSession boundary:

```text
Qdrant search runtime coordinator
-> bounded Segment reader call
-> temporary SegmentReadView
-> Qdrant Dense/Sparse kernel
-> owned batch reply
-> in-memory Shard merge
```

The coordinator already occupies a reserved Qdrant search worker. Segment
batch work therefore runs inline on that worker through
`ExactBatchExecutor::inline_on_current_worker()`.

Forbidden:

- nested Tokio spawn followed by blocking wait;
- `sync_channel` in the production hot path;
- permanent worker per Segment or channel;
- worker-slot reservation proportional to Segment count;
- per-point tasks;
- a second thread pool.

The test-only cross-thread executor remains only if it continues to prove that
rank state is reader-independent. It is not a compatibility path and is
compiled out of Production.

### 6.2 Kernel preservation

Shard merging must not flatten Segment work into a generic exhaustive scan.

Dense source refills continue through:

```text
DenseRankState
-> PVS/Scalar/Compact bound pruning
-> RawScorer::score_points batch
-> exact Segment prefix
```

Sparse source refills continue through:

```text
PostingBlockMaxState
-> Qdrant compressed posting range decode
-> SearchScratch reuse
-> exact Segment prefix
```

The merge consumes the existing batches. It does not reinterpret their
certificates or duplicate their scoring kernels.

### 6.3 Allocation discipline

- reuse source `Vec<ScoredPoint>` capacity;
- reuse heap capacity for the session;
- use Qdrant `AHashSet` for emitted identities;
- reserve only from observed batch/source counts;
- do not allocate a full-corpus owner map;
- do not clone query vectors, filters, or batches after source construction;
- preserve existing Qdrant scratch-pool limits.

### 6.4 Code ownership rule

If profiling reveals that an existing Qdrant primitive lacks one necessary
batch capability, implement it in the crate that owns the data:

```text
compressed posting access -> sparse crate
Dense score/refinement     -> segment/quantization crate
Segment ReadView           -> segment/shard reader boundary
Shard score merge          -> shard crate
WRRF proof                 -> reciprocal-rank-fusion module
```

Do not work around a missing bottom capability with a second representation in
Collection code. The change must have a direct caller, tests, and measured
benefit. When the replacement is complete, remove the superseded entry point
in the same migration; do not leave a forwarding compatibility layer.

## 7. Integration

The active path becomes:

```text
Collection exact_rrf
  -> ExactSegmentReadSet per selected Shard
  -> ExactDenseShardStream / ExactSparseShardStream
       -> Segment Dense/PBM batch sources
       -> ExactShardMergeState
  -> existing Collection channel K-way merge
  -> DynamicRrfSession
  -> exact Top-K
```

The Collection-level merge remains necessary because selected Shards are
independent physical universes. Scores are erased only after all selected
Shard sources for one channel have been merged.

Dense and Sparse may share the frozen `ExactSegmentReadSet`, query
cancellation, and runtime reservation. They do not share ranking state.

## 8. Deletion and migration

This is a replacement, not a compatibility migration.

After P0 and correctness gates pass, delete:

- `lib/shard/src/point_version_resolver.rs`;
- `ShardPointVersionResolver`;
- `LazyOwners`, `MaterializedOwners`, `ResolverPolicy`;
- `resolve_batch`, `maybe_materialize`, `prepare_for_continuation`;
- `open_with_resolver` constructors;
- resolver telemetry fields;
- lazy/materialized force modes and their benchmark hooks;
- snapshot-gate execution code used only by the failed candidate;
- deprecated aliases and old constructor forwarding;
- production error/log strings that still call this path `native`.

Do not leave feature flags selecting the failed resolver.

Git history is the archive. Keep one compact negative-result report explaining
why the resolver lost. Do not keep generated local JSON, duplicate active
design documents, or dead benchmark entry points in the main tree.

The old design/review documents become unnecessary once this design is
implemented. Their historical conclusion must be preserved in the negative
result report, after which the active-plan copies are deleted rather than
marked as indefinitely deprecated.

## 9. Tests

### 9.1 Correctness

- P0 lifecycle invariant suite.
- Property test over random source count, batch size, pause point, score ties,
  same-version duplicates, identity shapes, and Segment completion order.
- Every complete Shard stream equals exhaustive Qdrant Segment results merged
  with the same total order.
- Batch sizes `1/2/7/32/64/257` produce identical output.
- Dense and Sparse complete streams independently match exhaustive order.
- Dynamic WRRF Top-20 has zero ordered mismatch against full Dense + Sparse
  WRRF.
- Filter, delete, overwrite, deferred point, Proxy, optimizer finalization,
  restart, multi-Segment, multi-Shard, cancellation, timeout, and sticky error.
- A contradictory duplicate fails closed.
- Ordinary Query updates block on `update_operation_lock`.
- Optimizer publish and rollback block on the shared-read/exclusive-write
  Segment generation barrier.
- A session started before optimization sees its original generation only.
- A session started in Proxy+COW state keeps that complete state until it
  ends.
- Several exact sessions acquire shared generation guards concurrently.

### 9.2 Scheduling

- no Segment guard survives `next_batch`;
- no resident worker survives a checkpoint;
- production uses no channel send/receive per batch or per point;
- runtime task count is independent of emitted identity count;
- physical Segment refills equal observed buffer exhaustion;
- generation-barrier acquisition creates no per-batch task or channel;
- optimizer build continues while exact readers are active; only the short
  publish/rollback phase waits;
- four/eight Segments do not trigger materialized fallback;
- Dense/Sparse kernel counters do not increase relative to the Production
  ExactRankSession baseline except for explicitly measured initial prefetch.

### 9.3 Performance

Datasets:

```text
NFCorpus
SciFact
TREC-COVID 100K
```

Segment layouts:

```text
1 / 2 / 4 / 8 Segments
```

Counterbalanced release runs compare:

1. Production eager owner-map ExactRankSession;
2. failed adaptive resolver, retained only as historical measurement;
3. ExactShardMerge V1;
4. stock Qdrant fixed Top-K as a physical-prefix reference, not an exact
   dynamic-WRRF competitor.

Mandatory:

- ordered Top-20 mismatch: zero;
- certificate violation: zero;
- full owner scans: zero;
- per-identity cross-Segment probes: zero;
- Dense scored points, PVS refinements, Sparse posting decode, and WRRF pulls
  do not increase beyond measured prefetch variance;
- p50 improves over Production on the three-dataset geometric mean;
- at least two datasets improve p50 by 5% or more;
- no dataset/Segment cohort regresses p95 by more than 5%;
- temporary memory is bounded by source buffers, heap heads, emitted prefix,
  and existing kernel scratch.
- exact-query p50/p95 includes generation-barrier acquisition;
- optimizer publish/rollback wait time is reported separately;
- concurrent exact-query throughput must not regress through accidental
  reader serialization.

If latency does not improve despite removing resolver work, profile before
adding anything. Do not retain a non-winning merge variant or stack another
policy on top.

## 10. Stop rules

1. If the repaired P0 still finds a valid Qdrant lifecycle with two
   conflicting indexed-visible values, stop and request approval for a new
   authority mechanism.
2. If exact equivalence fails, delete the candidate implementation; do not add
   fallback semantics to hide it.
3. If scheduling cost dominates, first batch or inline existing Qdrant calls;
   do not add a new runtime.
4. If a bottom kernel dominates, optimize that kernel separately; do not move
   its logic into the Shard layer.
5. If the performance gate fails, retain only the result report and return to
   the Production ExactRankSession baseline.

## 11. Implementation sequence

```text
P0.1  reproduce optimizer-cancellation counterexample on current code
P0.2  replace the exclusive update mutex with a generation read/write barrier
P0.3  make ExactSegmentReadSet own a shared generation guard
P0.4  replace unsafe unwrap with propagate-then-replace rollback
P0.5  pass the complete lifecycle invariant and restart/WAL suite
P1    share the existing exact-stream comparator and keep BatchResultAggregator as oracle
P2    implement ExactShardMergeState + batch property tests
P3    connect Dense and Sparse Segment batch sources
P4    remove ShardPointVersionResolver and continuation checkpoint
P5    run correctness, scheduling, concurrency, and real-snapshot gates
P6    delete dead research code/docs and prepare one focused promotion commit
```

No later phase begins if an earlier phase fails.
