# Exact Shard Merge V1 — Strict Design Review

## Verdict

Approved as a research candidate after lifecycle repair and re-review.

Independent source review found that the current optimizer cancellation path
can produce the exact hidden-newer-version state that invalidates score-first
streaming. It also found that `ExactSegmentReadSet` currently freezes ordinary
updates but not optimizer publish/rollback.

The blocking review required:

1. deterministic reproduction of the counterexample;
2. Qdrant lifecycle repair using its existing Proxy propagation primitive;
3. shared generation pinning for exact sessions;
4. the complete P0 invariant suite.

Those gates now pass. The implementation uses the existing Qdrant propagation
primitive, an owned generation read guard, and one batch merge state. The
failed resolver implementation and its forwarding constructors were deleted;
no compatibility layer remains.

This condition is substantive. A resumable K-way merge cannot safely emit an
older high-scoring copy if a newer, lower-scoring visible copy may remain
unseen deep in another Segment. No heap algorithm can infer that fact from a
finite score prefix. The original design was not sound on the current
lifecycle. The revised design first makes the lifecycle prevent that state,
then proves the result.

Subject to that gate, this is materially cleaner than both existing
alternatives:

```text
eager resolver:
full Shard identity scan before useful work

adaptive resolver:
candidate probes, then often the same full scan

ExactShardMerge:
consume existing exact Segment batches and merge them
```

The design introduces no new index or general scheduling framework. The one
new state, `ExactShardMergeState`, is necessary because Qdrant's
`BatchResultAggregator` is fixed Top-K and cannot pause/resume.

## Review findings

### R1 — Hidden newer version can invalidate streaming output

Severity: blocking.

Failure example:

```text
Segment A: id 7, version 3, score 0.95
Segment B: id 7, version 4, score 0.40
```

A score-first merge sees version 3 first. Deduplicating after emission is too
late.

Current source establishes a valid construction:

```text
Proxy + COW
-> new version written to COW
-> Proxy masks old version
-> optimizer cancellation calls unwrap_proxy
-> wrapped old version becomes visible again without propagated delete
```

Resolution:

- reproduce the construction before changing code;
- replace cancellation unwrapping with
  `ProxySegment::propagate_to_wrapped()` followed by atomic replacement;
- keep the old Proxy state installed if propagation fails;
- rerun P0 across update, deferred, Proxy, cancellation, restart and failure
  paths;
- do not hide the issue with post-query validation, retry, or fallback.

Status: unresolved / blocking until lifecycle tests pass.

### R2 — ExactRankSession does not freeze optimizer transitions

Severity: blocking.

`ExactSegmentReadSet` holds `LocalShard::update_operation_lock`, while
optimizer finalization holds `LockedSegmentHolder`'s independent update lock.
The existing exact session therefore blocks ordinary updates but does not pin
all Segment transitions.

Resolution:

- convert the Segment update exclusion primitive into a
  shared-read/exclusive-write generation barrier;
- exact sessions hold an owned shared guard; update, snapshot, optimizer
  publish and rollback paths retain exclusive access;
- acquire the LocalShard update lock before the Segment generation guard;
- allow several exact sessions to share the generation;
- remove post-validation and rerun concepts;
- fail only on errors inside that frozen generation.

Status: resolved in revised design, pending implementation and concurrency
evidence.

### R2a — Holding the old exclusive update lock would serialize exact queries

Severity: high.

Reusing the current mutex as a long-lived Query guard would prevent optimizer
transitions, but it would also serialize all exact queries.

Resolution:

- use a read/write barrier, not an exclusive Query mutex;
- measure concurrent-query throughput and writer wait;
- keep slow optimizer build outside the exclusive publish section.

Status: resolved in design.

### R2b — Rollback propagation can deadlock or partially publish

Severity: blocking.

`propagate_to_wrapped()` performs Segment work. Calling it while holding the
holder write lock creates an unnecessary lock-order risk. Unwrapping some
Proxies after an earlier propagation and later error would expose a partial
generation.

Resolution:

- hold the generation barrier exclusively for the whole rollback;
- collect Proxy handles under a short holder read;
- release holder access before propagation;
- propagate every Proxy;
- only after all propagation succeeds acquire a short holder write and replace
  all Proxies;
- on failure leave all Proxies installed and return an error.

Status: resolved in design, pending injected-failure tests.

### R3 — Reimplementing Qdrant aggregation would create semantic drift

Severity: high.

The fork already has:

- `BatchResultAggregator` maximum-version selection;
- `SearchResultAggregator` identity suppression;
- an established `ScoredPoint` representation;
- SegmentHolder update/deletion rules.

Copying similar logic into a Stratumind-only helper would eventually diverge.

Resolution:

- keep `BatchResultAggregator` as the finite-result semantic oracle;
- reuse the existing exact-stream comparator already used by
  `KWayExactScoreStream`;
- use Qdrant's `AHashSet` identity-suppression pattern;
- do not pretend that maximum-version selection over a finite candidate matrix
  solves a hidden future version in a resumable stream;
- do not create a generic aggregation framework.

Status: blocked on P0; `BatchResultAggregator` remains the semantic oracle, but
the narrower stream merge is valid only after the authority invariant holds.

### R4 — Fixed Top-K aggregator cannot be reused unchanged

Severity: medium.

`BatchResultAggregator` assumes a finite result matrix and learns maximum
versions before updating each Top-K queue. ExactRankSession requires an
unbounded, resumable order.

Resolution:

- reuse semantics and primitives, not the fixed container;
- permit exactly one owned resumable merge state;
- keep scoring and storage outside that state.

This is the one justified new component for which user approval is required
before code implementation.

Status: awaiting implementation approval.

### R5 — Per-point API can conceal scheduling overhead

Severity: high.

The current Shard stream exposes `next_result()`. Although Segment sources
buffer some results, a future refactor could accidentally attach a runtime
handoff to every call.

Resolution:

- make `next_batch(max_results)` the physical API;
- allow only a private in-memory iterator adapter for the existing WRRF
  boundary;
- test that runtime tasks scale with source refill count, not emitted point
  count;
- forbid production `sync_channel`.

Status: resolved.

### R6 — Parallel fan-out would expand scope without evidence

Severity: medium.

Qdrant ordinary search fans out Segment work, but ExactRankSession currently
runs short reads inline on one reserved coordinator to avoid nested runtime
waits. Adding a new parallel executor while replacing the resolver would mix
two changes and risk thread-pool self-deadlock.

Resolution:

- retain the current Qdrant search runtime and inline bounded pulls;
- measure Segment refill time separately;
- consider batched fan-out only in a later design if profiling shows it
  dominates.

Status: resolved.

### R7 — Initial one-head pull may underuse batch kernels

Severity: medium.

A K-way merge logically needs one head per source, but Dense PVS and Sparse
PostingBlockMax are optimized for batches. Pulling exactly one physical result
from every Segment may create avoidable reader setup.

Resolution:

- distinguish one logical head from physical prefetch depth;
- keep prefetch internal;
- calibrate `1` versus the existing source batch size;
- choose one fixed winner;
- do not add a Query router or API option.

Status: resolved.

### R8 — Scores and versions can be erased too early

Severity: high.

WRRF needs identities, but Shard and Collection channel merges still need
scores. Version agreement must also be checked before identity emission.

Resolution:

- keep full `ScoredPoint` through Segment-to-Shard merge;
- keep score through Shard-to-Collection channel merge;
- erase score only after the complete selected-corpus channel order exists;
- retain version long enough to validate cross-channel agreement for final
  results.

Status: resolved.

### R9 — Compatibility wrappers would preserve the failed architecture

Severity: high.

Keeping `open_with_resolver`, resolver policies, aliases, force modes, and
feature flags would make the new path harder to understand and test.

Resolution:

- one-call-site migration;
- delete failed runtime code and forwarding constructors;
- no deprecated alias;
- no dual Production path;
- Git history plus one compact negative-result report is sufficient archive.

Status: resolved.

### R10 — Historical artifacts can become active-tree garbage

Severity: medium.

Raw local JSON and duplicate plan/review documents are useful during an
experiment but not as permanent active architecture.

Resolution:

- preserve one compact, versioned negative-result report;
- do not commit generated local JSON unless required for paper reproducibility;
- delete superseded active design/review documents after implementation;
- keep reproducible runner code only for active candidates.

Status: resolved.

### R11 — Conflicting duplicates may be detected only when consumed

Severity: medium.

Without a full owner scan, corruption outside the consumed prefix is not
eagerly diagnosed.

Resolution:

- exact result semantics are defined for valid Qdrant storage;
- P0 proves valid lifecycle invariants;
- contradictions encountered in the consumed prefix fail closed;
- do not charge every healthy query for a complete corruption audit.

Status: accepted trade-off.

### R12 — Collection-level merging remains necessary

Severity: medium.

Removing the resolver must not accidentally remove the merge across selected
Shards or send Shard-local ranks directly into WRRF.

Resolution:

```text
Segment exact batches
-> exact Shard score order
-> exact selected-corpus channel score order
-> identity rank stream
-> dynamic WRRF
```

Status: resolved.

## Architecture compliance checklist

- [x] Production API V1.1 unchanged.
- [x] Dynamic WRRF and proof semantics unchanged.
- [x] Dense PVS stays in the Segment kernel.
- [x] Sparse PostingBlockMax stays in the Segment kernel.
- [x] Qdrant ReadView, scratch, runtime, counters, cancellation reused.
- [x] No new persistent index.
- [x] No new global owner cache.
- [x] No new thread pool.
- [x] No per-point scheduler.
- [x] No compatibility layer.
- [x] Failed adaptive resolver scheduled for deletion.
- [x] One necessary new resumable merge state only.
- [x] Existing Proxy propagation primitive selected; no owner index invented.
- [x] Concurrent Query generation pin uses a shared barrier.
- [ ] Optimizer cancellation counterexample reproduced.
- [ ] Safe propagate-then-replace rollback implemented.
- [ ] Exact sessions pin optimizer transitions without serializing readers.
- [ ] P0 lifecycle invariant proven.
- [ ] Real-snapshot performance gate passed.

## Required evidence before implementation promotion

1. A deterministic before/after optimizer-cancellation counterexample.
2. Injected propagation-failure evidence showing no partial unproxy.
3. Long-query tests spanning optimizer start, publish and rollback.
4. P0 lifecycle test output with all transitions enumerated.
5. Source-level proof mapping each invariant to the Qdrant update/Proxy code.
6. Randomized complete-order parity against exhaustive Qdrant results.
7. Dynamic WRRF zero-mismatch report.
8. Runtime-task and buffer-refill counters proving batch scheduling.
9. Concurrent-reader and optimizer-writer wait measurements.
10. NFCorpus, SciFact, and TREC-COVID 100K paired latency results.
11. A deletion diff showing that resolver code and compatibility hooks are
   actually gone.

## Final decision

Proceed with lifecycle repair and P0 only.

The original P0 assumption has already failed. The revised route is accepted
because it moves authority maintenance to the Qdrant lifecycle, reuses the
existing safe Proxy propagation primitive, adds no per-query owner structure,
and pins one immutable generation without chasing future versions.

If the repaired P0 passes, proceed with the narrow merge replacement. If it
still fails, stop and bring the remaining concrete counterexample to the user.
Do not smuggle an ownership resolver back as a fallback or compatibility
layer.
