# Exact Fusion Fixed-16 Production Promotion Gate V1

Date: 2026-07-26

Base: Production V1.1.5 at `f49a6fccf`

Candidate branch: `codex/production-fixed16-fusion-v1.1.6`

## Change

The clean Production path now preserves batches through the complete exact
fusion boundary:

```text
ExactShardStream::next_batch
→ channel-global KWayExactScoreStream
→ ChannelRankBatch
→ DynamicRrfState::observe_batch
→ exact WRRF certificate
```

Dense and Sparse remain independent exact score orders. Scores are erased only
after each channel has been globally merged across its selected Shards.
Production uses one fixed physical batch size of 16; there is no runtime Router,
environment selector or legacy cadence adapter.

The API, WRRF formula, identity tie rule, snapshot scope and fail-closed
semantics are unchanged.

## Correctness gate

The release candidate passed:

- Segment dynamic WRRF: 29 tests;
- Segment channel-global score merge: 8 tests;
- Shard exact merge: 4 tests;
- Collection exact RRF: 5 tests;
- two-Shard HTTP seed, filter, explicit-consistency fallback, overwrite and
  delete;
- restart persistence and post-restart filter/fallback;
- ordered Top-20 mismatches: 0 in every HTTP case;
- `exhaustiveFallback=false` for the local Production path.

Batch-specific tests cover atomic envelope validation, explicit EOF, sticky
producer errors and partition invariance for batch sizes
`1/2/7/16/20/32/64/127`.

## Release smoke latency

The smoke used one persisted 512-document, two-Shard collection, 16-dimensional
Dense vectors, Sparse impacts and WRRF Top-20. Each binary received 30 warmups
and 300 measured requests. This small integration workload is a release
regression gate, not a replacement for the prior multi-dataset research sweep.

| Plan | p50 | p95 | p99 | Mismatch |
|---|---:|---:|---:|---:|
| V1.1.5 point fusion | 2.499 ms | 3.476 ms | 4.755 ms | 0 |
| Fixed-16 batch fusion | 1.124 ms | 1.346 ms | 1.782 ms | 0 |

Relative to V1.1.5, the candidate reduced p50 by 55.03% and p95 by 61.28%.
Both plans consumed the same logical channel ranks in the measured request:
Dense 496 and Sparse 133. The gain therefore comes from carrying the existing
physical batches through channel merge and WRRF instead of repeatedly
decomposing them into point-at-a-time calls.

## Local artifacts

The temporary artifacts were verified before cleanup:

| Artifact | SHA-256 |
|---|---|
| seed HTTP smoke | `9cc51babc2c5e6437de07878fac233e018b19c2715d66e3fc9b2c3e12ae5ba90` |
| restart HTTP smoke | `5c9530b6250fd89ab28e57f1c52410c1a2c78111f283a78baf7323e7bb1aca3f` |
| candidate benchmark | `97de85c72f35af19ff652a6a875e0f38259cc53cc175faa1aa09c5feb62160e9` |
| V1.1.5 benchmark | `f73296aee37b5e179df66bcbc7e28b5a116edc0f94c3c588840f1b081d83f6d4` |
