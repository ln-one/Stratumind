# Spectra N-channel experiment protocol

## Scope and frozen contracts

The reference-exact track must return the same ordered Top-K identities as fully
materializing every channel and applying the same weighted RRF implementation.
Ties use the frozen identity order. The production track may be approximate only
when compared at matched Recall@K and NDCG@K. No result may mix these contracts.

The implementation base is Qdrant v1.18.2 commit
`44ad62f8cd69642be5afa6441612525e24a0d063`. Every raw result records compiler,
build profile, machine, encoder, corpus checksum, channel count, seed, and policy.

## Datasets and split rules

- Synthetic: 100K, 1M, and 5M identities; seeds fixed before the main run. Vary
  1/2/4/8 channels, agreement, impact skew, ties, query nnz, filters, Dense
  dimension, and cluster separation. Generator names start with `synthetic_`.
- SciFact: full 5,183-document corpus and all 1,109 encoded queries for execution
  correctness; judged test queries only for Recall/NDCG. It is an integration
  corpus, not the scale claim.
- MIRACL English and Chinese: official corpus/dev split. Dataset queries are
  evaluation-only for the cost router. No FiQA, SciFact, NQ, MIRACL, or qrels
  signal may fit executor thresholds or train the final selector.
- Natural Questions: pinned public retrieval snapshot and official query split.
- Learned Sparse: MS MARCO-compatible benchmark with a pinned encoder and license
  note, used for WAND/BMW/Seismic-class comparisons.

## Fair baselines

1. Qdrant native Sparse fixed Top-K on the same compressed index.
2. Qdrant exact Dense scan and HNSW ef sweep on the same Segment.
3. Full materialization plus the same WRRF implementation.
4. Document Block Max, compressed Posting Block stream, and adaptive native
   prefix stream.
5. MaxScore/WAND/BMW/VBMW-compatible Sparse execution where available.
6. Traditional fixed per-channel candidate WRRF at matched quality and candidate
   budget.
7. Approximate Dense proposal/certificate/fallback cascades at matched Recall.

An optimized method and its baseline must share corpus, query order, filters,
thread count, build mode, storage residency, deterministic tie order, and warmup.
Execution order is counterbalanced. Clean upstream Qdrant runs are separate from
same-layout ablations.

## Metrics and aggregation

- Correctness: ordered Top-K mismatch count; exact runs require zero.
- Quality: Recall@K, NDCG@K, and MRR where the dataset defines relevance.
- Latency: p50/p95/p99, mean, throughput, and per-query paired deltas.
- Physical work: posting elements visited/skipped, blocks expanded, bounds,
  identities materialized, quantized scores, exact rescoring, and source pulls.
- Cost: CPU time, instructions/cycles and cache misses when available, peak RSS,
  index bytes, build time, and update amplification.
- Router: chosen executor, oracle-best executor, regret, abstention, and worst-case
  tail.

Main tables use at least five independent measured repetitions after warmup and
report mean plus 95% bootstrap confidence intervals for paired query deltas.
Latency percentiles are computed per run before cross-run aggregation. Synthetic
generation uses at least three frozen seeds. SciFact exploratory single-machine
runs remain labeled preliminary until repeated.

## Main comparisons

The primary exact comparison is Dynamic Exact WRRF with compressed Posting Block
streams versus full materialization and versus the best same-kernel exact
portfolio member. The primary production comparison is the routed portfolio
versus Qdrant native hybrid at matched quality. Broad superiority is allowed only
if the routed portfolio is Pareto-competitive across every declared workload
region; otherwise the paper states conditional regions and fallback rules.

## Efficiency and ablations

- Channel count: 1/2/4/8.
- Sparse stream: document block / posting block / adaptive native / native fixed.
- Posting batch: 128/256/512/1024/2048/4096/8192/16384.
- Adaptive initial limit and growth factor.
- Dynamic WRRF certification interval and scheduling policy.
- Shared catalog/cache on/off; physical identity dedupe on/off.
- Bitwise-identical equal-weight rank replay on/off; report logical versus
  physical streams, pulls, and buffered identities.
- Dense certificate: per-vector int8, native scalar, proposed tighter bound,
  HNSW proposal, and exact fallback.
- Router: oracle (diagnostic upper bound only), frozen training-free cost model,
  structural heuristic ablations, and always-native. The deployable router may
  use only the current Query, frozen index metadata, online execution telemetry,
  and hardware microbenchmarks gathered independently of relevance datasets.

## Generalization and robustness

Run correlated, independent, anti-correlated, flat-tie, outlier, ascending-impact,
and adversarial-filter workloads. Include cold/warm cache, RAM/mmap, f32/f16/u8/q8
postings, sparse and dense query rewrites, and update/rebuild scenarios. Failure
regions remain visible and become explicit router fallbacks.

Dataset-specific threshold sweeps are allowed only as negative diagnostics for
feature insufficiency. They cannot define the frozen production rule or support
the primary performance claim.
