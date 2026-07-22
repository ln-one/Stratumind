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

### Stratumind V0 scheduler matrix

Build the native synthetic executor once, then run the counterbalanced two-channel matrix:

```bash
cargo build --release -p segment --example spectra_n_channel_synthetic
python3 tools/spectra/run_stratumind_v0_executor_matrix.py \
  --binary target/release/examples/spectra_n_channel_synthetic \
  --output docs/spectra/results/generated/stratumind-v0-executor-matrix-100k-v3.local.json
```

The runner fixes one Dense and one Sparse-impact channel and compares `max-next`,
`competitor-cost`, and `safe-router` on correlated, independent, anti-correlated, and flat-tie
inputs. Every arm is rejected on any deterministic ordered Top-K mismatch. The primary comparison
is source pulls and native physical counters; latency is secondary. A reduction in logical pulls is
not described as a physical speedup when Dense still scores every quantized row or Sparse expands a
whole posting batch.

The current frozen 100K/20-query/five-repetition result is
`docs/spectra/results/generated/stratumind-v0-executor-matrix-100k-v3.local.json`. All 12 arms and
both the dynamic executor and prefix Router have zero ordered Top-K mismatches. Median dynamic pull
ratios for `max-next` are 0.00020/0.803676/1.0/0.00020 on
correlated/independent/anti-correlated/flat-tie input. `competitor-cost` is strictly worse in this
mixed Dense+Sparse profile (0.00022/0.9028465/1.0/0.00022), so it remains an all-incremental-stream
research option rather than the V0 default.

A separately labeled 1M/20-query/single-repetition scale check is stored in
`docs/spectra/results/generated/stratumind-v0-executor-matrix-1m-preliminary.local.json`. With
`max-next`, correlated and flat-tie inputs consumed 0.00002 of exhaustive source pulls, independent
inputs consumed 0.7996508, and adversarial anti-correlated inputs consumed 1.0. Every ordered Top-K
matched exhaustive fusion. This exposes both the favorable pruning region and the exact worst-case
degeneration; it is preliminary and is not merged into the five-repetition main table.

The Safe Router now distinguishes marginal pull cost from already-paid stream preparation. It
selects `max-next` for all mixed V0 queries and exactly matches its pull ratios. The independent
prefix Router selects Dynamic for every correlated and flat-tie query and Exhaustive for every
independent and anti-correlated query, again with zero mismatch. Its median p50 is about
0.66/42.3/34.3/14.8 ms respectively, including probe cost. Dense still performs all physical
quantized dot products in every arm; these numbers prove exact plan selection and consumption
behavior, not yet a native Dense physical speedup.

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

## Segment-owned V1 kernel gate

The first Segment-owned gate uses the regenerated 5,183-document SciFact snapshot, 100 frozen
queries, one warmup, five measured repetitions, release builds, within-query counterbalancing, and
10,000-sample query-cluster bootstrap intervals.

- Dense Auto portfolio versus Qdrant exact on SciFact: zero mismatches; 92,833 ns versus 180,583 ns
  median-run p50; median ratio 0.5137; 500/500 paired wins; mean delta 95% interval
  [-95,376.980, -89,273.032] ns. Auto selected Compact for all 500 queries.
- Dense 100K dimension gate: Auto/exact median ratios are 1.0172 at d64, 1.0158 at d192, 0.8893 at
  d256, and 0.7844 at d384. Auto uses exact prefix for d64/d192 and Scalar for d256/d384. At
  10K x d384 Auto uses Compact with ratio 0.8179. All 2,500 Auto comparisons have zero ordered
  mismatch. Production does not persist Compact above 16,384 points.
- Sparse Posting stream versus Qdrant SearchContext at Top-20: zero accepted-prefix mismatches;
  33,667 ns versus 6,167 ns median-run p50; median ratio 5.4769; zero paired wins. At the production
  64+1 prefix depth, SearchContext is 8,709 ns versus 36,000 ns for Posting; 490/500 prefixes had a
  strict accepted boundary and 10/500 correctly selected the Posting fallback.
- A clean upstream v1.18.2 (`44ad62f8c`) worktree with no library diff measured 183,958 ns Dense
  exact and 6,292 ns Sparse SearchContext median-run p50. This matches the fork's exact baselines
  after restoring upstream `TopK`; the earlier stable-TopK instrumentation regression is rejected.

The accepted exact Router therefore selects Compact Dense for small high-dimensional Segments,
exact-prefix Dense for low dimensions, Scalar Dense for large high-dimensional Segments, and one
Qdrant Sparse prefix before lazily opening Posting. These are kernel claims, not yet a clean
upstream end-to-end latency claim. Raw artifacts are under `docs/spectra/results/generated/`.

The signed-int8 manual NEON/AVX2 experiment is rejected: it regressed the Compact kernel and was
fully removed. Retained `*-v2.local.json` kernel artifacts document that negative result; no custom
SIMD implementation remains in production code.

## HTTP state and persistence gate

The final Docker image is checked with a two-Shard, 512-document independent baseline:

```bash
python3 tools/spectra/run_exact_rrf_http_smoke.py \
  --phase seed-and-verify \
  --output docs/spectra/results/generated/stratumind-exact-rrf-http-smoke-seed-v1.local.json
# Restart the same container against the same volume.
python3 tools/spectra/run_exact_rrf_http_smoke.py \
  --phase verify-existing \
  --output docs/spectra/results/generated/stratumind-exact-rrf-http-smoke-restart-v1.local.json
```

The seed phase has four zero-mismatch cases: base, payload filter, overwrite plus delete, and the
same mutation under filter. The restart phase has two zero-mismatch persistence cases. Every case
returns `native-local-dense-sparse-v1`, `native-exact-rank-streams`, and no exhaustive fallback.
This gate establishes functional end-to-end equivalence, not a latency claim.
