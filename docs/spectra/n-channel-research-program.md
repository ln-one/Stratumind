# Spectra N-channel certified retrieval research program

Status: active research plan. This document separates hypotheses from established
results and forbids a universal-superiority claim without an explicit workload
envelope.

## 1. Research question

Given `N` rank-producing retrieval channels over one frozen identity universe,
can one execution kernel return exactly the same ordered WRRF Top-K as exhaustive
per-channel evaluation while reducing end-to-end latency and physical work by:

1. exposing each channel as a resumable exact ordered stream;
2. attaching an admissible upper bound to unopened work;
3. stopping fusion as soon as the final order is certified;
4. sharing identity, generation, filter, layout, materialization, and cache state;
5. selecting a native fallback when certificates or cross-channel agreement are
   unlikely to pay for their overhead?

The target is a statistically credible Pareto advantage over selected strong
baselines in a declared workload envelope. “Better than every method for every
query and corpus” is neither assumed nor a scientifically defensible target.

## 2. Stable abstraction

The physical catalog stores each Node and identity once. A channel contributes
only two operations:

- `Value(channel, query, Node)`: an admissible upper bound on every result
  reachable through that Node;
- `Score(channel, query, identity)`: the exact channel score when the identity is
  materialized.

The executor does not require Dense, Sparse, graph, or text semantics. It requires
only finite exact scores, an admissible Node Value, deterministic identity ties,
and a resumable stream. Different channels may use unrelated score scales because
WRRF consumes rank rather than raw score.

Shared physical state does not merge channel values or queues. It removes repeated
work below their logical boundary:

```text
SharedNodeCatalog
  generation + identity + live bits + filter + Node ranges + materialization/cache
      |                |                  |
  channel 0 Value  channel 1 Value  ... channel N-1 Value
      |                |                  |
  exact stream 0   exact stream 1   ... exact stream N-1
                  Dynamic Exact WRRF
```

This distinction is essential. A Node opened by four channels may require four
different score computations, but its identity page, liveness/filter mask, and
physical materialization should be paid once.

## 3. Correctness contracts

### Exact track

- Every channel stream equals full-corpus channel sorting under score descending,
  frozen identity ascending.
- Dynamic WRRF output equals exhaustive full-corpus WRRF ordered Top-K.
- Ordered mismatch count must be zero. Full scan is a valid cost result; a wrong
  identity or order is not.
- Timeout, cancellation, generation change, invalid score, or bound violation
  must fail closed rather than return an exact claim.

### Production track

- Compare at matched Recall@K/NDCG@K, not at unequal quality.
- Approximate Dense or Sparse proposal may be used only with an explicit guarantee
  scope, or followed by a complete certified fallback.
- A conservative cost router may choose the native Qdrant executor. Router regret
  against the per-query oracle is a primary metric.

## 4. Prior-art and baseline matrix

| Area | Required strong baseline | What it already solves | Remaining Spectra question |
|---|---|---|---|
| Sorted top-k aggregation | Fagin/Lotem/Naor TA and NRA | Instance-optimal access under their access model | Can physical channel access be made cheaper and cancellable under exact WRRF? |
| Sparse exact pruning | MaxScore, WAND, BMW, VBMW, Lucene BMW | Safe skipping with global/block term maxima | Can Node Value augment Qdrant postings and share physical state across channels? |
| Sparse approximate | Seismic and SeismicWave | Learned-sparse block summaries, pruning, graph expansion | Does exact mode remain competitive anywhere; can production mode compose their proposal with exact fusion? |
| Dense approximate | Qdrant HNSW exact-off, ScaNN/AVQ, DiskANN | Strong latency/recall trade-offs at scale | Can a proposal/certificate/fallback cascade reach equal quality with less work? |
| Dense certified | Certified Cosine | Exact high-dimensional NN when a graph certificate succeeds | Does top-k certification scale on modern retrieval embeddings and Qdrant layouts? |
| Quantized bounds | Qdrant Scalar/PQ/Binary/TurboQuant; RaBitQ | Cheap approximate scores; RaBitQ supplies theoretical error bounds | Can score intervals safely avoid most full-precision rescoring? |
| Hybrid fusion | Qdrant fixed prefetch + weighted RRF/DBSF | Mature multi-stage hybrid retrieval | Can fusion stop pull work instead of consuming fixed lists? |
| Unified hybrid index | Allan-Poe | One GPU graph for dense, sparse, text, and KG paths | Can Spectra provide exact ordered equivalence, CPU/Qdrant integration, and arbitrary certified channels? |

Primary references:

- Fagin, Lotem, Naor, *Optimal Aggregation Algorithms for Middleware*:
  <https://arxiv.org/abs/cs/0204046>
- Broder et al., *Efficient Query Evaluation Using a Two-Level Retrieval
  Process*: <https://research.google/pubs/efficient-query-evaluation-using-a-two-level-retrieval-process/>
- Mallia et al., *Faster BlockMax WAND with Variable-sized Blocks*:
  <https://doi.org/10.1145/3077136.3080780>
- Grand et al., *From MaxScore to Block-Max WAND*:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7148045/>
- Francis-Landau and Van Durme, *Exact and/or Fast Nearest Neighbors*:
  <https://arxiv.org/abs/1910.02478>
- Malkov and Yashunin, *Efficient and Robust Approximate Nearest Neighbor
  Search Using HNSW*: <https://arxiv.org/abs/1603.09320>
- Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization*: <https://research.google/pubs/accelerating-large-scale-inference-with-anisotropic-vector-quantization/>
- Gao and Long, *RaBitQ*: <https://arxiv.org/abs/2405.12497>
- Bruch et al., *Efficient Inverted Indexes for Approximate Retrieval over
  Learned Sparse Representations*: <https://arxiv.org/abs/2404.18812>
- Bruch et al., *SeismicWave*: <https://arxiv.org/abs/2408.04443>
- Li et al., *All-in-one Graph-based Indexing for Hybrid Search on GPUs*:
  <https://arxiv.org/abs/2511.00855>
- Qdrant hybrid query and WRRF contract:
  <https://qdrant.tech/documentation/search/hybrid-queries/>

## 5. Implementation work packages

### A. Shared N-channel kernel

- one `SharedNodeCatalog`;
- arbitrary channel count;
- independent admissible bounds and scorers;
- shared Node/identity materialization telemetry;
- Dynamic Exact WRRF and cross-channel cancellation;
- property tests for 1–8 channels, partial overlap, ties, zero weights, and
  adversarial disagreement.

### B. Sparse channel

1. Keep Qdrant native inverted search as the baseline and fallback.
2. Add block-local maxima to posting storage/iterators instead of scanning a
   parallel document-block index.
3. Compare fixed BMW, VBMW, MaxScore/WAND-compatible scheduling, and native
   batching on BM25 and learned sparse vectors.
4. Route from observable query features: query nnz, posting lengths, impact
   skew, filter selectivity, and current threshold growth.

### C. Dense channel

1. Keep Qdrant exact scan and HNSW ef matrix as baselines.
2. Prototype a quantized score interval: approximate score plus a rigorous
   residual/rounding bound; rescore only candidates whose intervals overlap the
   current threshold.
3. Test HNSW proposal followed by Certified Cosine or interval certification;
   fall back to quantized/full exact scan when proof fails.
4. Compare Scalar, Binary, TurboQuant, RaBitQ-like bounds, clustered balls, and
   graph certificates. Index size and build time are first-class metrics.

### D. Fusion and router

- push Dynamic WRRF stopping into resumable physical streams;
- cancel unfinished channel work immediately after certification;
- predict executor cost conservatively;
- abstain to native Qdrant when confidence is low;
- report router regret and worst-case tail latency, not only average speedup.

## 6. Dataset matrix

### Synthetic

- sizes: 100K, 1M, 5M identities;
- channels: 1, 2, 4, 8;
- correlation: identical, correlated, independent, anti-correlated;
- score shape: Zipf, clustered, uniform, outliers, flat ties;
- Sparse: query nnz, posting length, learned-sparse density;
- Dense: 32/128/384/768 dimensions, cluster separation, query entropy;
- filters: none, 50%, 10%, 1%, correlated and anti-correlated with relevance.

### Real

- MIRACL English and Chinese for multilingual million-scale retrieval;
- Natural Questions for document-shaped retrieval under CC BY-SA 3.0;
- SciFact remains only a correctness and integration smoke corpus;
- a learned-sparse MS MARCO-compatible benchmark is required for Seismic-class
  comparison, subject to dataset/model licensing.

Every snapshot pins corpus version, license note, encoder revision, vocabulary,
identity map, checksums, compiler, Qdrant commit, and machine profile.

## 7. Metrics and decision rules

Report per configuration:

- ordered Top-K mismatches;
- Recall@K and NDCG@K for approximate tracks;
- p50/p95/p99 and throughput after warmup;
- CPU time, instructions/cycles when available, bytes/pages read, cache reuse;
- postings, Nodes, identities, and exact vector scores consumed;
- index bytes, build time, update amplification;
- 95% confidence intervals over repeated runs;
- router selection, oracle-best executor, and regret.

A configuration is Pareto-superior only if it is no worse in the declared quality
contract and improves at least one resource/latency dimension without worsening
the others beyond the predeclared tolerance. Workload regions where it loses must
remain in the paper and become router fallback rules.

## 8. Current evidence

- Dynamic Exact WRRF and existing two-channel streams have zero ordered mismatch
  in property, synthetic, adversarial, and SciFact runs.
- Standalone Sparse document Block Max loses badly to Qdrant native inverted
  search on SciFact; the bound must move into postings.
- Posting-level Block Max is now integrated into Qdrant's compressed posting
  chunks. Each 128-entry chunk stores one maximum impact. For a contiguous
  document-id batch, query-weighted per-posting maxima form an admissible score
  bound; a batch is skipped only when even its best score-and-identity cannot
  enter the current Top-K. The optimized and pruning-disabled paths share the
  same compressed index and SearchContext. All 127 sparse library tests pass,
  including f32/f16/u8/quantized-u8 RAM and mmap variants.
- On synthetic non-negative Sparse Top-20, the posting implementation had zero
  ordered mismatch at 100K, 1M, and 5M documents. A 4,096 document-id batch was
  the measured knee: 256 made bounds tighter but loop overhead dominated. At 1M
  documents with descending impacts, an endpoint-trend-routed executor skipped
  43.77% of posting elements and reduced p50 from 1.354 ms to 1.033 ms. At 5M it
  skipped 45.24% and reduced p50 from 6.867 ms to 5.392 ms. Query-weighted first
  and last block maxima provide an O(query terms) physical cost hint: ascending
  and flat profiles route back to the native 10,000-id batch path before search.
  Exactness never depends on the hint. The f32 index adds one four-byte maximum
  per full 128-posting chunk: about 93.2 KB (0.67%) at 1M in this workload.
- This Sparse result is not yet a stock-Qdrant or real-corpus win. The disabled
  baseline shares the modified chunk layout, the on-disk format still needs an
  explicit version migration, and BEIR/MIRACL plus a clean upstream binary must
  be measured. Flat/ascending routed runs remain within roughly one percent of
  baseline rather than demonstrating a statistically established improvement.
- Dense balls degenerate on 384-dimensional SciFact; they are not a production
  Dense answer.
- A second Dense bound gate used 57,600 FiQA documents, 20 queries, 384
  dimensions, 16-document leaves, and a hierarchical balanced-k-means tree. The
  original Euclidean-plus-spherical-cap executor evaluated 1,151,590 of
  1,152,000 possible document/query pairs. Adding a coordinate-wise bounding box
  was admissible and preserved zero ordered mismatch, but consumed exactly the
  same 1,151,590 pairs while increasing p50 from 26.09 ms to 28.96 ms. The box
  bound was therefore removed from the default code. This closes the
  query-independent ball/box parameter-sweep branch: future Dense work must test
  a proposal that raises the certified threshold, a materially different
  certificate, or a native fallback. Raw data are in
  `docs/spectra/results/fiqa-dense-bound-gate-v1.local.json`.
- HNSW proposal did not rescue the same certificate family on SciFact. With
  `ef=256`, Qdrant HNSW had perfect Recall@20 over the 100-query gate, but a
  batched exact stream seeded with those 20 identities still expanded all 128
  leaves and scored all 518,300 document/query pairs. The combined p50 was 1.75
  ms versus 0.181 ms for Qdrant exact scan. This isolates the failure: the
  proposal supplied the right lower bound, while the 384-dimensional
  Euclidean/spherical-cap Node bounds remained too loose to certify it. The
  next Dense work package must avoid a full per-query quantized scan using a
  tighter interval/block/graph certificate; it must not keep tuning this Ball
  family. Raw data are in
  `docs/spectra/results/scifact-dense-proposal-ball-gate-v1.local.json`.
- A per-vector signed-int8 residual certificate matched exhaustive Dense Top-20
  for all 1,109 SciFact queries. It scores all 5,747,947 document/query pairs in
  int8 but exactly rescores only 130,911 (2.28%). Flattening document codes into
  one contiguous array, using a safe i32 dot-product fast path, and heapifying all
  initial intervals in one pass reduced p50 from 205.8 us to 98.2--99.8 us across
  five repeated runs. In three matched Top-20 runs, Qdrant exact scan measured
  159.1--162.1 us p50 and HNSW `ef=256` measured 132.6--141.5 us p50 with three
  ordered-list mismatches. The certificate had zero mismatches in every run.
  This is a promising exact single-channel kernel result, not yet an end-to-end
  system claim: the prototype uses a compact standalone array while the Qdrant
  baselines cross the Segment index interface. Its 2.16 MB encoded certificate is
  auxiliary to, not a replacement for, the 7.96 MB original vectors required by
  exact fallback. Native storage integration and larger datasets remain required.
- The larger FiQA gate removes the “small SciFact only” limitation. Across five
  counterbalanced 100-query runs over 57,600 vectors, the compact certificate had
  zero ordered mismatch, 0.853 ms median-run p50, and 1.011 ms p95. Qdrant exact
  scan also had zero mismatch but measured 1.970/2.193 ms. In 500 paired
  observations the compact executor won all 500; mean delta was -1.132 ms with a
  query-cluster bootstrap 95% interval of [-1.145, -1.118] ms. Only 0.266% of
  vectors required full-precision rescoring. HNSW `ef=256` was faster at 0.659 ms
  p50 but had 50 ordered-list mismatches across the five runs and mean Recall@20
  0.995, so it remains in the approximate track. Raw runs are in
  `docs/spectra/results/fiqa-dense-exact-repeats-v1.local.json`.
- A handwritten per-vector ARM NEON kernel preserved exactness but regressed the
  same full SciFact p50 from 205.8 us to 329.3 us. It was removed. The likely
  useful integration point is Qdrant's batched quantized scorer and storage
  layout, not thousands of tiny per-document SIMD calls.
- The certificate was then attached to Qdrant's native Scalar quantized scorer
  and compared in the same real Segment against exact scan and HNSW. Across all
  1,109 SciFact queries, certified Top-20 had zero ordered mismatches, while HNSW
  `ef=256` mismatched 3 ordered lists (mean Recall@20 0.999865). Qdrant's global
  Scalar quantizer produced looser intervals than the per-vector prototype:
  14.88% exact rescoring and 311.7--323.3 us p50 in the matched reruns, versus
  159.1--162.1 us exact scan and 132.6--141.5 us HNSW. Native integration is
  therefore correct but not yet competitive; it is a
  baseline and a precise argument for per-vector/RaBitQ/TurboQuant bounds.
- The first isolated shared-catalog matrix has zero mismatch for 1/2/4/8 channels.
  Correlated and flat-tie cases reuse physical Node materialization almost exactly
  `N` times. Independent and anti-correlated cases approach full stream
  consumption and can be slower than exhaustive sorting. This establishes both
  the value of sharing and the necessity of the cost router.
- A real mixed-channel control now shares one per-vector Dense certificate index
  and one Sparse block index across 1--8 query-rewrite streams. Property tests for
  every channel count match exhaustive WRRF exactly. On all 1,109 SciFact queries,
  N=2/4/8 with a 64-pull warmup certification interval had zero ordered mismatch
  and consumed 19.07%/23.08%/26.49% of full logical rank entries. Median latency
  was 1.569/3.060/6.474 ms versus 2.967/5.192/9.774 ms for fully exhausting the
  same streams. However p99 was worse: 4.103/6.848/14.282 ms versus
  3.595/6.374/11.739 ms. Mid-query drain-to-exhaustion budgets made N=4/8 worse
  because they paid both dynamic-prefix and full-completion costs; this fallback
  is retained only as a negative ablation.
- Logical stream pulls are not physical work. A unique Dense query computes all
  signed-int8 approximate scores, while exact Dense rescoring is sharply reduced.
  Bitwise-identical, equal-WRRF-weight Dense queries are now deduplicated before
  ranking construction and replay one exact physical stream; non-identical or
  differently weighted queries remain independent. This distinction remains
  explicit in every result table.
- The five-run all-unique Dense boundary confirms that this is also the fastest
  observed training-free layout: independent compact streams had 1.477 ms p50
  versus 1.622 ms for one document-major pass, with mean paired delta -0.159 ms
  and query-cluster bootstrap 95% interval [-0.183, -0.134] ms. Both performed
  identical dot products and exactly matched exhaustive WRRF. The accepted rule
  is duplicate replay followed by independent construction per unique query;
  document-major execution is retained only as an ablation.
- NFCorpus adds a third real retrieval domain and a legal empty-Sparse-query
  boundary. Across all 323 queries, Posting Block produced zero exhaustive WRRF
  ordered mismatches for N=1/2/4/8. Dynamic p50 was
  0.075/0.561/1.102/2.591 ms versus exhaustive
  1.412/1.714/2.935/5.445 ms. At N=8, dynamic execution consumed 23.83% of full
  logical rank entries. Fifteen queries whose remapped Sparse vector was empty
  completed without synthetic terms or special fusion rules.
- Five counterbalanced NFCorpus runs compared Posting Block, Shared Lazy, and
  qrels-blind Auto Shared Lazy at N=4/8. All 9,690 measured query executions had
  zero ordered mismatch. At N=4, Shared Lazy versus Posting Block had a mean
  paired delta of -1.1 us with a query-cluster bootstrap 95% interval of
  [-8.5, 5.9] us, so no latency winner is established. Auto's routing overhead
  made it 13.9 us slower on average at N=4. At N=8, Auto selected sharing on
  1,540 of 1,615 executions, decoded 926,005 physical versus 2,791,375 logical
  posting elements, and had a mean paired delta of -19.3 us with a 95% interval
  of [-32.0, -6.8] us. The primary conclusion is the exact physical-work
  reduction; the small latency effect is dataset- and scale-specific.
- The deterministic TREC-COVID 100K-document scale gate ran all 50 queries with
  Posting Block at N=1/2/4/8. Every ordered Top-K matched exhaustive WRRF. At
  N=8, Dynamic Exact WRRF consumed 4.77% of the exhaustive logical rank entries
  and had 35.14 ms p50 latency versus 323.59 ms for fully materializing and
  sorting the same eight streams. N=1/2/4 p50 was 1.41/5.41/11.27 ms versus
  71.98/91.28/166.78 ms exhaustive. These are same-kernel exact-execution
  comparisons, not yet a claim against Qdrant's independently optimized native
  retrieval APIs. The run also preserved two negative qrels judgments as signed
  values and treated every non-positive grade as zero gain.
- A strict Qdrant RAM-inverted Top-20 gate exposed one stable-order boundary.
  Native max-next pruning returned the same identity set on 49/50 TREC-COVID
  queries but could discard the smaller identity at an exactly tied cutoff; 36
  other queries differed only in f32 accumulation bits. Strict mode therefore
  disables that pruning certificate while retaining Qdrant's native posting
  scorer. With pruning disabled, all 50 scored Top-20 lists were bitwise equal
  to plain exhaustive search, p50 was 0.385 ms, build time was 71.3 ms, and the
  RAM posting index occupied 86.7 MB. The enabled-pruning path visited only 70
  fewer of 4,940,772 posting elements and was slower in this gate.
- The snapshot HTTP adapter now calls `NChannelExactIndex` directly instead of
  the superseded two-channel `HybridExactIndex`. Its wire request carries one to
  64 typed `channels[]`, finite non-negative weights, and one
  `rerankCandidateK`; empty Sparse streams are valid. The SciFact v2 smoke
  returned per-channel pull/exhaustion state, physical-stream telemetry, the
  full-corpus ordered-exact declaration, and rejected a stale generation. This
  closes the runnable kernel-to-Spectra identity boundary. The HTTP adapter now
  also holds an armed drop guard around its blocking worker. Handler abandonment
  sets the kernel flag, rank streams stop before their next pull, Adaptive Native
  receives the same cooperative flag, and a final kernel check forbids partial
  success. HTTP/1 half-close is disabled so connection EOF drops the handler.
  A real TCP regression starts the blocking worker, disconnects its client, and
  verifies both flag propagation and worker exit.
- The compressed Posting Block certificate is now a resumable exact score-ordered
  stream and is integrated into the same 1--8 channel executor. A second exact
  stream geometrically grows Qdrant native Top-K prefixes. Property tests run the
  document-block, posting-block, and adaptive-native strategies for every channel
  count from one through eight; all match exhaustive WRRF ordering.
- In a sequential full SciFact strategy matrix (1,109 queries, Top-20, 64-pull
  certification warmup), all three strategies had zero ordered mismatch and
  identical logical source pulls. Replacing document blocks with compressed
  Posting Block reduced dynamic p50 from 1.558/3.029/6.429 ms to
  0.862/1.857/4.154 ms at N=2/4/8, reductions of 44.7%/38.7%/35.4% in this local
  run. Relative to fully exhausting the same streams, Posting Block p50 was
  3.46x/2.79x/2.36x faster. It visited 4.14M/6.29M/12.43M Sparse posting elements,
  while document blocks evaluated 5.74M/11.43M/22.83M full documents. N=8 p99
  remained 11.516 ms versus 11.480 ms exhaustive, so the tail claim is not yet a
  win.
- The first Adaptive Native profile started at Top-20 and was rejected: at
  TREC-COVID N=8 it repeated 1,599 full posting traversals and reached 45.48 ms
  p50 versus 35.14 ms for Posting Block. A declared initial-limit sweep froze
  `4096` before cross-domain validation; `8192` was slower. Six fully
  counterbalanced TREC-COVID repetitions then found Adaptive(4096) faster than
  Posting Block by 1.10/2.22/9.22 ms mean at N=2/4/8, with query-cluster
  bootstrap 95% intervals [-1.70,-0.64]/[-3.21,-1.24]/[-12.82,-6.24] ms and
  zero ordered mismatch. The same frozen profile remained exact and improved
  N=4/8 on NFCorpus by 0.011/0.050 ms and SciFact by 0.128/0.339 ms; N=2 was a
  small win on SciFact and statistically tied on NFCorpus. Adaptive Native with
  initial limit 4096 is therefore the current default exact Sparse executor.
  Posting Block and Shared Lazy remain replaceable fallbacks and ablations.
  `4096` is a V3 Profile value, not a theorem or dataset-fitted router.
- The raw three-strategy matrix is
  `docs/spectra/results/scifact-n-channel-sparse-strategy-matrix-v1.local.json`.
  SciFact is still a small integration corpus. The result does not establish
  million-scale, multilingual, filtered, clean-upstream, or statistical
  superiority; those gates remain open.
- A controlled Sparse-only N-channel matrix now exposes the full performance
  phase change. At 100K identities, correlated and flat-tie channels consumed
  only 0.024--0.032% of full rank entries, while independent channels consumed
  82--97% and anti-correlated channels consumed 100%. N=8 correlated dynamic p50
  was 0.194 ms versus 57.826 ms exhaustive; independent dynamic p50 was 1.029 s
  versus 110.202 ms, and anti-correlated was 579.289 ms versus 58.312 ms. Every
  case retained zero ordered mismatch. Raw data is
  `docs/spectra/results/n-channel-synthetic-100k-matrix-v1.local.json`.
- A correctness-neutral prefix router now probes native exact Top-64 prefixes and
  selects either Dynamic Exact WRRF or exhaustive execution. The first average
  pairwise-overlap rule failed on N>=4 anti-correlation because internally
  coherent opposing groups inflated the mean. The retained conservative rule
  uses minimum pairwise overlap. It chooses dynamic for fully correlated/flat
  prefixes and exhaustive for independent/anti-correlated prefixes in the
  current controlled matrix. A wrong choice changes cost only; both executors
  retain the same exact result contract.
- At 1M Sparse identities, the router preserved the two extreme regions. On three
  correlated queries, routed N=2/4/8 latency was 0.536/0.944/1.901 ms versus
  326.5/457.1/761.2 ms exhaustive. On the single intentionally expensive
  anti-correlated boundary query, unrouted dynamic took 16.7/44.6/173.3 seconds,
  exhaustive took 0.374/0.561/0.941 seconds, and routed execution took
  0.340/0.503/0.841 seconds. These are exploratory boundary observations, not
  repeated publication estimates. They establish why portfolio routing is part
  of the method rather than an optional engineering add-on. Raw data is
  `docs/spectra/results/n-channel-synthetic-1m-router-boundaries-v1.local.json`.
- The minimum-overlap rule does not generalize as a complete real-corpus router.
  On 300 SciFact queries it sent 129/234/292 queries to exhaustive execution at
  N=2/4/8, even though Dynamic Posting Block was the measured per-query latency
  oracle for 294/295/288 queries. The resulting simulated routed p50 was
  1.500/5.384/11.094 ms versus 0.838/1.885/4.246 ms for always-dynamic. An
  even/odd held-out threshold sweep selected threshold zero for both average and
  minimum overlap: probing and then always choosing dynamic added 16.7--17.9%
  total cost. Overlap remains useful for extreme conflict detection, but the
  production router must model both expected pull depth and executor unit cost,
  and must abstain from probing where its expected value is negative. Per-query
  observations and held-out analysis are in
  `docs/spectra/results/scifact-router-observations-300-v1.local.json` and
  `docs/spectra/results/scifact-router-threshold-heldout-v1.local.json`.
- FiQA now supplies a second real-corpus exact control: 57,600 non-empty
  documents and 647 judged test queries after the snapshot manifest explicitly
  removed 38 empty documents and the one affected query. Dynamic Posting Block
  matched exhaustive WRRF ordered Top-20 for every query at N=2/4/8. It consumed
  7.57%/5.21%/7.10% of full logical rank entries. In the quality-enabled rerun,
  p50 was 3.77/6.91/20.96 ms versus 45.31/84.89/161.56 ms exhaustive;
  Recall@20 was 0.526/0.513/0.513 and NDCG@20 was 0.407/0.397/0.411. Exact and
  exhaustive quality are identical because their identities and order are
  identical. The snapshot is research-only pending a resolved license path, and
  these single-machine latencies are not yet publication estimates. Raw summary:
  `docs/spectra/results/fiqa-n-channel-exact-quality-v2.local.json`.
- A static eager-Shared Sparse router based only on union/logical posting volume
  is disproved. On 100K correlated rankings with eight channels sharing one term
  (`U/L=0.125`), Posting Block certified Top-20 at 0.227 ms p50 while eager Shared
  Full required 7.360 ms. The same rule failed throughout `U/L=0.125--0.75`.
  Posting overlap cannot predict how little of each exact stream Dynamic WRRF
  will consume.
- Dynamic WRRF is now a recoverable `DynamicRrfSession`. It can pause at a
  declared pull boundary, force a certificate check, replace a stream only after
  replaying and matching every observed identity rank, resume, or be cancelled by
  dropping the session. Property and mixed-channel tests reject prefix mismatch
  and preserve exhaustive ordered Top-K. Probe-then-eager-Shared protects the
  early-stop counterexample but lost in three interleaved FiQA N=8 repetitions
  because it repeats posting work, so it remains a negative ablation rather than
  the default.
- The new Shared Lazy Posting coordinator combines admissible per-channel batch
  heaps with one physical union-term materialization cache. Opening a batch in
  any channel computes exact batch scores for all Sparse channels; score scales,
  bounds, and rank streams remain independent. On the 100K maximum-overlap
  boundary it reduced physical decode exactly 8x (655,360 logical to 81,920
  physical elements across 20 queries) while p50 remained 0.232 ms and exactness
  remained zero-mismatch. With zero term overlap, physical equaled logical and
  p50 was 0.323 ms versus roughly 0.308 ms independent Posting Block, exposing a
  small coordination cost rather than catastrophic degradation. On 100 FiQA
  queries, N=4/N=8 physical decode fell from 3.46M/6.68M logical elements to
  2.23M physical elements, with preliminary p50 6.60/21.66 ms. The frozen
  `AutoSharedLazy` rule is training-free and qrels-blind: choose Shared Lazy iff
  query/index metadata proves duplicate posting work (`U<L`), otherwise choose
  independent Posting Block. Full evidence and limits are recorded in
  `docs/spectra/results/shared-sparse-executor-study-v1.local.json`.

- Exact rank replay now handles the strongest overlap boundary without forcing
  every channel through a shared scorer. Bitwise-identical Sparse queries with
  bitwise-identical WRRF weights retain N logical streams, but one physical
  producer feeds all readers. The frozen `AutoSharedLazy` cost rule compares
  union posting volume `U` against deduplicated independent posting volume `D`
  and selects Shared Lazy iff `U<D`; otherwise it selects Posting Block plus
  replay. This uses only the current query and frozen posting lengths. On five
  counterbalanced 100K-document, 100-query N=8 repetitions, the fully repeated
  boundary reduced physical/logical rank pulls to 0.125 and achieved median-run
  p50 0.129 ms, versus the earlier Shared Lazy p50 0.232 ms. The independent
  boundary performed no replay and remained near Posting Block (0.296 versus
  0.292 ms). Every run had zero exhaustive ordered Top-K mismatch. Raw runs are
  in `docs/spectra/results/synthetic-rank-sharing-repeats-v1.local.json`.
- The same generic replay layer now handles Dense duplicates before their
  all-document int8 pass. In five counterbalanced 100K-document N=8 mixed runs,
  four identical Dense channels reduced physical/logical Dense dots from 1.0 to
  0.25 and physical/logical rank pulls from 1.0 to 0.625. Median-run p50 fell
  from 1.632 ms to 0.477 ms. Across 500 paired observations, mean delta was
  -1.191 ms with query-cluster bootstrap 95% interval [-1.219, -1.164] ms; both
  strategies matched exhaustive WRRF exactly. The training-free rule is exact
  bitwise Query equality plus exact WRRF-weight equality. Raw runs are in
  `docs/spectra/results/synthetic-dense-replay-repeats-v1.local.json`.

## 9. Claim ladder

The final paper must use the strongest claim actually supported:

1. **Correctness:** exact ordered equivalence for arbitrary `N` admissible streams.
2. **Mechanism:** physical sharing reduces duplicate materialization without
   coupling channel scores.
3. **Conditional performance:** faster inside measured high-agreement/tight-bound
   regions.
4. **Portfolio performance:** router is competitive with the best included
   executor across the declared workload envelope.
5. **Broad superiority:** allowed only if every selected strong baseline is beaten
   at matched quality with statistically credible results. It is not presumed.
