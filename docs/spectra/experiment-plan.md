# Spectra certified retrieval experiment plan

The active N-channel extension, prior-art matrix, falsifiable claim ladder, and
full workload plan are maintained in `docs/spectra/n-channel-research-program.md`.
This file retains the implementation history of the first two-channel kernel.

This branch evaluates exact lazy multi-channel retrieval inside the Qdrant
execution kernel. It is based on Qdrant `v1.18.2` at commit
`44ad62f8cd69642be5afa6441612525e24a0d063` and remains subject to the upstream
Apache-2.0 license.

## Reproducible baseline

- Branch: `spectra-research-v1.18.2`
- Qdrant package version: `1.18.2`
- Verified compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Locked dependencies: repository `Cargo.lock`

Qdrant declares Rust 1.94 as its minimum version, but this release uses standard
library features that do not compile with Rust 1.94. Experiments therefore record
the exact compiler commit in addition to the Qdrant commit.

## Invariant under test

For each channel, the physical executor must expose an exact stream ordered by
channel score and then by frozen point identity. Dynamic RRF may inspect prefixes
of those streams, but it may stop only when no observed or unseen point can change
the first `K` identities or their order.

The reference result is produced by fully materializing every channel stream and
running Qdrant's existing `rrf_scoring`. The optimized result is correct exactly
when its ordered point identities equal that exhaustive result. Exact final RRF
numbers are not required for stopping; each returned rank may be fixed while its
unmaterialized score contribution remains bounded.

The identity order is part of the correctness contract, not presentation polish.
Upstream Sparse `TopK` compared score only, so equal-score selection inherited
arbitrary `select_nth_unstable` behavior. This branch gives Sparse top-k a total
order of score descending and frozen identity ascending. The current physical
prototype uses point offset as the identity ordinal; production integration must
provide an ordinal derived from stable Qdrant point identity rather than segment
insertion order.

## Test layers

### 1. Logical property tests

Generate complete and partially overlapping channel permutations with:

- 1 to 47 point identities;
- 1 to 4 channels;
- zero and non-zero channel weights;
- RRF `k` from 1 to 100;
- result limits from 1 to the generated universe size;
- tied channel order keys resolved by point identity.

For every generated case:

```text
dynamic ordered Top-K == exhaustive ordered Top-K
```

Deterministic adversarial tests also cover opposite channel orders, all-zero
weights, duplicate identities within one source, and early stopping on identical
sources.

### 2. Physical synthetic indexes

Sparse datasets generate non-negative impact vectors, controlled posting-list
lengths, term-frequency skew, and coordinate-wise Node Max bounds. Dense datasets
generate clustered unit vectors plus admissible center/radius bounds. Both retain
the exhaustive executor as the oracle.

### 3. Real corpora

Real benchmark adapters preserve document and section structure when available:

```text
Corpus -> Document -> Section -> Chunk
```

Retrieval quality metrics verify that query preparation remains useful. They are
not evidence for execution correctness: the optimized executor must return the
same WRRF result as exhaustive execution before relevance is measured.

## Same-kernel comparisons

- exhaustive Qdrant execution;
- Qdrant native execution;
- Block Max execution;
- Node Max execution;
- Node Max with Dynamic Exact RRF.

Each run records:

- ordered Top-K mismatch count;
- p50, p95, and p99 latency;
- throughput;
- postings, blocks, Nodes, and Chunks visited;
- bound evaluations and stopping reason;
- index size and peak memory;
- offline build time.

The primary correctness gate is zero ordered Top-K mismatches. A worst-case query
may legitimately scan the full corpus; it may not return a different result.

## Current implementation status

`DynamicRrfState` is implemented next to Qdrant's exhaustive RRF primitive in
`lib/segment/src/common/reciprocal_rank_fusion.rs`. It accepts one point at a time
from each exact stream, tracks the greatest possible unseen contribution, and
returns an ordered Top-K only when the order is fixed.

`SearchContext` now exposes a common Sparse telemetry snapshot: posting lists and
elements, evaluated and safely skipped elements, batch span, prune attempts and
successes, and cancellation state. Sparse `TopK` now preserves the total
score-and-identity order even when the K-th score is tied. The deterministic
`spectra_sparse_baseline` example generates physical Sparse indexes and compares
native indexed execution and Block Max execution with exhaustive plain search.

`BlockMaxIndex` groups documents into fixed physical blocks and stores one
coordinate-wise maximum impact envelope per block. For a non-negative Sparse
query, the envelope dot product is an admissible upper bound. `BlockMaxStream`
expands blocks in bound order and emits a point only when it must precede every
point in every unopened block. A 1,000-case property test checks the entire stream
against exhaustive score-and-identity order; deterministic tests cover zero-bound
blocks and ties across blocks.

`DenseBallIndex` is the first Dense admissible-bound source. Each physical block
stores a centroid and covering radius. For query `q`, Cauchy-Schwarz gives the
bound `q·center + ||q|| radius`. Normalized embeddings also receive a spherical
cap bound derived from cluster direction, maximum angle, and norm range; the
executor safely takes the smaller of the two bounds. A hierarchical build can
refine an internal Node into smaller balls without scoring documents. Property
tests cover flat, clustered, and hierarchical builds over mixed-sign coordinates,
ties, zero vectors, and degenerate identical-vector clusters.

A FiQA gate also tested a coordinate-wise bounding box at every Dense Node. It
was exact but did not eliminate a single additional document score at 384
dimensions and increased p50 by about 11%. The code was removed after the gate;
the negative result remains in
`docs/spectra/results/fiqa-dense-bound-gate-v1.local.json`. Further Dense work
must be isolated behind a proposal/certificate executor and earn its place
against the native fallback.

`execute_dynamic_rrf` accepts heterogeneous exact streams, schedules the source
with the greatest next possible RRF contribution, and certifies in safe batches.
Batching changes only when the proof is checked, never its condition. This avoids
the quadratic behavior of scanning all observed candidates after every pull while
retaining the same exact stopping rule.

The first local release-profile baseline used 50,000 documents, 200 queries and
Top-20 under descending, ascending and flat weight patterns. All 600 query-pattern
pairs had zero ordered-result mismatches. Native execution was much faster than
plain search, but skipped only 1,448 to 9,992 of 10,664,064 posting elements
(approximately 0.014% to 0.094%). This is an initial local observation, not a
publication result. It shows why latency and physical access counts must both be
reported: batching dominated the observed speedup while suffix-bound pruning
removed little work. Raw results are in
`docs/spectra/results/sparse-native-synthetic-v1.initial.json`.

The first Block Max run used the same 50,000-document, 200-query, Top-20 corpus
with 256-document blocks. Native, Block Max and exhaustive execution had zero
ordered Top-K mismatches for descending, ascending and flat weight patterns.
Block Max evaluated approximately 6.20 million documents for the non-flat runs
and 7.75 million for the flat run, versus 10 million for exhaustive execution.
Its p50 latency was 194 to 234 microseconds, around 10 to 12 times faster than
plain exhaustive search but around 2.9 to 3.6 times slower than Qdrant's mature
native inverted path on this synthetic corpus. This supports combining the
admissible bound with the inverted execution structure instead of treating a
parallel Block Max scanner as the final Sparse engine. Raw results are in
`docs/spectra/results/sparse-block-max-synthetic-v1.local.json`.

The first end-to-end hybrid run used 20,000 aligned Sparse and 32-dimensional
Dense documents, 100 queries, Top-20, and 128-document blocks. Dynamic exact WRRF
matched exhaustive hybrid WRRF for all 100 queries and fixed every result before
either source was exhausted. It evaluated 1,062,304 Sparse and 1,025,248 Dense
documents, versus 2,000,000 of each under exhaustive execution. After batched
certification, p50 latency was 1.46 ms versus 5.46 ms exhaustive (approximately
3.7 times faster in this local synthetic run). Raw results are in
`docs/spectra/results/hybrid-dynamic-exact-v1.local.json`.

The adversarial matrix then varied physical layout, cross-channel agreement, and
tie density while keeping the corpus size fixed. All 400 query-scenario pairs had
zero ordered Top-K mismatches and stopped before source exhaustion. Clustered and
interleaved p50 latency was 1.05 to 1.09 ms versus 5.86 to 6.03 ms exhaustive.
The anti-correlated case required much deeper prefixes: p50 remained faster at
3.36 ms versus 5.80 ms, but dynamic p99 rose to 11.73 ms versus 7.59 ms. This is a
real tail-latency risk, not hidden by the mean. The flat-tie case expanded one
block per channel after the zero-radius Dense bound was made exact. Raw results
are in `docs/spectra/results/hybrid-adversarial-matrix-v1.local.json`.

The query-path audit also found that root fusion currently happens at collection
level after every shard has materialized fixed-limit prefetches. Wiring only
`local_shard::fusion_rescore` would therefore be incorrect for global ranks and
would not create a genuinely lazy executor. The first implementation target is a
single-shard streaming executor; multi-shard exact stopping requires a collection
coordinator that merges shard-level channel bounds.

The first real snapshot uses the full BEIR SciFact corpus: 5,183 documents,
1,109 encoded queries, and 300 judged test queries. FastEmbed 0.8.0 generated
384-dimensional `BAAI/bge-small-en-v1.5` vectors and non-negative `Qdrant/bm25`
impacts. Dynamic WRRF had zero ordered Top-20 mismatches across all 1,109 queries,
fixed 1,106 before source exhaustion, and achieved Recall@20 0.9069 and NDCG@20
0.7536. The release identity-contiguous run had 2.26 ms dynamic p50 versus 2.80 ms
exhaustive p50.

The physical result is deliberately less flattering. Flat identity blocks and
four-iteration balanced k-means blocks both evaluated every Dense document. An
8-document hierarchical tree with Euclidean and spherical-cap bounds still
evaluated every Dense document on a 100-query diagnostic. Two-document leaves
finally skipped approximately 7.5% of exact document scores, but evaluated all
3,380 Node bounds and increased debug latency. This is evidence of high-dimensional
exact-search degeneration, not a correctness failure. `reference-exact` remains
the oracle; production mode must not claim that this Dense executor is faster
than HNSW without a larger-scale result. Raw baseline results are in
`docs/spectra/results/scifact-vector-snapshot-v1.local.json`.

The same SciFact vectors were also loaded into a real Qdrant Segment and queried
through both Qdrant HNSW and its exact Dense scan in one release process. The
single-threaded deterministic HNSW build took approximately 420 to 438 ms. The
base vector Segment occupied 44,065,422 bytes and the additional HNSW files
173,826 bytes. At Top-100, `ef=64` achieved 97.45% mean recall at 62.3 µs p50;
`ef=256` achieved 99.80% at 127.0 µs p50; and `ef=512` achieved 99.98% at
207.4 µs p50. Exact scan was stable around 175 µs p50. Thus HNSW crosses behind
exact scan before it becomes ordered-identical for every query on this small
corpus. The full four-point speed/recall matrix is in
`docs/spectra/results/scifact-dense-native-matrix-v1.local.json`.

The real Sparse snapshot initially exposed a generator defect: raw FastEmbed
hashed term IDs reached approximately 2.1 billion, while Qdrant's RAM inverted
index expects collection-local contiguous dimensions. The generator now freezes
the sorted corpus vocabulary, maps it to `0..V-1`, applies the same mapping to
queries, drops query-only terms, and records the raw vocabulary checksum. The
SciFact snapshot contains 26,542 remapped dimensions.

After remapping, Qdrant native Sparse build time fell from an invalid 29.9-second
run to 4.73 ms. Across 1,109 Top-100 queries, native inverted, standalone Block
Max, and exhaustive plain search had zero ordered mismatches. Native p50 was
27.1 µs (38,555 QPS), Block Max p50 699.5 µs (1,406 QPS), and exhaustive p50
323.2 µs. Standalone Block Max is therefore not the production Sparse executor;
its admissible Node Value must augment the inverted/WAND path. Raw results are in
`docs/spectra/results/scifact-sparse-native-v1.local.json`.

The original two-channel `spectra_exact_server` smoke is preserved in
`docs/spectra/results/scifact-exact-server-smoke-v1.local.json` as implementation
history. It has been superseded by the actual N-channel server. The v2 HTTP gate
accepts `channels[]`, calls `NChannelExactIndex`, returns per-channel progress and
physical-stream telemetry, rejects a stale generation with HTTP 400, and declares
the full-corpus ordered-exact guarantee. On its first SciFact request the Dense
channel pulled 4,936 identities without exhausting, while the Sparse channel
exhausted after 1,703; the Dynamic WRRF result was certified after 67 checks. The
recorded debug-build latency is a functional smoke value, not a performance
claim. Raw output is in
`docs/spectra/results/scifact-exact-server-smoke-v2.local.json`.

The 100K-document TREC-COVID gate changed the accepted Sparse executor. Qdrant
RAM inverted search with max-next pruning disabled produced bitwise-identical
Top-20 scored lists for all 50 queries at 0.385 ms p50. Adaptive native prefixes
starting at 4096 were then compared against Posting Block in six fully balanced
runs. Mean paired latency improved by 1.10/2.22/9.22 ms at N=2/4/8 with all
query-cluster bootstrap intervals below zero and no ordered mismatch. The frozen
4096 profile also improved N=4/8 on NFCorpus and SciFact. Raw artifacts are
`docs/spectra/results/trec-covid-100k-sparse-native-strict-v1.local.json`,
`docs/spectra/results/trec-covid-100k-posting-vs-adaptive4096-repeats-v2.local.json`,
`docs/spectra/results/nfcorpus-posting-vs-adaptive4096-repeats-v1.local.json`, and
`docs/spectra/results/scifact-posting-vs-adaptive4096-repeats-v1.local.json`.

A query/document signed-int8 kernel computes one low-bit dot product and a
deterministic residual-norm upper bound for every Dense identity, then reads the
original vector only while its interval can affect exact order. The flattened
per-vector implementation supersedes the first prototype. On five SciFact runs
it had zero Top-20 mismatches and 98.2--99.8 microsecond p50. On five
counterbalanced FiQA runs (57,600 documents, 100 queries), its median-run p50 was
0.853 ms versus 1.970 ms for Qdrant exact scan; all 500 paired observations were
faster and both exact methods had zero mismatch. It exactly rescored only 0.266%
of FiQA document/query pairs. Raw results are in
`docs/spectra/results/scifact-dense-quantized-certificate-v2.local.json` and
`docs/spectra/results/fiqa-dense-exact-repeats-v1.local.json`.

Run the current gate with:

```bash
cargo test -p segment common::reciprocal_rank_fusion --locked
cargo test -p segment dense_ball --locked
cargo test -p common top_k --locked
cargo test -p sparse search_context --locked
cargo test -p sparse block_max --locked
cargo run --release -p sparse --example spectra_sparse_baseline --locked
cargo run --release -p segment --example spectra_hybrid_exact --locked
```

The backend request and guarantee boundary is specified in
`docs/spectra/backend-contract.md`. The root `spectra_exact_server` binary exposes
the snapshot-backed single-shard reference kernel over one HTTP request. Spectra's
production adapter issues one native Qdrant Query request with Dense and Sparse
prefetch plus weighted RRF; its separate exact adapter verifies the stronger
full-corpus guarantee and restores authoritative Chunk identities. Sparse Node
Max should next move into the inverted executor rather than remain a parallel
scanner. A collection-level exact coordinator follows only after the single-shard
cost model is stable.
