# Dense certified execution

## Accepted exact portfolio

The current Dense default is a metadata-routed portfolio of exact Segment-owned streams:

- dimension at most 192: one lazy Qdrant exact `B+1` prefix;
- higher dimension and at most 16,384 eligible points: compact per-vector signed-int8 residual
  certificate;
- higher dimension and larger Segment: Qdrant Scalar reconstruction certificate;
- unsupported configuration or unsafe boundary: exact scan.

The compact plan computes a low-bit dot product and adds a deterministic residual-norm and
floating-point guard. The Scalar plan uses its persisted reconstruction error. An identity is
rescored in full precision only while its upper interval can still precede the greatest known exact
point. Score descending and frozen identity ascending define the total order. Prefix candidates are
exposed only after a strict `B/B+1` boundary or exact EOF; a tie starts the fallback before any
prefix identity is emitted.

Every member is a complete exact stream, not an approximate proposal. A difficult query may scan
and rescore every vector; it may not return a different rank. The original vectors remain
authoritative. Router thresholds influence cost only.

## Evidence

On five counterbalanced FiQA runs with 57,600 documents, 100 queries, 384
dimensions, and Top-20:

- compact certificate: zero mismatch, p50 0.853 ms, p95 1.011 ms;
- Qdrant exact scan: zero mismatch, p50 1.970 ms, p95 2.193 ms;
- Qdrant HNSW `ef=256`: p50 0.659 ms, mean Recall@20 0.995, 50 ordered-list
  mismatches over five runs;
- compact full-precision rescore ratio: 0.266%;
- compact versus Qdrant exact mean paired delta: -1.132 ms, query-cluster
  bootstrap 95% interval [-1.145, -1.118] ms.

The raw artifact is
`docs/spectra/results/fiqa-dense-exact-repeats-v1.local.json`.

The portfolio is persisted and executed through the same Qdrant Segment. On the regenerated
SciFact snapshot (5,183 documents, 100 queries, Top-20), five counterbalanced runs produced zero
ordered mismatches. Auto selected Compact for all 500 measured queries and measured 92,833 ns
median-run p50 versus 180,583 ns for Qdrant exact, a median latency ratio of 0.5137. It won all 500
paired comparisons; its mean paired delta had query-cluster bootstrap 95% interval
[-95,376.980, -89,273.032] ns. The raw artifact is
`docs/spectra/results/generated/stratumind-dense-auto-portfolio-scifact-v1.local.json`.

The frozen 100K synthetic dimension sweep also had zero mismatch. Auto used exact prefix at 64 and
192 dimensions, measuring ratios of 1.0172 and 1.0158 versus exact scan: the exact-stream
abstraction cost about 1.6--1.7%, rather than claiming a speedup. It used Scalar at 256 and 384
dimensions, measuring ratios of 0.8893 and 0.7844. At 10K x 384 it used Compact and measured a ratio
of 0.8179. These thresholds are a frozen training-free V0 profile, not a universal hardware
constant. Production omits the Compact file above 16,384 points; forced-Compact scale artifacts are
ablation storage, not production storage.

## N-channel sharing

`ReplayIdentical` detects bitwise-identical Dense queries with identical WRRF
weights before interval construction. One physical exact producer feeds N
logical readers. Every remaining unique Dense query constructs its compact
exact stream independently. `SharedDocumentMajor` remains only an exact
no-replay layout ablation.

In the five-run 100K-document N=8 mixed boundary, four duplicate Dense channels
reduced physical/logical Dense dots to 0.25 and p50 from 1.632 ms to 0.477 ms,
with zero exhaustive WRRF mismatch. The raw artifact is
`docs/spectra/results/synthetic-dense-replay-repeats-v1.local.json`.

With all eight Dense queries unique, the independent layout was faster than
`SharedDocumentMajor`: 1.477 ms versus 1.622 ms median-run p50 across five
counterbalanced runs, despite identical physical dot-product counts. The mean
paired delta was -0.159 ms with query-cluster bootstrap 95% interval
[-0.183, -0.134] ms. This freezes the training-free layout choice without
qrels or dataset identity. The raw artifact is
`docs/spectra/results/synthetic-dense-independent-repeats-v1.local.json`.

## Rejected certificate family

Query-independent Euclidean balls, spherical caps, and coordinate boxes were
safe but degenerated in 384 dimensions. On FiQA, a 16-document hierarchical tree
still scored 1,151,590 of 1,152,000 document/query pairs. Adding coordinate boxes
removed no additional work and increased p50 about 11%.

HNSW proposal did not repair the bound. On the 100-query SciFact gate HNSW found
the exact Top-20 proposal every time, yet the Ball certificate still expanded
every leaf and rescored every vector. This path is experimental evidence, not a
default executor.

## Literature boundary

Certified Cosine provides strict certificates using an exact K-nearest-neighbor
graph and a certified neighborhood radius for every indexed point. Reusing an
ordinary HNSW graph is insufficient because omitted closer neighbors make that
radius unsafe; constructing the required exact graph can dominate offline cost.

RaBitQ provides a sharp randomized distance-estimation bound and fast bitwise
execution. Its published probabilistic bound is not by itself a deterministic
ordered-exact certificate. A future integration must either retain an actual
per-vector residual bound, or classify RaBitQ under the approximate track.

Primary references:

- Francis-Landau and Van Durme, *Exact and/or Fast Nearest Neighbors*:
  https://arxiv.org/abs/1910.02478
- Gao and Long, *RaBitQ*:
  https://arxiv.org/abs/2405.12497

## Architecture boundary

- `dense_quantized.rs`: accepted compact exact interval index and stream;
- `n_channel_streams.rs`: channel-native identity and telemetry adapters;
- `rank_stream.rs`: channel-agnostic physical replay;
- `n_channel_exact.rs`: strategy selection and orchestration only;
- `dense_proposal_certified.rs`: experimental HNSW/Ball gate, not the default;
- `examples/` and `docs/spectra/results/`: reproducible baselines and rejected
  paths.
