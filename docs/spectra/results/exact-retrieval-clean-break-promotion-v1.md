# Exact Retrieval Clean-Break Promotion Gate V1

## Decision

**PASS with a declared dataset substitution.**

The clean-break candidate preserves the exact retrieval contract and passes every
measured correctness, work, fallback, p50, and p95 gate. The deleted local
TREC-COVID vector snapshot was replaced by a deterministic 100K/384D hybrid
scale gate. NFCorpus and SciFact remain the real-data gates. A historical
ExactRankSession run on TREC-COVID 100K is supporting evidence, not a current
binary result.

The candidate is suitable for a separate Production promotion change. It should
not be described as having rerun the current binary on TREC-COVID.

## Compared builds

| Role | Source | Binary SHA-256 |
|---|---|---|
| Baseline | `b26cbf157` | `5826b141958dacace6e2ab057cdc6a6755811ccaa301029d25d566da0698ee8c` |
| Candidate | `74b7e5e1f` | `ddb179350cf4acf79e3d81d9b683983bfb81d47351430f180c9b37e0e8d1293e` |

Both binaries were built in release mode from clean source states. Baseline and
candidate used separate immutable Collections because the clean break
intentionally replaces implicit PVS discovery with the explicit
`dense_sparse_v1` profile. Collection construction and optimization were
excluded from measured latency.

## Results

Negative latency deltas favor the candidate.

| Dataset | Queries | Paired observations | Ordered mismatch | Candidate p50 vs baseline | Candidate p95 vs baseline | Mean delta | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus, 3,633 documents | 323 | 1,615 | 0 | +0.44% | -8.13% | -6.23% | 0% |
| SciFact, 5,183 documents | 1,109 | 5,545 | 0 | -5.57% | -16.16% | -11.92% | 0% |
| Synthetic, 100K documents, 384D | 100 | 500 | 0 | -3.57% | -1.42% | -2.33% | 0% |

The candidate and baseline materialized exactly the same mean number of source
points and performed exactly the same mean number of source pulls on every
dataset. The refactor therefore did not hide extra retrieval work behind a
latency win.

Paired query-cluster bootstrap 95% confidence intervals for the mean candidate
latency delta were:

- NFCorpus: `[-209,508 ns, -82,167 ns]`
- SciFact: `[-261,908 ns, -210,853 ns]`
- Synthetic 100K/384D: `[-3,899,039 ns, -2,497,287 ns]`

All three intervals favor the clean-break candidate.

## Gate evaluation

| Gate | Limit | Result |
|---|---:|---|
| Ordered Top-20 mismatch | 0 | PASS |
| Exact query errors | 0 | PASS |
| Exhaustive fallback | 0% | PASS |
| Source pulls/materialization | no increase | PASS, identical |
| p50 regression | no more than 3% | PASS, worst case +0.44% |
| p95 regression | no more than 5% | PASS, all improved |
| Explicit profile persistence | round-trip | PASS |
| Profile toggle | marks Segment incompatible for rebuild | PASS in both directions |

## Scope and limitation

The original local TREC-COVID 100K vector snapshot had been deleted before this
gate. Rebuilding BGE-small embeddings locally was measured first and rejected as
an hours-long prerequisite. The scale replacement uses deterministic normalized
384D Dense vectors and non-negative Sparse impacts over 100K points. It validates
large-Collection execution, exact ordering, work preservation, and latency, but
does not reproduce TREC-COVID score distributions.

The preceding ExactRankSession candidate did run on the former TREC-COVID 100K
snapshot: 250 paired queries, zero mismatch, p50 improved by 2.38%, p95 by
5.54%, and source work was identical. That historical run supports the
architecture but is deliberately not counted as a clean-break binary gate.

## Reproduction

The current results are:

- `nfcorpus-exact-clean-break-gate-v1.local.json`
- `scifact-exact-clean-break-gate-v1.local.json`
- `synthetic-100k-384d-exact-clean-break-gate-v1.local.json`

The black-box runner is `tools/spectra/run_exact_clean_break_gate.py`. It creates
and optimizes both Collections before timing, counterbalances binary order,
reuses the same stable Query permutations, and compares the complete ordered
Top-20 identities.
