# ExactRankSession Production Promotion V1.1.3

Date: 2026-07-24

Status: promotion gate passed.

## Compared plans

- Baseline: resident Segment workers at `487b5efb8`.
- Candidate: reader-independent `ExactRankSession` after `5fdd51ab9`.
- Dense + Sparse, WRRF Top-20, `k=60`.
- Apple Silicon macOS, release fat-LTO.
- One complete Query sweep for warmup and five counterbalanced measured sweeps.
- Index construction, optimization and service startup are excluded from query latency.

## Single-query results

| Dataset / layout | Requests | Baseline p50 | Session p50 | p50 improvement | Baseline p95 | Session p95 | p95 improvement | Throughput improvement | Mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus / 2 Segments | 1,615 | 1.607 ms | 1.425 ms | 11.33% | 3.482 ms | 3.031 ms | 12.96% | 13.10% | 0 |
| SciFact / 2 Segments | 5,545 | 2.016 ms | 1.854 ms | 8.05% | 5.275 ms | 4.609 ms | 12.63% | 10.27% | 0 |
| TREC-COVID 100K / 2 Segments | 250 | 13.223 ms | 12.907 ms | 2.38% | 41.383 ms | 39.090 ms | 5.54% | 4.43% | 0 |
| NFCorpus / 4 Segments | 1,615 | 1.653 ms | 1.455 ms | 11.95% | 3.516 ms | 3.117 ms | 11.36% | 13.11% | 0 |
| NFCorpus / 8 Segments | 1,615 | 1.796 ms | 1.409 ms | 21.57% | 3.682 ms | 3.079 ms | 16.37% | 25.23% | 0 |

The candidate completed every request exactly, used no exhaustive fallback, consumed the same
Dense/Sparse outputs as the baseline, and produced zero ordered Top-20 mismatches.

## Four concurrent requests over eight Segments

NFCorpus, 323 Queries × 5 rounds:

| Metric | Resident workers | ExactRankSession |
|---|---:|---:|
| Exact success | 1,040 | 1,615 |
| Capacity fail-closed | 575 | 0 |
| p50 | 3.657 ms | 2.861 ms |
| p95 | 7.009 ms | 5.396 ms |
| Throughput | 796.38 QPS | 1,287.06 QPS |

`ExactRankSession` raised exact success from `64.40%` to `100%`, reduced p50 by `21.78%`, reduced
p95 by `23.00%`, and increased throughput by `61.61%`. The 1,040 requests completed by both plans
had zero ordered Top-20 mismatches.

## Promotion decision

The implementation passed the frozen gates:

- zero ordered mismatch;
- no Segment-count materialized fallback;
- no single-query p50 or p95 regression;
- more than 15% throughput improvement at eight Segments and four-query concurrency;
- no resident Segment worker between checkpoints.

Production therefore promotes `ExactRankSession` as the local Dense + Sparse physical executor.
The HTTP API, exact WRRF semantics, tie rule, PVS/PBM mathematics and persisted index formats do not
change. The earlier paired-worker gate remains in the repository as a historical negative result.
