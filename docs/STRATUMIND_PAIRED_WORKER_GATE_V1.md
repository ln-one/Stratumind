# Paired Dense/Sparse Worker Gate V1

This gate evaluates the internal one-worker-per-Segment Dense/Sparse session candidate. It does
not change Production API V1.1 or the ordered Top-K guarantee.

Both variants used the same release binary, persisted collection, query and complete exact WRRF
oracle. Each row contains 20 warmups followed by 200 measured localhost HTTP requests.

| Points | Plan | p50 | p95 | Ordered mismatch |
|---:|---|---:|---:|---:|
| 512 | paired native | 1.388 ms | 1.648 ms | 0 |
| 512 | bounded materialized exact | 1.102 ms | 1.302 ms | 0 |
| 20,000 | paired native | 20.864 ms | 23.339 ms | 0 |
| 20,000 | bounded materialized exact | 16.639 ms | 18.336 ms | 0 |

The paired plan reduced four Segment task sets from eight workers to four and successfully ran with
`4 high-IO + 1 high-CPU coordinator`; it also preserved exactness, filtering, overwrite/delete and
restart behavior. It did not pass the latency gate: p50 was about 26% slower at 512 points and 25%
slower at 20,000 points, while p95 also regressed.

Production therefore keeps bounded materialized exact execution as the default. The paired
implementation remains available only for controlled experiments through
`STRATUMIND_EXPERIMENTAL_PAIRED_NATIVE_WORKERS=1`. It must not become the default without a new
counterbalanced release gate.
