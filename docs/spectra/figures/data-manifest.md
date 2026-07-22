# Figure data manifest

| Figure | Claim measured | Real data files | Script/status |
|---|---|---|---|
| F1 | Sparse strategy latency and physical work across N | `../results/trec-covid-100k-posting-vs-adaptive4096-repeats-v2.local.json`, `../results/nfcorpus-posting-vs-adaptive4096-repeats-v1.local.json`, `../results/scifact-posting-vs-adaptive4096-repeats-v1.local.json`, `../results/sparse-posting-block-max-v1.local.json`, `../results/generated/stratumind-sparse-exact-plans-scifact-v1.local.json` | six-run multi-domain matrix plus five-run Segment-kernel matched gate present |
| F2 | Exact prefix/refill break-even | `../results/trec-covid-100k-adaptive-native-v1.local.json`, `../results/trec-covid-100k-adaptive-native-initial4096-v1.local.json`, `../results/trec-covid-100k-adaptive-native-initial8192-v1.local.json`, `../results/trec-covid-100k-sparse-native-strict-v1.local.json`, `../results/generated/stratumind-sparse-exact-plans-scifact-prefix65-v1.local.json` | historical adaptive sweep plus accepted one-prefix 64+1 gate present |
| F3 | Router regret and exact physical sharing boundaries | `../results/shared-sparse-executor-study-v1.local.json`, `../results/synthetic-rank-sharing-repeats-v1.local.json`, `../results/synthetic-dense-replay-repeats-v1.local.json`, `../results/synthetic-dense-independent-repeats-v1.local.json`, `../results/generated/stratumind-v0-executor-matrix-100k-v3.local.json`, `../results/generated/stratumind-v0-executor-matrix-1m-preliminary.local.json` | repeated 100K scheduler boundary and preliminary 1M degeneration check present; broader plotting pending |
| F4 | Dense recall/certificate/latency frontier | `../results/scifact-dense-native-matrix-v1.local.json`, `../results/scifact-dense-quantized-certificate-v2.local.json`, `../results/scifact-dense-qdrant-scalar-certified-v1.local.json`, `../results/fiqa-dense-exact-repeats-v1.local.json`, `../results/fiqa-dense-bound-gate-v1.local.json`, `../results/scifact-dense-proposal-ball-gate-v1.local.json`, `../results/generated/stratumind-dense-auto-portfolio-scifact-v1.local.json`, `../results/generated/stratumind-dense-auto-portfolio-100k-d384-v1.local.json` | repeated FiQA evidence plus five-run Segment-owned SciFact and 10K/100K Auto portfolio gates present |
| F5 | Cross-dataset exact latency, work, Recall@20, NDCG@20 | `../results/fiqa-n-channel-exact-quality-v2.local.json`, `../results/nfcorpus-n-channel-exact-v1.local.json`, `../results/trec-covid-100k-n-channel-exact-v1.local.json` | FiQA, full-query NFCorpus, and 100K-document TREC-COVID exact scale gate present; multilingual quality gate pending |
| F6 | Clean upstream exact-kernel control | `../results/generated/qdrant-stock-v1.18.2-exact-kernels-scifact-v1.local.json` | five-run v1.18.2 Dense/Sparse control with empty upstream library diff and harness checksums present |

The HTTP interface smoke is not plotted as a performance figure. Its current functional artifacts
are `../results/generated/stratumind-exact-rrf-http-smoke-seed-v1.local.json` and
`../results/generated/stratumind-exact-rrf-http-smoke-restart-v1.local.json`; they cover two-Shard
native execution, filters, overwrites, deletes, and restart persistence.

No mock values are used in current result prose. Any future planning-only dataset
must use a `mock_` or `synthetic_` prefix and carry the required planning label.
