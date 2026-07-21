# Figure data manifest

| Figure | Claim measured | Real data files | Script/status |
|---|---|---|---|
| F1 | Sparse strategy latency and physical work across N | `../results/trec-covid-100k-posting-vs-adaptive4096-repeats-v2.local.json`, `../results/nfcorpus-posting-vs-adaptive4096-repeats-v1.local.json`, `../results/scifact-posting-vs-adaptive4096-repeats-v1.local.json`, `../results/sparse-posting-block-max-v1.local.json` | six-run balanced Posting-versus-Adaptive comparison on three real domains present |
| F2 | Adaptive prefix refill break-even | `../results/trec-covid-100k-adaptive-native-v1.local.json`, `../results/trec-covid-100k-adaptive-native-initial4096-v1.local.json`, `../results/trec-covid-100k-adaptive-native-initial8192-v1.local.json`, `../results/trec-covid-100k-sparse-native-strict-v1.local.json` | Top-20/4096/8192 sweep and strict native baseline present; 4096 frozen in V3 Profile |
| F3 | Router regret and exact physical sharing boundaries | `../results/shared-sparse-executor-study-v1.local.json`, `../results/synthetic-rank-sharing-repeats-v1.local.json`, `../results/synthetic-dense-replay-repeats-v1.local.json`, `../results/synthetic-dense-independent-repeats-v1.local.json`, `../results/n-channel-synthetic-1m-router-boundaries-v1.local.json` | five-run Sparse and Dense rank-sharing plus all-unique Dense layout boundaries present; broader plotting pending |
| F4 | Dense recall/certificate/latency frontier | `../results/scifact-dense-native-matrix-v1.local.json`, `../results/scifact-dense-quantized-certificate-v2.local.json`, `../results/scifact-dense-qdrant-scalar-certified-v1.local.json`, `../results/fiqa-dense-exact-repeats-v1.local.json`, `../results/fiqa-dense-bound-gate-v1.local.json`, `../results/scifact-dense-proposal-ball-gate-v1.local.json` | repeated FiQA exact win plus FiQA ball/box and SciFact HNSW+Ball negative gates present |
| F5 | Cross-dataset exact latency, work, Recall@20, NDCG@20 | `../results/fiqa-n-channel-exact-quality-v2.local.json`, `../results/nfcorpus-n-channel-exact-v1.local.json`, `../results/trec-covid-100k-n-channel-exact-v1.local.json` | FiQA, full-query NFCorpus, and 100K-document TREC-COVID exact scale gate present; multilingual quality gate pending |

The HTTP interface smoke is not plotted as a performance figure. Its current
functional artifact is `../results/scifact-exact-server-smoke-v2.local.json`;
the v1 file documents the superseded two-channel adapter.

No mock values are used in current result prose. Any future planning-only dataset
must use a `mock_` or `synthetic_` prefix and carry the required planning label.
