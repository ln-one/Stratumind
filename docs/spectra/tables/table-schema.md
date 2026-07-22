# Result table schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| T1 | Exactness across N | Dataset x N x strategy | mismatches, stop reason, pulls | property logs and real matrices | retrieval team |
| T2 | Sparse physical execution | dataset x scale x strategy | p50/p95/p99, postings, blocks, bytes, build | Sparse raw JSON | retrieval team |
| T3 | Prefix-depth boundary | requested depth x growth policy | refills, visits, latency, regret | adaptive sweep JSON | retrieval team |
| T4 | Dense executor matrix | exact/HNSW/certificate variants | mismatch/recall, exact rescoring, latency, bytes | Dense raw JSON | retrieval team |
| T5 | Shared N-channel ablation | correlation x N x sharing | materializations, cache hits, latency | shared synthetic JSON | retrieval team |
| T6 | Router evaluation | workload region x router | oracle choice, regret, p99, abstention | held-out router logs | retrieval team |
| T7 | End-to-end quality | dataset x production system | Recall/NDCG, latency, throughput, memory | MIRACL/NQ/learned-sparse logs | retrieval team |
| T8 | Segment-owned native gate | channel x size x dimension x exact plan | mismatch, p50/p95, paired delta CI, physical counters, certificate bytes | `results/generated/stratumind-{dense,sparse}-*.json` | retrieval team |

Repeated tables report mean and 95% paired bootstrap confidence intervals.
Single-run local values are labeled preliminary and never merged with repeated
publication values.
