# Exact composition matrix summary

Every correctness comparison uses full materialization of the same logical network. Topology drift is reported separately and is not an execution mismatch.

## Artifact overview

| Artifact | Configs | Mismatch | Leaf ratio min/median/max | Dynamic/full latency min/median/max | Max replay | Paired RSS MiB | Isolated RSS D/E MiB | D/E RSS ratio min/median/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact-composition-synthetic-bounded-v1.local.json | 60 | 0 | 0.004/1.000/1.000 | 0.03/1.18/15.55 | 5001 | 34.4 | 16.4/30.5 | 0.24/0.46/0.92 |
| exact-composition-synthetic-unbounded-small-v1.local.json | 30 | 0 | 0.020/0.922/1.000 | 0.11/4.45/4986.65 | 1001 | 12.1 | 9.1/11.2 | 0.65/0.83/1.10 |
| exact-composition-scifact-bounded-v1.local.json | 12 | 0 | 0.933/1.000/1.000 | 1.11/4.75/5.56 | 5184 | 58.8 | 38.3/56.3 | 0.64/0.71/0.74 |
| exact-composition-nfcorpus-bounded-v1.local.json | 12 | 0 | 0.730/1.000/1.000 | 0.94/5.17/8.20 | 3634 | 47.1 | 29.8/40.6 | 0.70/0.72/0.89 |
| exact-composition-synthetic-leaves-2-v1.local.json | 24 | 0 | 0.004/1.000/1.000 | 0.04/1.12/1.31 | 1 | 22.3 | 10.3/24.1 | 0.33/0.62/0.75 |
| exact-composition-synthetic-leaves-8-v1.local.json | 24 | 0 | 0.004/1.000/1.000 | 0.03/1.19/20.75 | 1 | 47.0 | 18.5/41.7 | 0.24/0.43/0.94 |
| exact-composition-synthetic-leaves-16-v1.local.json | 24 | 0 | 0.004/1.000/1.000 | 0.03/1.18/21.20 | 1 | 75.1 | 31.2/64.3 | 0.20/0.36/0.81 |

## Shared versus unfolded diamond

Ratios below 1 favor sharing for that metric. Replay is the exact simultaneous identity count.

| Artifact | Workload | Budget | Shared/unfolded leaf pulls | Shared/unfolded latency | Replay shared/unfolded |
|---|---|---:|---:|---:|---:|
| exact-composition-nfcorpus-bounded-v1.local.json | spectra-nfcorpus-snapshot | 32 | 0.685 | 0.971 | 3634/1 |
| exact-composition-nfcorpus-bounded-v1.local.json | spectra-nfcorpus-snapshot | 128 | 0.706 | 0.921 | 3634/1 |
| exact-composition-scifact-bounded-v1.local.json | spectra-scifact-snapshot | 32 | 0.759 | 0.943 | 5184/1 |
| exact-composition-scifact-bounded-v1.local.json | spectra-scifact-snapshot | 128 | 0.767 | 0.912 | 5183/1 |
| exact-composition-synthetic-bounded-v1.local.json | anti_correlated | 32 | 0.800 | 0.995 | 5001/1 |
| exact-composition-synthetic-bounded-v1.local.json | anti_correlated | 128 | 0.800 | 0.977 | 5001/1 |
| exact-composition-synthetic-bounded-v1.local.json | clustered | 32 | 0.800 | 0.986 | 5000/1 |
| exact-composition-synthetic-bounded-v1.local.json | clustered | 128 | 0.800 | 0.925 | 5000/1 |
| exact-composition-synthetic-bounded-v1.local.json | correlated | 32 | 0.800 | 0.960 | 2/1 |
| exact-composition-synthetic-bounded-v1.local.json | correlated | 128 | 0.800 | 0.805 | 2/1 |
| exact-composition-synthetic-bounded-v1.local.json | flat_ties | 32 | 0.800 | 1.013 | 2/1 |
| exact-composition-synthetic-bounded-v1.local.json | flat_ties | 128 | 0.800 | 0.945 | 2/1 |
| exact-composition-synthetic-bounded-v1.local.json | independent | 32 | 0.800 | 0.919 | 5001/1 |
| exact-composition-synthetic-bounded-v1.local.json | independent | 128 | 0.800 | 0.941 | 5001/1 |
| exact-composition-synthetic-unbounded-small-v1.local.json | anti_correlated | none | 0.800 | 1.047 | 1001/1 |
| exact-composition-synthetic-unbounded-small-v1.local.json | clustered | none | 0.801 | 1.015 | 983/1 |
| exact-composition-synthetic-unbounded-small-v1.local.json | correlated | none | 0.800 | 1.125 | 2/1 |
| exact-composition-synthetic-unbounded-small-v1.local.json | flat_ties | none | 0.800 | 0.855 | 2/1 |
| exact-composition-synthetic-unbounded-small-v1.local.json | independent | none | 0.799 | 0.969 | 538/1 |

## Snapshot topology quality and cost

Pareto means no topology in the same artifact and execution budget has at least as much Recall and nDCG with no more dynamic latency.

| Artifact | Topology | Budget | Judged | Recall@K | nDCG@K | Dynamic mean ms | Leaf ratio | Pareto |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| exact-composition-nfcorpus-bounded-v1.local.json | balanced_finite | 32 | 10 | 0.1555 | 0.1468 | 3.959 | 1.000 | yes |
| exact-composition-nfcorpus-bounded-v1.local.json | balanced_open | 32 | 10 | 0.1665 | 0.1458 | 26.284 | 1.000 | no |
| exact-composition-nfcorpus-bounded-v1.local.json | chain_open | 32 | 10 | 0.1565 | 0.0972 | 21.985 | 1.000 | no |
| exact-composition-nfcorpus-bounded-v1.local.json | diamond_shared | 32 | 10 | 0.1740 | 0.2550 | 26.598 | 1.000 | yes |
| exact-composition-nfcorpus-bounded-v1.local.json | diamond_unfolded | 32 | 10 | 0.1740 | 0.2550 | 27.398 | 1.000 | no |
| exact-composition-nfcorpus-bounded-v1.local.json | flat | 32 | 10 | 0.1723 | 0.2606 | 3.987 | 1.000 | yes |
| exact-composition-nfcorpus-bounded-v1.local.json | balanced_finite | 128 | 10 | 0.1555 | 0.1468 | 3.431 | 0.730 | yes |
| exact-composition-nfcorpus-bounded-v1.local.json | balanced_open | 128 | 10 | 0.1665 | 0.1458 | 34.482 | 1.000 | no |
| exact-composition-nfcorpus-bounded-v1.local.json | chain_open | 128 | 10 | 0.1565 | 0.0972 | 23.297 | 1.000 | no |
| exact-composition-nfcorpus-bounded-v1.local.json | diamond_shared | 128 | 10 | 0.1740 | 0.2550 | 27.523 | 1.000 | yes |
| exact-composition-nfcorpus-bounded-v1.local.json | diamond_unfolded | 128 | 10 | 0.1740 | 0.2550 | 29.871 | 0.969 | no |
| exact-composition-nfcorpus-bounded-v1.local.json | flat | 128 | 10 | 0.1723 | 0.2606 | 3.748 | 1.000 | yes |
| exact-composition-scifact-bounded-v1.local.json | balanced_finite | 32 | 10 | 0.9000 | 0.4839 | 8.805 | 1.000 | yes |
| exact-composition-scifact-bounded-v1.local.json | balanced_open | 32 | 10 | 0.9000 | 0.5630 | 41.399 | 1.000 | no |
| exact-composition-scifact-bounded-v1.local.json | chain_open | 32 | 10 | 0.3500 | 0.1326 | 49.426 | 1.000 | no |
| exact-composition-scifact-bounded-v1.local.json | diamond_shared | 32 | 10 | 1.0000 | 0.6718 | 48.863 | 1.000 | yes |
| exact-composition-scifact-bounded-v1.local.json | diamond_unfolded | 32 | 10 | 1.0000 | 0.6718 | 51.844 | 1.000 | no |
| exact-composition-scifact-bounded-v1.local.json | flat | 32 | 10 | 0.9000 | 0.6452 | 9.090 | 1.000 | yes |
| exact-composition-scifact-bounded-v1.local.json | balanced_finite | 128 | 10 | 0.9000 | 0.4839 | 9.200 | 1.000 | no |
| exact-composition-scifact-bounded-v1.local.json | balanced_open | 128 | 10 | 0.9000 | 0.5630 | 39.647 | 1.000 | no |
| exact-composition-scifact-bounded-v1.local.json | chain_open | 128 | 10 | 0.3500 | 0.1326 | 49.663 | 0.974 | no |
| exact-composition-scifact-bounded-v1.local.json | diamond_shared | 128 | 10 | 1.0000 | 0.6718 | 44.032 | 0.943 | yes |
| exact-composition-scifact-bounded-v1.local.json | diamond_unfolded | 128 | 10 | 1.0000 | 0.6718 | 48.259 | 0.933 | no |
| exact-composition-scifact-bounded-v1.local.json | flat | 128 | 10 | 0.9000 | 0.6452 | 9.098 | 1.000 | yes |

## Interpretation guardrails

- A lower leaf-work ratio is not a latency guarantee; certification, replay, and iterator overhead remain visible.
- Shared execution is selected only when saved physical work exceeds replay and coordination cost.
- Different topologies are different ranking semantics unless separately proved equivalent.
- Snapshot quality uses the deterministic next-query rewrite stress construction; it is not a production rewrite quality claim.
