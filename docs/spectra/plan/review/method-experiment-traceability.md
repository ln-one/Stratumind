# Method--experiment traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Exact N-channel fusion | Dynamic Exact WRRF | Property tests 1--8 channels; real full-corpus parity | T1 | Ordered Top-K equivalence under exact stream contract | Passed locally; broader datasets open |
| Sparse admissible physical stream | One SearchContext prefix + lazy compressed Posting stream | Repeated SciFact and synthetic-100K Top-20 and 64+1 prefix gates | T2, T3, F1, F2 | Exact portfolio; SearchContext wins shallow prefix, Posting preserves resumable completion | Five-run local gates passed; broader datasets open |
| Shallow exact prefix | One `B+1` SearchContext call with strict boundary | Boundary/tie property tests; repeated SciFact prefix gate | T3, F2 | Exact shallow prefix without repeated Top-N | Passed locally |
| Dense exact portfolio | Lazy exact prefix + compact signed-int8 residual + Scalar reconstruction + exact fallback | Repeated matched SciFact and 10K/100K dimension-scale matrices | T4, F4 | Exact Top-K; profile selects conditional latency winner and avoids large Compact storage | Five-run local gates passed from d64 to d384; broader hardware/datasets open |
| Shared N-channel execution | SharedNodeCatalog and identity dedupe | Channel agreement/adversarial matrix | T5 | Physical reuse increases with channel overlap | Synthetic preliminary |
| Executor portfolio | Safe exact plan Router and cancellation | 100K repeated and 1M preliminary scheduler matrices; Segment-owned kernel gates | T6, F3 | Plan errors affect cost, never result; conditional executor selection and exact full-scan degeneration | Local V1 implemented; held-out regret open |
| Production hybrid quality | Proposal/certificate/fallback | MIRACL/NQ/learned-sparse quality matrix | T7 | Pareto performance at matched quality | Pending |
