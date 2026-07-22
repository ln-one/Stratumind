# Generated local Stratumind results

These artifacts are reproducible local measurements, not committed benchmark claims from upstream
Qdrant.

- `stratumind-v0-executor-matrix-100k-v3.local.json` is the current primary 100K two-channel
  scheduler and exact-plan Router matrix.
- `stratumind-v0-executor-matrix-1m-preliminary.local.json` is the single-repetition 1M scale and
  degeneration check. It is not publication-grade repeated evidence.
- `stratumind-v0-executor-matrix-100k-v1.local.json` and `v2` preserve the pre-fix result that exposed
  the incorrect treatment of eager Dense preparation as marginal pull cost.
- `stratumind-v0-linear-certifier-100k-preliminary.local.json` preserves the rejected linear
  certifier experiment; it was exact but slower than the short-circuit certifier at Top-20.
- `stratumind-v0-executor-matrix-100k-preliminary.local.json` is the initial smoke artifact.
- `stratumind-dense-auto-portfolio-scifact-v1.local.json` is the current five-run matched
  real-Segment exact/compact/scalar/Auto gate. All ordered Top-K comparisons are exact.
- `stratumind-dense-auto-portfolio-{10k-d384,100k-d64,100k-d192,100k-d256,100k-d384}-*.local.json`
  are the frozen size/dimension Router gates. `d64-v1` and `d192-v1` predate lazy candidate-set
  construction; `v2` supersedes them. Forced certificate arms are ablations and may persist files
  that the production Router omits.
- `stratumind-dense-segment-certified-synthetic-100k-d384-v2.local.json` and the corresponding d64
  `v2` preserve the rejected manual-SIMD experiment. The code was reverted.
- `stratumind-sparse-exact-plans-scifact-v1.local.json` is the five-run matched Top-20
  SearchContext-versus-Posting gate that rejects Posting as the default shallow plan.
- `stratumind-sparse-exact-plans-scifact-prefix65-v1.local.json` repeats the gate at the production
  64+1 prefix depth used by the one-prefix/lazy-Posting Router.
- `stratumind-sparse-exact-plans-synthetic-100k-{top20,prefix65}-v1.local.json` repeat both gates at
  100K scale; every strict prefix was accepted and every ordered list matched.
- `qdrant-stock-v1.18.2-exact-kernels-scifact-v1.local.json` records five clean-upstream runs at
  commit `44ad62f8c`; its artifact includes empty library diff evidence and harness checksums.
- `stratumind-exact-rrf-http-smoke-{seed,restart}-v1.local.json` record eight independent full-WRRF
  parity checks across two Shards, payload filtering, overwrite updates, deletes, and container
  restart persistence. Native and explicit-consistency exact-fallback cases all have zero ordered
  mismatch.

Regenerate the primary matrix and Segment-owned gates with the commands and frozen protocol in
`docs/spectra/plan/experiment-protocol.md`.
