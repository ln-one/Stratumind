# Generated local Stratumind results

These artifacts are reproducible local measurements, not committed benchmark claims from upstream
Qdrant.

- `stratumind-v0-executor-matrix-100k-v3.local.json` is the current primary 100K two-channel
  scheduler and exact-plan Router matrix.
- `stratumind-v0-executor-matrix-100k-v1.local.json` and `v2` preserve the pre-fix result that exposed
  the incorrect treatment of eager Dense preparation as marginal pull cost.
- `stratumind-v0-linear-certifier-100k-preliminary.local.json` preserves the rejected linear
  certifier experiment; it was exact but slower than the short-circuit certifier at Top-20.
- `stratumind-v0-executor-matrix-100k-preliminary.local.json` is the initial smoke artifact.

Regenerate the primary matrix with the command in `docs/spectra/plan/experiment-protocol.md`.
