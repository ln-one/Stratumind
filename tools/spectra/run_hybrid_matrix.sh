#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-/tmp/spectra-qdrant-target}"
cargo_bin="${CARGO:-cargo}"
output_dir="${SPECTRA_OUTPUT_DIR:-${repo_root}/docs/spectra/results/generated}"

documents="${SPECTRA_DOCUMENTS:-20000}"
queries="${SPECTRA_QUERIES:-100}"
top_k="${SPECTRA_TOP_K:-20}"
block_size="${SPECTRA_BLOCK_SIZE:-128}"
dense_dimension="${SPECTRA_DENSE_DIMENSION:-32}"

mkdir -p "${output_dir}"

(
  cd "${repo_root}"
  CARGO_TARGET_DIR="${target_dir}" "${cargo_bin}" build \
    --release \
    --locked \
    -p segment \
    --example spectra_hybrid_exact
)

binary="${target_dir}/release/examples/spectra_hybrid_exact"
for scenario in clustered interleaved anti_correlated flat_ties; do
  SPECTRA_SCENARIO="${scenario}" \
  SPECTRA_DOCUMENTS="${documents}" \
  SPECTRA_QUERIES="${queries}" \
  SPECTRA_TOP_K="${top_k}" \
  SPECTRA_BLOCK_SIZE="${block_size}" \
  SPECTRA_DENSE_DIMENSION="${dense_dimension}" \
    "${binary}" > "${output_dir}/hybrid-${scenario}.json"
done
