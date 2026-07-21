#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-/tmp/spectra-qdrant-target}"
cargo_bin="${CARGO:-cargo}"
output_dir="${SPECTRA_OUTPUT_DIR:-${repo_root}/docs/spectra/results/generated}"
dataset_dir="${SPECTRA_DATASET_DIR:?SPECTRA_DATASET_DIR must point to a prepared snapshot}"

top_k="${SPECTRA_TOP_K:-100}"
hnsw_m="${SPECTRA_HNSW_M:-16}"
ef_construct="${SPECTRA_HNSW_EF_CONSTRUCT:-100}"
build_threads="${SPECTRA_HNSW_BUILD_THREADS:-1}"

mkdir -p "${output_dir}"

(
  cd "${repo_root}"
  CARGO_TARGET_DIR="${target_dir}" "${cargo_bin}" build \
    --release \
    --locked \
    -p segment \
    --example spectra_dense_native_baseline
)

binary="${target_dir}/release/examples/spectra_dense_native_baseline"
for ef in 64 128 256 512; do
  SPECTRA_DATASET_DIR="${dataset_dir}" \
  SPECTRA_TOP_K="${top_k}" \
  SPECTRA_HNSW_EF="${ef}" \
  SPECTRA_HNSW_M="${hnsw_m}" \
  SPECTRA_HNSW_EF_CONSTRUCT="${ef_construct}" \
  SPECTRA_HNSW_BUILD_THREADS="${build_threads}" \
    "${binary}" > "${output_dir}/dense-native-ef-${ef}.json"
done
