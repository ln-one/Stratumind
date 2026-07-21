# Shared N-channel matrix v1 (local, preliminary)

Command shape:

```bash
SPECTRA_DOCUMENTS=20000 SPECTRA_QUERIES=50 \
SPECTRA_CHANNELS={1,2,4,8} \
SPECTRA_SCENARIO={correlated,independent,anti_correlated,flat_ties} \
SPECTRA_TOP_K=20 SPECTRA_BLOCK_SIZE=128 \
target/release/examples/spectra_shared_n_channel
```

This isolates exact fusion and shared identity/Node materialization. Channel
scores are deterministic synthetic values; therefore these timings are not a
Dense or Sparse production claim.

All 800 query/configuration pairs had zero ordered Top-K mismatches against
exhaustive full-channel WRRF.

| Scenario | N | Dynamic p50 | Exhaustive p50 | Logical Node opens | Unique physical opens | Reuse factor |
|---|---:|---:|---:|---:|---:|---:|
| correlated | 1 | 50.8 us | 2.70 ms | 937 | 937 | 1.0x |
| correlated | 2 | 73.2 us | 3.27 ms | 1,874 | 937 | 2.0x |
| correlated | 4 | 110.3 us | 4.36 ms | 3,748 | 937 | 4.0x |
| correlated | 8 | 174.8 us | 6.35 ms | 7,496 | 937 | 8.0x |
| independent | 1 | 49.5 us | 2.75 ms | 951 | 951 | 1.0x |
| independent | 2 | 8.67 ms | 3.41 ms | 15,700 | 7,850 | 2.0x |
| independent | 4 | 22.49 ms | 4.47 ms | 31,400 | 7,850 | 4.0x |
| independent | 8 | 58.20 ms | 6.79 ms | 62,800 | 7,850 | 8.0x |
| anti-correlated | 1 | 54.5 us | 2.69 ms | 937 | 937 | 1.0x |
| anti-correlated | 2 | 8.54 ms | 3.40 ms | 15,700 | 7,850 | 2.0x |
| anti-correlated | 4 | 17.15 ms | 4.48 ms | 31,400 | 7,850 | 4.0x |
| anti-correlated | 8 | 43.48 ms | 6.56 ms | 62,800 | 7,850 | 8.0x |
| flat ties | 1 | 22.8 us | 2.70 ms | 192 | 192 | 1.0x |
| flat ties | 2 | 32.7 us | 3.25 ms | 384 | 192 | 2.0x |
| flat ties | 4 | 48.4 us | 4.26 ms | 768 | 192 | 4.0x |
| flat ties | 8 | 81.0 us | 6.20 ms | 1,536 | 192 | 8.0x |

Interpretation:

- Physical identity/Node materialization sharing works exactly as designed: when
  channels touch the same Nodes, unique opens stay constant as `N` grows.
- Sharing alone does not solve deep rank-fusion inspection. Independent and
  anti-correlated channels consume nearly complete streams and the current heap
  executor loses to exhaustive sort.
- A production executor therefore needs both fusion-aware early stopping and a
  conservative native/exhaustive fallback. Cross-channel agreement and bound
  tightness are candidate router features.
- These numbers are one warm local run and have no confidence intervals yet.
