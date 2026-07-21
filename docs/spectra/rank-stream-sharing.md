# Exact rank-stream sharing

## Contract

Dynamic Exact WRRF continues to observe one logical sorted stream per channel.
Each logical stream keeps its original weight, rank, cancellation point, and
source-pull history. Physical sharing is legal only when it cannot change any
identity at any logical rank.

The implementation shares Sparse or Dense channels whose query coordinates and
WRRF-weight bits are identical. It deliberately does not normalize positive
scalar multiples because floating-point reassociation could change a tie. Dense
duplicate detection happens before signed-int8 ranking construction: only one
physical exact stream is built for each equal-weight duplicate group. Replay
after constructing every Dense stream would be fake optimization and is retained
only as the `SharedDocumentMajor` ablation.

## Physical mechanism

`index/rank_stream.rs` owns the replay coordinator. One exact producer appends
identities to a bounded replay window and independent readers consume that
window. Once every live reader has passed an identity, it is removed; dropping
a reader also releases its retention point. Therefore memory is proportional to
the maximum temporary rank divergence inside a shared group, not to the total
stream length. `buffered_identities` reports the peak resident window. The
orchestrator does not know how the producer finds its next identity. The
implementation reports logical/physical streams, logical/physical pulls, shared
groups, and buffered identities.

The equal-WRRF-weight restriction bounds divergence between readers under the
current greatest-next-contribution scheduler. Different-weight duplicates stay
independent until a bounded replay or weight-coalescing proof is implemented.

## Training-free executor rule

For the current Sparse query set:

- `U` is the union posting volume decoded by Shared Lazy.
- `D` is the sum of posting volumes after exact equal-weight rank duplicates are
  removed.

`AutoSharedLazy` chooses Shared Lazy only when `U<D`; otherwise it uses Posting
Block plus transparent rank replay. No qrels, corpus label, fitted threshold, or
dataset identity participates in this decision.

The router computes `U` and `D` together: it builds both term-multiplicity maps
but reads each union term's posting length only once. This removes duplicate
index metadata probes without changing the rule. Latency comparisons must still
run without concurrent encoder load; the mathematical work estimate alone is
not a latency claim.

## Evidence and limits

The five-run Sparse 100K-document N=8 boundary is recorded in
`results/synthetic-rank-sharing-repeats-v1.local.json`. Fully repeated channels
used one eighth of the logical rank pulls and reached 0.129 ms median-run p50;
the independent boundary performed no replay and stayed within roughly one
percent of Posting Block. All runs matched exhaustive WRRF ordered Top-K.

The Dense boundary is recorded in
`results/synthetic-dense-replay-repeats-v1.local.json`. With N=8 mixed channels,
four bitwise-identical Dense queries, 100K documents, and 100 queries per run,
`ReplayIdentical` reduced Dense physical/logical dot products to 0.25 and rank
pulls to 0.625. Across five counterbalanced runs its median-run p50 was 0.477 ms
versus 1.632 ms for the same `SharedDocumentMajor` execution with replay
disabled. The 500 paired observations had mean delta -1.191 ms with a
query-cluster bootstrap 95% interval of [-1.219, -1.164] ms. Both strategies had
zero exhaustive WRRF mismatch.

The no-duplicate boundary is recorded separately in
`results/synthetic-dense-independent-repeats-v1.local.json`. When all eight
Dense queries were unique, independent compact-stream construction beat the
document-major layout: 1.477 ms versus 1.622 ms median-run p50, with mean paired
delta -0.159 ms and query-cluster bootstrap 95% interval
[-0.183, -0.134] ms. Both layouts performed exactly the same dot products and
produced zero mismatches. The frozen rule is therefore: replay exact duplicates,
construct every remaining unique Dense stream independently, and retain
`SharedDocumentMajor` only as an ablation.

The cross-domain Sparse boundary is recorded in
`results/nfcorpus-n-channel-executor-repeats-v1.local.json`. Five
counterbalanced runs over all 323 NFCorpus queries produced zero ordered Top-K
mismatches. At N=8, `AutoSharedLazy` selected sharing for 1,540 of 1,615
executions and reduced the selected groups from 2,791,375 logical to 926,005
physical posting elements. Its mean paired latency delta against Posting Block
was -19.3 us with a query-cluster bootstrap 95% interval of
[-32.0, -6.8] us. At N=4 the Auto estimator overhead was measurable, so the
accepted claim remains conditional physical reuse, not universal latency
dominance.

This is not a claim that shared scans or batched inverted retrieval are new.
Fagin et al.'s threshold work established exact aggregation over sorted access,
and prior batch-query systems share inverted-index work across externally
batched queries. Recent GPUSparse similarly uses batched scatter-add for exact
learned Sparse retrieval. The Spectra claim is the composition of replaceable
exact channel streams, a Dynamic WRRF stopping certificate, and safe physical
sharing inside one N-channel retrieval request.

Primary comparison points:

- Fagin, Lotem, and Naor, “Optimal Aggregation Algorithms for Middleware”:
  https://doi.org/10.1016/S0022-0000(03)00026-6
- Mackenzie and Moffat, “Index-Based Batch Query Processing Revisited”:
  https://jmmackenzie.io/pdf/mm23-ecir.pdf
- Sharma, “GPUSparse: GPU-Accelerated Learned Sparse Retrieval with Parallel
  Inverted Indices”: https://arxiv.org/abs/2606.26441

## Architecture boundary

- `n_channel_exact.rs`: public query/result contract and orchestration only.
- `n_channel_streams.rs`: channel-specific identity and telemetry adapters.
- `rank_stream.rs`: generic physical replay beneath logical exact streams.
- `shared_posting_block_stream.rs`: Sparse aligned-batch certificates and
  union-term materialization.
- benchmark and router failures remain in `examples/` and `docs/spectra/results/`;
  they do not become default production branches.
