# Exact V4 composition research

Status: archived research prototype

The implementation paths in this document are preserved only in Git history at
`b26cbf157`; they are not compatibility entry points in the active crate tree.

Date: 2026-07-21

Canonical semantics live in Spectra's
`docs/retrieval/retrieval-next-algorithm-v4.md`. The finite-DAG extension lives
in `docs/retrieval/retrieval-next-composition-v4.md`. This document records only
the Qdrant-kernel implementation and experiment boundary.

## Question

Can an exact leaf rank stream or the output of one complete V4 fusion be used as
an input to another V4 fusion, over a finite DAG, while preserving the exact
root Top-K of full materialization of that same network?

The implementation must answer a narrower systems question after correctness:
when do recursive demand, cross-level stopping, common-subexpression sharing,
cancellation, and safe exhaustion reduce physical cost?

An arbitrary DAG is the expression space, not the construction objective. A
physical planner minimizes cost for a fixed logical network. A logical topology
selector must optimize retrieval quality and cost together because rank-of-rank
fusion is not associative.

## Architecture boundary

Composition is isolated in three places:

- `lib/segment/src/index/exact_composition.rs`: validated logical plan, lazy
  execution, exhaustive oracle, and telemetry;
- `lib/segment/src/index/rank_stream.rs`: generic one-producer/many-reader exact
  stream replay and reference-counted cancellation;
- `lib/segment/examples/spectra_exact_composition.rs`: synthetic network and
  workload generation.

Dataset generation, topology matrices, subprocess resource measurement, and
statistical analysis stay under `lib/segment/examples` or `tools/spectra`. They
must not enter Qdrant's normal search path.

## Logical contract

`ExactCompositionNode` has two variants:

- `Leaf { order }`: an exact identity order used as the oracle; execution may
  replace it with any physically equivalent Dense, Sparse, remote, or nested
  exact stream;
- `Fusion { inputs, weights, rrf_k, output_limit }`: one WRRF operator.

`output_limit = Some(K)` is a semantic finite V4 output. `None` is an open exact
prefix family that a parent may continue pulling. These forms are not
interchangeable.

The external boundary also exposes a fallible exact stream. It distinguishes
normal EOF from cancellation, snapshot change, and source failure. Any failure
that execution actually observes aborts the root request; it cannot be converted
to EOF and certified as success. An unobserved tail failure does not invalidate
an already certified prefix because that tail was never part of the request.

The plan rejects missing inputs, cycles, duplicate identities inside a leaf,
empty fusion nodes, invalid weights, zero `rrf_k`, and zero finite limits. Every
failure must fail closed; iterator EOF cannot stand in for cancellation or an
internal fusion error.

## Physical execution

Production plans use `ExternalLeaf`: they retain no exhaustive identity order
and require an exact stream at execution time. Materialized `Leaf { order }`
exists only for deterministic tests and explicit oracles.

Every logical node owns one `SharedRankStream`. All outgoing edges subscribe at
rank zero before production starts. Each consumer has an independent cursor.
The producer advances only to the fastest cursor, while replay is retained down
to the slowest cursor.

For consumer cursors `p_c`, the instantaneous trade-off is explicit:

```text
saved duplicate pulls = sum_c(p_c) - max_c(p_c)
required replay       = max_c(p_c) - min_c(p_c)
```

Sharing is therefore not a free optimization. Similar consumer progress gives
both useful pull reuse and a small replay interval; a large cursor skew can
exchange duplicate work for replay approaching the whole corpus. A physical
planner must estimate both terms instead of selecting sharing from graph shape
alone.

An internal fusion uses one recoverable `DynamicRrfSession`. A parent request for
rank `k+1` extends the existing session instead of restarting child streams.
When a session reaches exact exhaustion, its complete fused order is sorted and
cached once, so later prefix extensions do not repeatedly rank the same state.

Dropping one reader cancels only that logical edge. The physical producer is
dropped when its last reader disappears. Root early stop therefore propagates
through unreachable work without invalidating another parent.

A synchronous pull executor cannot safely enforce a replay hard cap by simply
blocking the fastest reader. That reader may be the only path that can complete
the root proof and release the slow reader, so the cap can create an execution
deadlock. Safe bounded-memory choices are: select an unshared plan before the
request, restart or fork a deterministic producer from a frozen snapshot,
spill replay to bounded external storage, or use a global scheduler with a
proved progress rule. The prototype has neither a restart factory nor spill;
it must preselect the unshared plan when the replay budget cannot be honored.

## Exactness oracle

`exhaustive_execute` materializes every leaf and every internal node in
topological order, applying the same weights, tie order, finite limits, and
positive-score rule as lazy execution.

Every experiment has two separate comparisons:

1. lazy root Top-K versus full materialization of the same network: ordered
   mismatch must be zero;
2. one logical topology versus another: report ranking drift or judged quality,
   never call it an execution mismatch.

The current tests cover adversarial chains, shared diamonds, finite versus open
children, a concrete non-associativity counterexample, 250 deterministic random
DAGs, root cancellation, and actual Dense-quantized plus Sparse block-max leaf
streams.

## Cost telemetry

Lazy execution reports:

- physical leaf pulls;
- outputs produced by internal fusion nodes;
- per-node logical input pulls and certification checks;
- exact simultaneous peak replay identities across the whole execution;
- the sum of per-node replay peaks as a separate upper bound;
- cancelled node producers.

Full materialization reports leaf, intermediate, per-node, and total materialized
identity counts. Identity payload bytes are a lower bound, not process memory;
allocator overhead, score maps, source history, indexes, and input vectors are
not hidden inside that number. The matrix runner records a paired correctness
process plus separate dynamic-only and exhaustive-only child processes for every
configuration. Production `ExternalLeaf` plans retain no oracle arrays. The
resulting `ru_maxrss` values are end-to-end process peaks including indexes and
input baselines, not allocator-level increments, but they now compare the two
physical executions without cross-plan contamination.

## Safe exhaustion policy

`ExactCompositionExecutionPolicy` can set
`fusion_policy.exhaustive_after_pulls_per_source`. Once a local fusion spends the
configured pull budget without fixing its requested prefix, it stops repeated
certification attempts, drains its exact inputs, materializes one complete
order, and serves later ranks from that cache.

This is a physical fallback, not an approximation. The logical network and root
result do not change. Its purpose is to cap proof overhead on independent or
adversarial streams.

No router that sees only a finite consumed prefix can guarantee the cheapest
physical action for every input. Two stream families can expose the same prefix
at the decision point and then continue either highly correlated or strongly
conflicting; continuing proof is favorable in the first family, while draining
once is favorable in the second. The router cannot distinguish those futures.
The deployable objective is therefore expected cost with measured regret, plus
deterministic proof-work, replay, and latency fallbacks. Routing may switch only
between semantically equivalent physical plans, so a prediction miss changes
cost, never the exact root result.

## Exploratory observations, not publication results

The first debug-only sweep used 500 identities, four leaves, two generated
queries, Top-20, and one repeat. It exists to find failure modes before release
experiments.

- Correlated streams fixed after about 2% of full leaf materialization across
  the tested flat, chain, balanced, and diamond networks.
- Independent streams consumed 97-100% of leaf orders.
- Without exhaustion fallback, independent open chains and trees performed
  thousands of certification checks and were orders of magnitude slower than
  the full oracle in the debug build.
- A budget of 32-128 pulls per local input reduced those checks to tens and kept
  zero ordered mismatch. It still did not make nested dynamic execution faster
  than full materialization in that adverse case.
- Shared and unfolded diamonds returned identical rankings. Sharing removed the
  duplicate physical leaf pulls, but the shared case retained a larger replay
  window when its two consumers progressed at different rates.
- Independent chain, balanced, and diamond topologies differed from flat fusion,
  confirming that topology is query semantics rather than a free physical tree
  rotation.

These observations justify a router and a memory-aware sharing decision. They do
not establish a universal threshold, speedup, or preferred topology.

The first automated release artifacts are:

- `results/exact-composition-synthetic-bounded-v1.local.json`;
- `results/exact-composition-synthetic-unbounded-small-v1.local.json`;
- `results/exact-composition-scifact-bounded-v1.local.json`;
- `results/exact-composition-nfcorpus-bounded-v1.local.json`;
- `results/exact-composition-synthetic-leaves-{2,8,16}-v1.local.json`;
- `results/exact-composition-analysis-v1.local.json` and `.md`.

Across 186 configurations the ordered mismatch count is zero. The seven source
matrices and this analysis were regenerated from the reviewed Stratumind
release binaries on 2026-07-22. The original 114 core configurations retain the
following result: the bounded synthetic matrix ranges from 0.4% to 100% of full
leaf work and from 0.03x to 15.55x full-materialization latency. The deliberately unbounded small matrix
contains open-network cases thousands of times slower than full materialization,
which makes a safety budget mandatory. SciFact and NFCorpus confirm that shared
diamonds can save physical leaf pulls while retaining roughly one corpus worth
of replay and not reliably improving latency. These are execution findings;
next-query rewrite quality remains a stress construction rather than a product
claim.

The additional 72-configuration fan-in sweep holds the corpus at 5,000 and
varies the exact leaf count across 2, 8, and 16. Correlated streams use 0.4% of
full leaf work throughout and take roughly 0.026x-0.045x exhaustive latency. In
independent or conflicting streams, open chains and balanced networks consume
the whole leaf order; the worst measured dynamic/exhaustive latency ratio grows
from 1.31x at two leaves to 20.75x at eight and 21.20x at sixteen. N-channel
correctness composes, but nested proof overhead does not scale safely without a
router and fallback.

Isolated dynamic/exhaustive process-RSS ratios have medians of 0.46 on bounded
synthetic, 0.71 on SciFact, and 0.72 on NFCorpus. The two-, eight-, and
sixteen-leaf sweeps have medians 0.62, 0.43, and 0.36. The unbounded adversarial
matrix reaches 1.10, so lazy execution is usually smaller in these runs but has
no universal memory advantage.

The analyzer now reports a three-objective topology table over Recall@K,
nDCG@K, and dynamic latency. On both ten-query snapshot samples there is no
single dominant logical topology: flat and shared-diamond plans both remain on
the measured Pareto set, trading quality dimensions against latency, while the
semantically equivalent unfolded diamond is dominated by physical sharing in
these runs. This demonstrates the selection method, not a production topology
winner: the judged sample is tiny and the next-query rewrite is deliberately a
stress construction. Logical topology needs representative qrels or downstream
labels; only physical-plan routing can remain quality-neutral.

## Required experiment matrix

Formal synthetic runs must cross:

- correlated, clustered, independent, anti-correlated, and tie-heavy rankings;
- flat, open chain, open balanced tree, finite balanced tree, shared diamond,
  and semantically equivalent unfolded diamond;
- multiple corpus sizes, depths, fan-outs, Top-K values, intermediate limits,
  and exhaustion budgets;
- dynamic, safe-fallback, and full-materialization execution;
- counterbalanced order, warmups, repeated release runs, and isolated-process
  memory measurement.

Real runs must reuse frozen SciFact, NFCorpus, and scale snapshots. Dense and
Sparse leaf streams must be produced by the existing exact indexes over the same
snapshot and query generation. Each topology is compared to its own full oracle;
topology selection additionally reports qrels-based Recall/nDCG and cost.

The primary outputs are zero ordered mismatch, leaf work, intermediate work,
certification checks, replay peak, process memory, cancellation latency, and
end-to-end latency. Shared versus unfolded execution is a mandatory ablation.

## Decision rule

No execution strategy is selected because it wins one mean latency. A candidate
must first pass zero mismatch and fail-closed tests, then show a stable Pareto
improvement on held-out workload families. If observable query or prefix
features cannot route a case reliably, the system uses the conservative local
budget and full-materialization fallback.

No claim of novel ranked-operator composition is currently justified. A paper
claim, if supported by later experiments, must be about the exact V4 semantics,
shared-DAG execution, topology non-associativity, cost routing, and measured
Dense/Sparse boundary rather than the existence of nested ranked iterators.
