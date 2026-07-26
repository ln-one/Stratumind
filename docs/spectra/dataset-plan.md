# Spectra retrieval dataset plan

Status: archived experiment plan. Commands and example paths refer to
`b26cbf157`, not the active clean-break crate tree.

The test set is not one frozen bag of queries. It is a reproducible generator
plus adapters for externally licensed real corpora. Correctness, physical cost,
and retrieval quality are separate gates.

## Synthetic generator

`lib/segment/examples/spectra_hybrid_exact.rs` creates aligned Sparse and Dense
representations over one frozen point identity universe. The same exhaustive
rankings are the oracle for every generated query.

The required scenarios are:

- `clustered`: physical blocks follow semantic clusters, so admissible bounds
  should be tight;
- `interleaved`: clusters are mixed across physical blocks, deliberately making
  Dense balls and Sparse envelopes loose;
- `anti_correlated`: Sparse and Dense queries target different clusters, forcing
  dynamic WRRF to inspect deep prefixes;
- `flat_ties`: both channels contain large equal-score regions, exercising the
  frozen identity tie contract.

Every scenario must report ordered Top-K mismatches, stop reason, source pulls,
certification checks, blocks expanded, exact document evaluations, build cost,
and latency percentiles. Full-corpus fallback is a valid cost result; a different
ordered Top-K is not.

Run the fixed matrix with:

```bash
bash tools/spectra/run_hybrid_matrix.sh
```

Environment variables control corpus size, query count, Top-K, block size, Dense
dimension, Cargo target directory, and result directory. Compilation is outside
the timed experiment.

## Real corpus adapter contract

The Qdrant-kernel experiment should not own model inference. A preparation step
produces two immutable JSONL files plus a manifest:

```text
manifest.json
corpus-vectors.jsonl
query-vectors.jsonl
qrels.tsv                  # optional for execution-only corpora
```

Each corpus row contains a stable string identity, a finite Dense vector, and a
sorted finite non-negative Sparse impact vector. Each query row contains the same
two channel representations. The manifest pins:

- source dataset name, version, URL, checksum, and license note;
- exact corpus/query split and any deterministic sample seed;
- chunking procedure;
- Dense model name, revision, normalization, and dimension;
- Sparse model/tokenizer parameters and vocabulary checksum;
- identity-to-Qdrant-point-ordinal mapping;
- generator commit and output checksums.

This boundary lets the kernel compare execution strategies over identical input
without silently mixing encoder changes into latency or correctness claims.

Sparse model term IDs are not written directly. The generator sorts every raw
term ID observed in the selected corpus, maps that vocabulary to contiguous `u32`
dimensions, remaps documents and queries through the same table, and drops query
terms absent from the corpus. The manifest records the dimension count, mapping
rule, and SHA-256 of the ordered raw vocabulary. This is required for Qdrant's
RAM inverted representation and prevents a large hashed term ID from allocating
billions of empty posting-list slots.

The first preparation implementation should use Qdrant-maintained FastEmbed
instead of owning model inference code. Its supported-model registry currently
lists the MIT-licensed `BAAI/bge-small-en-v1.5` Dense model and Apache-2.0
`Qdrant/bm25` Sparse model:
<https://qdrant.github.io/fastembed/examples/Supported_Models/>. The generator must
pin the FastEmbed version and model revisions; “latest” is not reproducible.
The current evaluation environment pins `fastembed==0.8.0` in
`tools/spectra/requirements-eval.txt`; generated snapshots additionally record
the installed version and checksums of every output.

`tools/spectra/prepare_beir_fastembed.py` converts BEIR files into the snapshot
contract. It preserves every positively judged document for selected queries,
uses FastEmbed passage/query modes, computes Qdrant's collection-level BM25 IDF
modifier explicitly, rejects negative or non-finite impacts, and writes outputs
atomically. Model inference and dataset preparation remain outside timed kernel
runs.

The first frozen local adapter run used BEIR SciFact with these source checksums:

```text
archive     536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165
corpus      dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6
queries     8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313
test qrels  0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c
```

It produced 5,183 corpus rows, 1,109 query rows, and 340 qrels rows. The generated
vectors are local research artifacts and are not committed because SciFact is
non-commercial. Reproduce the adapter with:

```bash
python tools/spectra/prepare_beir_fastembed.py \
  --dataset-dir /path/to/scifact \
  --output-dir /tmp/spectra-scifact-snapshot \
  --dataset-name scifact \
  --dataset-version beir-release \
  --dataset-url https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
  --dataset-license "CC BY-NC 2.0; research-only, not bundled with Spectra"
```

Then run the same-kernel oracle comparison with:

```bash
SPECTRA_DATASET_DIR=/tmp/spectra-scifact-snapshot \
cargo run --release -p segment --example spectra_vector_snapshot --locked
```

Before a publication run, the downloaded model directories must be copied to an
immutable artifact store and checksummed in the manifest. FastEmbed version plus
model name is sufficient for local iteration, but is not by itself a permanent
pin of a mutable remote model repository.

The cross-domain adapter was then run on NFCorpus. The BEIR archive MD5 is
`a89dba18a62ef92f7d323ec890a0d38d`; the local snapshot contains 3,633 documents,
323 test queries, and 12,334 qrels under CC-BY-SA-4.0. Fifteen valid queries have
no Sparse dimension after corpus-vocabulary remapping. They remain empty exact
Sparse streams rather than receiving invented terms; this exposed and removed an
experiment-runner assumption that every Sparse query was non-empty. The snapshot
manifest is local at `/tmp/spectra-nfcorpus-snapshot/manifest.json` and the
committed execution artifacts contain only measurements, not dataset vectors.

The next scale gate uses the official TREC-COVID BEIR archive, whose MD5 is
`ce62140cb23feb9becf6270d0d1fe6d1`. Its deterministic 100K-document snapshot
preserves every positively judged document before filling the remainder by the
frozen sample seed. Empty source documents are skipped and any affected query is
recorded explicitly by the manifest; no silent qrel loss is permitted. The BEIR
dataset card labels its packaged dataset CC-BY-SA-4.0, but NIST identifies the
document collection as CORD-19 and states that it is subject to its data license.
Because CORD-19 aggregates papers from several publishers and archives, the
snapshot is research-only and is not redistributed; source-specific document
rights must not be collapsed into one blanket product license.

The generated local snapshot contains 100,000 documents, all 50 test queries,
and 54,718 retained judgments. Every positively judged document is present.
TREC-COVID also contains two `-1` judgments; the evaluator stores signed grades,
counts only grades above zero as relevant, and assigns non-positive grades zero
DCG gain. The first exact scale artifact is
`results/trec-covid-100k-n-channel-exact-v1.local.json`.

## First real datasets

1. A fixed Natural Questions development subset is the first document-shaped
   corpus candidate. Google publishes the dataset under CC BY-SA 3.0 and provides
   stable JSONL data and utilities:
   <https://ai.google.com/research/NaturalQuestions/download>.
2. BEIR format is the general retrieval adapter because it standardizes corpus,
   query, and qrels files across many domains:
   <https://github.com/beir-cellar/beir>. BEIR explicitly warns that each
   underlying dataset must be licensed and cited separately.
3. Qdrant's Apache-2.0 vector database benchmark is the Dense systems-baseline
   harness, not the source of hybrid relevance truth:
   <https://github.com/qdrant/vector-db-benchmark>.

## Multilingual gate

The first multilingual execution gate is the MIRACL Chinese development split.
MIRACL is Apache-2.0 and publishes 393 Chinese development queries with 3,928
judgments. Its full Chinese corpus is approximately 4.9 million passages, so the
first local gate uses the same deterministic selection rule as the TREC-COVID
scale snapshot: preserve every positive, then fill to a declared limit with the
frozen seed. This sampled gate tests cross-language executor correctness and
physical behavior; it is not reported as full-corpus MIRACL retrieval quality.

Dense encoding uses Qdrant/FastEmbed's supported MIT
`BAAI/bge-small-zh-v1.5` model (512 dimensions). Qdrant BM25 must first pass a
Chinese tokenization diagnostic. Raw `Qdrant/bm25` treated each of three
unspaced diagnostic sentences as one token, so raw mode is rejected for Chinese
quality evaluation. The frozen adapter now offers MIT `jieba==0.42.1`
pretokenization before the same Qdrant BM25 impact and collection-IDF pipeline.
The diagnostic sentences then produced 4/5/5 tokens and 4/5/5 nonzero impacts.
The manifest records `sparse.pretokenizer`; Dense encoding always receives the
unaltered source text. A later full MIRACL run requires a binary vector snapshot
format; serializing 4.9 million float vectors as JSONL is deliberately outside
the current gate.

MS MARCO and SciFact may be used only in separately marked research runs: their
official terms are non-commercial. They must never be bundled with Spectra,
published in generated artifacts, or treated as evidence that the product data
pipeline itself is commercially licensed.

## Quality versus execution

The optimized executor passes the execution gate only when:

```text
dynamic ordered WRRF Top-K == exhaustive ordered WRRF Top-K
```

NDCG, Recall, MRR, reranker gain, and evidence quality measure the chosen query
and representation pipeline. They cannot repair an execution mismatch. Likewise,
exact execution does not prove that the encoder or chunking scheme is relevant.
