# RAG benchmark suite

The repository contains a reproducible matrix of 75 benchmark entries covering
15 families. The source pipeline files live below
`config/pipelines/benchmarks/`, grouped by family. Query and ingestion
pipelines are separate files; the manifest joins them for each experiment.

## Contents

| Family | Scope |
| --- | --- |
| 01 baselines | MiniLM, BGE-M3, and a stronger hybrid baseline |
| 02 chunking | Fixed, recursive, sentence, structural, block-aware, semantic, hierarchical, late, and LLM-guided chunking |
| 03 embeddings | MiniLM, BGE-M3, Nomic, and Aqueduct Qwen |
| 04 vector databases | Qdrant, pgvector, and FAISS |
| 05 retrieval | Dense, summary, BM25, PostgreSQL FTS, learned sparse, hybrid, multivector, and full-context retrieval |
| 06 reranking | Identity, top-k, cross-encoder, BGE, Jina, mxbai, and Qwen reranking |
| 07 knowledge | Basic, structural, rules, spaCy, and GLiNER knowledge construction |
| 08 assessors | No-op, relevance, diversity, evidence sufficiency, confidence, and source-reference assessment |
| 09 decision policies | Acceptance, selection, budget, consensus, adaptive, and fallback policies |
| 10 interactions | Six high-value stage combinations |
| 11–13 branching | Retrieval, reranker, and assessor branch topologies |
| 14 recursive alternatives | Bounded assessor, reranker, and retriever retries |
| 15 combined | Branching followed by partial retry and adaptive multi-alternative execution |

The authoritative experiment metadata is in
[`benchmarks/rag_benchmark_manifest.yaml`](../benchmarks/rag_benchmark_manifest.yaml).
The checked-in files can be regenerated deterministically with
`python scripts/generate_rag_benchmarks.py`.
It records stable IDs, pipeline references, changed components, hypotheses,
required resources, metrics, and compatibility constraints.

## Validation and import

Validate one preset without starting application services:

```bash
llm-grounding pipeline validate config/pipelines/benchmarks/01_baselines/benchmark_base_02.yaml
```

Import the complete suite through the existing application-backed pipeline
synchronizer:

```bash
llm-grounding pipeline import config/pipelines/benchmarks
```

The loader recursively discovers the numbered family directories. The normal
`config/pipelines` discovery also sees the suite, so the same files are
available to startup synchronization. Pipeline import validates component IDs,
model references, execution topology, and representation compatibility before
persistence.

## Experimental controls

The suite keeps parser, reasoner, renderer, prompts, chat history, retrieval
depth, context budgets, and generation settings fixed unless a family explicitly
studies that stage. Representation-changing experiments have their own
ingestion preset and therefore produce a new representation hash/index. Existing
document representations are not rewritten by selecting a benchmark pipeline.

Use the same fixed corpus, query set, seed, and evaluation harness for every
family. Report deterministic pipeline metrics separately from stochastic model
quality measurements. Suggested metrics are recall@K, precision@K, MRR,
nDCG@K, context relevance, answer correctness, faithfulness, citation
accuracy, evidence sufficiency, abstention accuracy, latency, inference time,
token consumption, resource usage, and estimated cost. Only metrics supported by
the evaluation framework should be emitted.

## Known boundaries

`CHUNK-08` is retained in the manifest as `unsupported`: the registered late
chunker requires `embedding_jina_v2_base_en`, but this checkout has no
composition component for that model. Adding an invented alias would violate
the closed registry requirement.

Learned sparse and multivector retrieval entries are valid query compositions,
but require their corresponding sparse or multivector indexes to be prepared by
the deployment/indexing setup. The ordinary dense ingestion preset does not
silently create those representations. Branch definitions use the current
execution contract's implementation names; exact per-model settings for a
branch require the runtime's branch-aware model configuration and are called out
in each affected manifest entry.

No credentials are stored in the presets or manifest. Model YAMLs retain only
credential environment-variable names, and CLI discovery continues to redact
secret values.
