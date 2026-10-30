# Configurable RAG pipelines

Pipeline YAML is the source format; PostgreSQL is the runtime source of truth
after synchronization. Canonical composition files live under
`config/pipelines/`. They are validated against the standalone model registry in
`config/models/` and component registry in `config/components/`, then persisted
as resolved pipeline snapshots.

Each pipeline has two identities:

- `config_hash` identifies the complete source configuration.
- `representation_hash` identifies parser, knowledge builder, chunking,
  embedding, vector-backend, and index-version choices. A changed representation
  identity requires a distinct ingestion/index representation and cannot reuse an
  incompatible vector index.

Model connection settings are kept in standalone model definitions. Pipeline
files contain component IDs such as `embedding_local_minilm` and
`reasoner_basic_grounded`; components reference model IDs internally. Credentials
are resolved by the existing environment-backed provider adapters.

Standalone model definitions use the stage tags `embedding`, `reasoner`,
`reranker`, or `validation`. The registry uses these tags to expose models only
to compatible component stages; backend type alone does not determine stage
eligibility.

YAML-owned records are updated when their file fingerprint changes. Records
created or edited through the API are marked `source_kind: api` and are not
overwritten by startup synchronization with a same-name YAML file.

## Reranking configuration

The canonical query `reranker` stage references a component ID. Operational
settings stay in the reusable component file. `identity` preserves retriever
ordering, `top_k` only truncates it, and `cross_encoder` uses an optional local
Sentence Transformers model. The generative reranker uses a configured model
through the normal model runtime. `retrieval_top_k` controls the candidate count
requested from the retriever; `rerank_top_k` controls the final count.

```yaml
query:
  reranker: reranker_cross_encoder

# config/components/reranker/reranker_cross_encoder.yaml
id: reranker_cross_encoder
type: reranker
implementation: cross_encoder
config:
  model_id: cross_encoder_minilm
  retrieval_top_k: 20
  rerank_top_k: 5
  device: cpu
  batch_size: 16
```

The cross-encoder is loaded lazily and stores weights below
`~/.cache/rag-eval/models` by default. Set `RAG_EVAL_MODEL_CACHE_DIR` in the
environment or `.env` to use another persistent cache.

## Query execution

Query pipelines can define an `execution.alternatives` sequence. The first entry
is the primary path; each later entry is an ordered fallback alternative. An
alternative can define local branches, stage overrides, and a source alternative
to reuse. The configured decision policy evaluates each candidate result. An
accepted answer stops execution; a retry, abstain, or reject outcome can advance
to the next alternative according to the configured policy.

An alternative may reference an earlier alternative and restart at `retriever`,
`reranker`, or `assessor`. The runtime passes the referenced alternative's
in-memory candidates or chunks into that stage, so reuse does not perform a
second retrieval or depend on persisted intermediate data. When the source
alternative is branched, `source.path_id` identifies the exact branch output to
reuse.

When all alternatives are exhausted, `final_failure_policy: best_result` returns
the strongest candidate according to the assessments already produced by the
pipeline. `reject` and `abstain` remain available for explicit failure behavior.
Ingestion execution remains project-scoped, while query execution is selected by
the chat.

For example:

```yaml
query:
  execution:
    max_attempts: 2
    final_failure_policy: best_result
    alternatives:
      - id: primary
      - id: evidence-retry
        source:
          alternative_id: primary
          path_id: primary
          restart_stage: assessor
```
