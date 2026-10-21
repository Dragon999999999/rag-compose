# Configurable RAG pipelines

Pipeline YAML is an import mechanism; PostgreSQL is the runtime source of truth.
The files under `configs/pipelines/` are validated at startup against the closed
component registry and `configs/models.yaml`, then synchronized into the
`pipelines` table.

Each pipeline has two identities:

- `config_hash` identifies the complete source configuration.
- `representation_hash` identifies parser, knowledge builder, chunking, embedding,
  vector-backend, and index-version choices. A changed representation identity
  requires a distinct ingestion/index representation and cannot reuse an
  incompatible vector index.

Model connection settings are kept in the shared catalog. Pipeline files contain
only named references such as `embedding_001` and `reasoning_001`; credentials are
resolved by the existing environment-backed provider adapters.

YAML-owned records are updated when their file fingerprint changes. Records
created or edited through an API are marked `source_kind: api` and are never
overwritten by startup synchronization with a same-name YAML file.

## Reranking configuration

The query `reranker` stage accepts the legacy `type` spelling and the equivalent
`provider` spelling. `identity` preserves the retrieval response, `top_k` only
truncates it, and `cross_encoder` uses the optional local Sentence Transformers
cross-encoder. `retrieval_top_k` controls the candidate count requested from the
retriever; `rerank_top_k` controls the final count.

```yaml
reranker:
  provider: cross_encoder
  retrieval_top_k: 20
  rerank_top_k: 5
  model_name: cross-encoder/ms-marco-MiniLM-L6-v2
  device: cpu
  batch_size: 16
```

The cross-encoder is loaded lazily and stores weights below
`~/.cache/rag-eval/models` by default. Set `RAG_EVAL_MODEL_CACHE_DIR` in the
environment or `.env` to mount a persistent application-specific cache.
