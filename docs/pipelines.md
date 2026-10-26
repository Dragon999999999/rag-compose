# Configurable RAG pipelines

Pipeline YAML is an import mechanism; PostgreSQL is the runtime source of truth.
Canonical composition files live under `config/pipelines/`. They are validated
against the standalone model registry in `config/models/` and component registry
in `config/components/`, then synchronized into the `pipelines` table.

Each pipeline has two identities:

- `config_hash` identifies the complete source configuration.
- `representation_hash` identifies parser, knowledge builder, chunking, embedding,
  vector-backend, and index-version choices. A changed representation identity
  requires a distinct ingestion/index representation and cannot reuse an
  incompatible vector index.

Model connection settings are kept in standalone model definitions. Pipeline files
contain only component IDs such as `embedding_local_minilm` and
`reasoner_basic_grounded`; components reference model IDs internally. Credentials
are resolved by the existing environment-backed provider adapters.

Each standalone model definition includes a required `stage` tag: `embedding`,
`reasoner`, or `reranker`. The registry uses this tag to expose models only to
the matching pipeline stage; backend type alone does not determine stage
eligibility.

YAML-owned records are updated when their file fingerprint changes. Records
created or edited through an API are marked `source_kind: api` and are never
overwritten by startup synchronization with a same-name YAML file.

## Reranking configuration

The canonical query `reranker` stage references a component ID. Operational
settings stay in the reusable component file. `identity` preserves the retrieval
response, `top_k` only truncates it, and `cross_encoder` uses the optional local
Sentence Transformers cross-encoder. `retrieval_top_k` controls the candidate
count requested from the retriever; `rerank_top_k` controls the final count.

```yaml
query:
  reranker: reranker_cross_encoder

# config/components/reranker/cross_encoder.yaml
id: reranker_cross_encoder
type: cross_encoder
model_id: cross_encoder_minilm
retrieval_top_k: 20
rerank_top_k: 5
device: cpu
batch_size: 16
```

The cross-encoder is loaded lazily and stores weights below
`~/.cache/rag-eval/models` by default. Set `RAG_EVAL_MODEL_CACHE_DIR` in the
environment or `.env` to mount a persistent application-specific cache.
