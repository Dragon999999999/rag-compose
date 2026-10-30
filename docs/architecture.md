# Configuration architecture

The application separates declarative configuration from runtime construction
and persistence. A pipeline names reusable components; components select a
registered implementation and its operational settings; models provide local or
remote runtime details.

```text
Pipeline composition
        │ references component IDs
        ▼
Configuration registry
        │ resolves component implementations and model IDs
        ▼
Typed runtime configuration
        │ constructs providers, stores, and stage services lazily
        ▼
Application services and pipeline runtime
```

## Repository layout

```text
config/
├── models/                         # standalone local and remote model definitions
├── components/<category>/          # reusable stage definitions and schemas
├── pipelines/                      # ingestion, query, test, and benchmark compositions
└── defaults.yaml                   # application-level component defaults
prompts/                            # registered prompt assets and templates
src/llm_grounding/
├── domain/                         # projects, documents, evidence, answers, and runs
├── parsing/                        # positioned document extraction
├── knowledge/                      # canonical evidence and metadata builders
├── retrieval/                      # dense, lexical, sparse, hybrid, and multivector retrieval
├── reasoning/assessment/decision/  # answer generation and grounded decisions
├── rendering/                      # validated answer and citation rendering
├── orchestration/                  # component registry and RAG execution
├── infrastructure/                 # configuration, persistence, providers, and stores
└── application/                    # use cases, synchronization, and API composition
```

## Standalone definitions

Canonical definitions live under [`../config`](../config):

```text
config/
├── models/                         # one provider or local model per YAML file
├── components/<category>/          # reusable stage configurations
└── pipelines/                      # composition files containing stable IDs
```

Model filenames are discovery details only. The YAML `id` is the authoritative
identity and must be unique across all model files. Model definitions currently
cover remote OpenAI-compatible models, local Sentence Transformers models,
GLiNER knowledge models, and deterministic test models. Supported model stages
are `embedding`, `reasoner`, `reranker`, and `validation` (the CLI presents the
last stage as knowledge-related model usage where appropriate).

Credentials are represented only by environment-variable names such as
`AQUEDUCT_API_KEY` and `AQUEDUCT_API_KEY_FILE`; secret values never belong in
YAML. Local weights are downloaded and verified lazily through the model manager.
Local embedding models may declare query and document prefixes or dense, sparse,
and multivector representations; the selected provider applies those settings at
both ingestion and query time.

Components have stable IDs, a category, an implementation, optional model IDs,
operational settings, requirements, and optional input/output terms. Component
family schemas under `config/components/*/schema.yaml` define accepted settings,
defaults, and override scopes. Pipeline files contain component IDs and
composition metadata; they do not contain provider endpoints, model names, or
credentials.

## Resolution and persistence

`ConfigurationRegistry` loads standalone files, rejects duplicate IDs, checks
category ownership, resolves model references, and converts definitions into
typed runtime schemas. `ComponentRegistry` validates that every implementation
is registered before a provider or storage client is constructed. The loader
also validates execution topology, representation compatibility, and model-stage
eligibility.

`load_ingestion_config`, `load_pipeline_config`, and `load_model_catalog` accept
the standalone format. Pipeline YAML uses `schema_version: 2` and must declare
either `kind: ingestion` with representation stages or `kind: query` with query
stages. There is no combined pipeline format.

At application startup, component definitions and YAML-owned pipeline definitions
are synchronized into PostgreSQL. Persisted pipelines retain resolved snapshots,
source kind, configuration hashes, representation hashes, and active state. API-
owned definitions remain independent from later synchronization of checkout YAML.
Projects select ingestion pipelines for future document representations; chats
select compatible query pipelines. Representation-changing edits are checked
against existing project indexes before they can be applied.

The runtime owns execution order, persistence, and provider lifecycle. File
organization does not leak into retrievers, reasoners, assessors, renderers, or
storage interfaces.
