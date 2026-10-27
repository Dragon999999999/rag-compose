# Configuration architecture

The framework separates configuration identity from implementation and provider
runtime details:

```text
Pipeline composition
        │ references component IDs
        ▼
Component registry
        │ resolves implementation and operational settings
        ▼
Model registry
        │ resolves model IDs and credential references
        ▼
Provider or local runtime
```

## Standalone definitions

Canonical definitions live under `config/`:

```text
config/
├── models/                         # one provider/local model per YAML file
├── components/<category>/          # reusable stage configurations
└── pipelines/                      # composition-only pipeline examples
```

Model filenames are discovery details only. The YAML `id` is the authoritative
identity and must be unique across all model files. Two IDs may use the same
provider model name while differing in endpoint, credentials, timeout, retries,
or other connection settings. Provider-specific schemas are selected by the
model `backend`, currently `openai_compatible`, `sentence_transformers`, and
`fake`.

Credentials are represented only by environment-variable names such as
`AQUEDUCT_API_KEY` and `AQUEDUCT_API_KEY_FILE`; secret values never belong in
YAML. Models and local weights are initialized lazily by their existing adapters.
Local embedding models may declare query and document prefixes in `parameters`;
the shared provider applies them at both ingestion and query time so the two
representations remain compatible.

Components have their own stable IDs, implementation `type`, optional `model_id`,
and operational settings. A component owns settings such as chunk sizes,
retrieval limits, reranking batch size, device, and prompt selection. Pipeline
files contain only component IDs and composition metadata; they do not contain
provider endpoints, model names, or implementation tuning.

## Resolution and compatibility

`ConfigurationRegistry` loads standalone files, rejects duplicate IDs, checks
category ownership, resolves model references, and converts the result into the
established runtime schemas. `ComponentRegistry` then validates that each
implementation is registered before any provider is constructed.

`load_ingestion_config`, `load_pipeline_config`, and `load_model_catalog` accept
the standalone format. Pipeline YAML uses `schema_version: 2` and must declare
either `kind: ingestion` with representation stages or `kind: query` with query
stages. There is no combined pipeline format.

The runtime still owns execution order and persistence. Configuration file
organization does not leak into retrievers, reasoners, assessors, renderers, or
storage interfaces.
