# Command-line interface

The `llm-grounding` command and `python -m llm_grounding` expose the same
application-service-backed CLI. Project, chat, and persisted-pipeline commands
use PostgreSQL, MinIO, and the selected vector backend. Validation and discovery
commands can run without starting the database-backed application.

## Projects

```bash
llm-grounding project create "Photonics"
llm-grounding project list
llm-grounding project inspect PROJECT_ID
llm-grounding project set-ingestion-pipeline PROJECT_ID PIPELINE_ID
llm-grounding project add-document PROJECT_ID paper.pdf
llm-grounding project add-document PROJECT_ID paper.pdf --ingestion config/pipelines/ingestion_minimal_v1.yaml
llm-grounding project delete PROJECT_ID
```

The ingestion-pipeline selection is project-scoped and applies only to future
documents. Existing document representations are not rewritten. The optional
`--ingestion` path selects one explicit ingestion configuration for the command.

## Chats

```bash
llm-grounding chat create PROJECT_ID "Literature review"
llm-grounding chat list [--project-id PROJECT_ID]
llm-grounding chat inspect CHAT_ID
llm-grounding chat set-pipeline CHAT_ID QUERY_PIPELINE_ID
llm-grounding chat ask CHAT_ID "What method did the paper evaluate?"
llm-grounding chat ask CHAT_ID "Summarize the corpus" --summary
```

`chat ask` also accepts `--pipeline-id` for one-turn pipeline selection and
`--grounding-diagnostics off|summary|verbose`. Summary diagnostics log retrieved
IDs, assessments, and the policy decision to stderr. Verbose diagnostics also
include retrieved text, conversation history, and the raw structured model
response, so use them only with data that is appropriate to log.

Chat pipeline and stage overrides are managed through the validated application
service:

```bash
llm-grounding chat override list CHAT_ID
llm-grounding chat override set CHAT_ID retriever top_k 10
llm-grounding chat override remove CHAT_ID retriever top_k
llm-grounding chat override reset CHAT_ID
llm-grounding chat override reset CHAT_ID retriever
```

Override values are JSON, so strings must be quoted JSON strings such as
`"cross_encoder"`; numbers, booleans, arrays, and objects can be passed as
ordinary JSON values. Overrides are validated before runtime construction and
recorded with the effective configuration used for the run.

## Pipelines

```bash
llm-grounding pipeline list [--type ingestion|query]
llm-grounding pipeline inspect PIPELINE_ID
llm-grounding pipeline validate CONFIG_FILE
llm-grounding pipeline create CONFIG_FILE
llm-grounding pipeline import [DIRECTORY]
llm-grounding pipeline delete PIPELINE_ID
llm-grounding pipeline repair-representation-hashes [--dry-run]
```

`pipeline validate` checks YAML, component IDs, model references, topology, and
compatibility without importing anything. `pipeline create` creates an
API-owned record. `pipeline import` synchronizes YAML-owned files into
PostgreSQL; without a directory it uses the canonical `config/pipelines` tree.
`pipeline delete` deactivates a record while retaining its persisted
configuration.

Use `repair-representation-hashes --dry-run` first to preview verified legacy
hash repairs. The operation updates only canonical hash metadata for unchanged
YAML-owned ingestion pipelines; it does not alter pipeline definitions or IDs.

The checked-in benchmark suite can be imported as a directory; discovery is
recursive across its numbered family folders:

```bash
llm-grounding pipeline import config/pipelines/benchmarks
```

See [`rag_benchmarks.md`](rag_benchmarks.md) for the manifest, validation
coverage, and known representation boundaries. Ingestion pipelines are valid
for projects. Query pipelines are valid for chats. The pipeline service enforces
those scopes and checks index compatibility before persistence or selection.

## Discovery

```bash
llm-grounding component list [--stage STAGE]
llm-grounding component inspect COMPONENT_ID
llm-grounding model list [--type TYPE]
llm-grounding model inspect MODEL_ID
llm-grounding model download MODEL_ID [MODEL_ID ...]
llm-grounding vector-db list
llm-grounding vector-db inspect VECTOR_DB
llm-grounding vector-db test VECTOR_DB
```

Component inspection includes configured values, schema defaults, requirements,
override scopes, and registered runtime implementations. Model inspection and
listing report provider, source, runtime, storage, parameters, and local
availability without starting a model; `model download` prepares local weights.
Vector-database tests are bounded connectivity/configuration checks and do not
create or mutate indexes.

Discovery output redacts credential values and environment mappings; API keys are
never printed.
