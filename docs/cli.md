# Command-line interface

The `llm-grounding` command and `python -m llm_grounding` expose the same
application-service-backed CLI. Persistent project and chat commands use
PostgreSQL, MinIO, and the configured vector backend; discovery commands read
the validated standalone registries without calling the HTTP API.

## Projects

```bash
llm-grounding project create "Photonics"
llm-grounding project list
llm-grounding project inspect PROJECT_ID
llm-grounding project set-ingestion-pipeline PROJECT_ID PIPELINE_ID
llm-grounding project add-document PROJECT_ID paper.pdf
llm-grounding project delete PROJECT_ID
```

The ingestion-pipeline selection is project-scoped and applies only to future
documents. Existing document representations are not rewritten.

## Chats

```bash
llm-grounding chat create PROJECT_ID "Literature review"
llm-grounding chat list
llm-grounding chat inspect CHAT_ID
llm-grounding chat set-pipeline CHAT_ID QUERY_PIPELINE_ID
llm-grounding chat ask CHAT_ID "What method did the paper evaluate?"
```

Chat overrides are validated by the same application service used by the API:

```bash
llm-grounding chat override list CHAT_ID
llm-grounding chat override set CHAT_ID retriever top_k 10
llm-grounding chat override remove CHAT_ID retriever top_k
llm-grounding chat override reset CHAT_ID
llm-grounding chat override reset CHAT_ID retriever
```

Override values are JSON, so strings must be quoted JSON strings such as
`"cross_encoder"`, while numbers, booleans, arrays, and objects can be passed
as ordinary JSON values.

## Pipelines

```bash
llm-grounding pipeline list [--type ingestion|query]
llm-grounding pipeline inspect PIPELINE_ID
llm-grounding pipeline validate CONFIG_FILE
llm-grounding pipeline create CONFIG_FILE
llm-grounding pipeline import [DIRECTORY]
llm-grounding pipeline delete PIPELINE_ID
```

Ingestion pipelines are valid for projects. Query pipelines are valid for
chats. The pipeline service enforces those scopes and validates configuration
compatibility before persistence.

## Discovery

```bash
llm-grounding component list [--stage STAGE]
llm-grounding component inspect COMPONENT_ID
llm-grounding model list [--type TYPE]
llm-grounding model inspect MODEL_ID
llm-grounding vector-db list
llm-grounding vector-db inspect VECTOR_DB
llm-grounding vector-db test VECTOR_DB
```

Component inspection includes configured values, requirements, registered
runtime implementations, and family JSON Schema including documented defaults.
Model inspection includes provider, source, runtime, storage, parameters, and
non-starting local availability status. Vector-database tests are bounded
connectivity/configuration checks and do not create or mutate indexes.

Discovery output redacts credential values and environment mappings; API keys
are never printed.
