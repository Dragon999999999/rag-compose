# LLM Grounding Project

The project provides configurable scientific-document grounding with separate
ingestion and query pipelines. Ingestion parses documents into canonical
evidence and optional indexes; query pipelines retrieve, rerank, assess, and
render grounded answers. A FastAPI adapter and a React frontend expose the same
application services to browser clients and evaluation tools.

## Run

Requires Python 3.13+ and `uv`.

```bash
uv sync --locked
export AQUEDUCT_API_KEY_FILE=/path/to/aqueduct-key.txt
uv run python -m llm_grounding project add-document PROJECT_ID path/to/paper.pdf
```

Alternatively, set `AQUEDUCT_API_KEY`; a nonempty value takes precedence over the
UTF-8 key file. Both are stripped of surrounding whitespace. `.env.example`
documents the variables, and the application loads the repository `.env` without
overriding values already present in the environment.

The canonical pipeline definitions are under [`config/pipelines`](config/pipelines).
Change model and component settings in those pipeline files when needed:

```bash
uv run python -m llm_grounding \
  pipeline validate config/pipelines/query_basic_v1.yaml
```

The installed `llm-grounding` command exposes the same CLI. Default YAML is bundled
in the wheel, so installed usage does not depend on the working directory.

The canonical reusable configuration definitions are under [`config/`](config/),
with standalone models in [`config/models/`](config/models/) and components in
[`config/components/`](config/components/). See
[`docs/architecture.md`](docs/architecture.md) for the resolution rules.

## Persistent development setup

The persistent application keeps PostgreSQL as canonical structured state, MinIO
as canonical PDF storage, and a configured vector backend as a rebuildable index.
The default ingestion pipeline uses Qdrant; pgvector and FAISS are also supported
by the vector-backend registry. Start the local services and apply the canonical
schema before using project or chat commands:

```bash
cp .env.example .env
docker compose up -d
docker compose ps
uv sync --locked --extra embeddings
uv run alembic upgrade head
```

The first command that uses a local model may download its configured weights.
Local models are stored below `MODEL_STORAGE_ROOT` (default
`~/.cache/llm-grounding/models`); the configured runtime may be in-process or a
managed Docker inference container. Query-time model use reuses the local weights
and does not require Hub authentication once they are available. Install the
`embeddings` extra when using in-process Sentence Transformers models.

Then create a project, ingest a PDF, and ask a project-scoped question:

```bash
uv run python -m llm_grounding project create "Quantum Key Distribution"
uv run python -m llm_grounding project add-document PROJECT_ID paper.pdf
uv run python -m llm_grounding chat create PROJECT_ID "Protocol comparison"
uv run python -m llm_grounding chat ask CHAT_ID "What are the main differences?"
uv run python -m llm_grounding project delete PROJECT_ID
```

For grounding investigations, add `--grounding-diagnostics summary` to a chat
query to log retrieved IDs, assessments, and the policy decision to stderr. Use
`--grounding-diagnostics verbose` only for local debugging: it additionally logs
retrieved text, conversation history, and the raw structured LLM response.

MinIO creates `MINIO_BUCKET` lazily and idempotently when a document is first added.
Stop services with `docker compose down`. `docker compose down -v` also deletes all
local PostgreSQL, Qdrant, and MinIO data and is intentionally destructive.

## Architecture

```text
src/llm_grounding/
├── domain/             # Sources, evidence, claims, outputs, and run records
├── parsing/            # Structured PDF blocks with provenance
├── knowledge/          # In-memory canonical evidence
├── retrieval/          # Dense, lexical, sparse, hybrid, and multivector retrieval
├── reasoning/          # Grounded prompt and typed answer generation
├── assessment/         # Grounding assessments
├── decision/           # Grounding decisions
├── rendering/          # Grounded answer rendering
├── orchestration/      # Split-pipeline query execution
├── infrastructure/     # Typed YAML and Aqueduct adapter/credentials
└── application/        # Service and CLI
config/pipelines/             # Ingestion, query, test, and benchmark compositions
benchmarks/rag_benchmark_manifest.yaml
tests/unit/
tests/integration/
```

The normal execution path is:

```text
parse → build knowledge → retrieve → rerank → reason → assess → decide → render
```

The service returns a `RagRun` with the `Answer`, retrieved evidence, candidate
output, assessments, decision, timing, and stage trace. A component failure raises
`RunFailedError`, whose `.run` retains partial results. The CLI uses concise error
messages and nonzero exits. Library clients can inspect the chained cause.

See [`docs/architecture.md`](docs/architecture.md) for the configuration contract,
runtime resolution rules, persistence boundaries, and extension points.

### Configurable retrieval and reranking

Query pipelines can preserve or truncate retrieval order, use dense, PostgreSQL
FTS, BM25, learned sparse, ColBERT MaxSim, full-context, or hybrid retrieval, and
then apply an identity, top-k, local cross-encoder, or generative reranker. Local
cross-encoder settings are kept in a reusable component definition:

```yaml
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

Install the optional dependency with `uv sync --locked --extra reranking`. The
cross-encoder is loaded lazily and cached in `~/.cache/rag-eval/models`, or in
`RAG_EVAL_MODEL_CACHE_DIR` when set. No API key or model server is needed for an
in-process cross-encoder.

Persistent project corpora and the ingestion/query workflow are documented in
[`docs/persistent_corpus.md`](docs/persistent_corpus.md).
The complete CLI command reference is in [`docs/cli.md`](docs/cli.md).

## Development checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

## Browser frontend

The browser application is a separate React/TypeScript/Vite workspace. It uses the
FastAPI adapter as another application interface; it never receives database,
Qdrant, MinIO, embedding, or Aqueduct credentials. With the persistent services
prepared as above, start the API and frontend in separate terminals:

```bash
uv run llm-grounding-api
cd frontend
cp .env.example .env
npm install
npm run dev
```

`VITE_API_BASE_URL` is the only frontend environment setting and is intentionally
non-secret. The API defaults to `http://127.0.0.1:8002`; set
`VITE_API_BASE_URL=http://127.0.0.1:8002/api` for the local frontend. Change
`LLM_GROUNDING_CORS_ORIGINS` when serving the Vite app from a different origin.
The Evidence pane receives canonical, validated claim citations and source metadata
from the API, and PDF binaries are streamed through the API rather than MinIO.

Tests generate small PDFs locally and substitute either the generation client,
PydanticAI's test model, or an in-memory HTTP transport. They block external socket
connections and do not need Aqueduct credentials. The HTTP tests exercise the
actual adapter/provider/SDK stack, including schema serialization and client
closure, without contacting a service.

Follow [`AGENTS.md`](AGENTS.md) for repository conventions. `uv.lock` records the
resolved dependencies. Record the Git commit alongside experimental results;
run metadata includes the package version, input digest, effective architecture
configuration, and prompt/schema identity.

## Limitations

Query results are grounded only in indexed evidence; insufficient evidence is
returned explicitly. Cross-encoder and generative reranker scores reorder
candidates but do not establish claim entailment or answer correctness. The
current PDF parser uses position-based reading order and does not perform OCR.
Models are nondeterministic and remote aliases may change over time.

The project includes a native `/eval/v1` adapter for reproducible retrieval and
grounded-answer evaluation. Its canonical observations are defined in
`src/llm_grounding/observability` and can be persisted and recovered by request
identity. The checked-in [`rag_eval_protocol_schema.json`](rag_eval_protocol_schema.json)
captures the public observation models for tooling. See
[`docs/eval_protocol.md`](docs/eval_protocol.md) for the HTTP contract.
