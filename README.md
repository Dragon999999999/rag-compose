# LLM Grounding Project

Baseline 0 is a **Basic Summary Extraction Baseline** for scientific PDFs. It
extracts all available text, asks Aqueduct/Qwen for 5–8 keywords and a short
summary, and returns the result through the application's configured pipeline.
It performs no semantic search, claim verification, or grounding assessment.

## Run

Requires Python 3.13+ and `uv`.

```bash
uv sync --locked
export AQUEDUCT_API_KEY_FILE=/path/to/aqueduct-key.txt
uv run python -m llm_grounding path/to/paper.pdf
```

Alternatively, set `AQUEDUCT_API_KEY`; a nonempty value takes precedence over the
UTF-8 key file. Both are stripped of surrounding whitespace. `.env.example`
documents the variables; the application does not automatically load `.env`.

The default architecture is defined in
[`configs/architectures/baseline_v0.yaml`](configs/architectures/baseline_v0.yaml).
It uses `qwen-3.5-397b` at `https://aqueduct.ai.datalab.tuwien.ac.at/v1`.
Change the model settings in an explicit architecture file when needed:

```bash
uv run python -m llm_grounding \
  --architecture configs/architectures/baseline_v0.yaml \
  path/to/paper.pdf
```

The installed `llm-grounding` command exposes the same CLI. Default YAML is bundled
in the wheel, so installed usage does not depend on the working directory.
Output contains numbered keywords, the summary, and an extraction-success line.
Success means the pipeline completed; it does not establish factual correctness.

## Persistent Basic RAG v1 development

Basic RAG keeps PostgreSQL as canonical structured state, MinIO as canonical PDF
storage, and Qdrant as a rebuildable vector index. Start the services and apply the
canonical schema before using project/chat commands:

```bash
cp .env.example .env
docker compose up -d
docker compose ps
set -a; source .env; set +a
uv sync --extra embeddings
uv run alembic upgrade head
```

The first command that embeds a document may download the configured Sentence
Transformers model into the Hugging Face cache. Query embedding is cache-only and
does not contact the Hub. Each one-shot CLI process still loads the cached weights
into memory; a long-running application would reuse the same in-memory embedder.

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
├── retrieval/          # Retrieval plus top-k and local cross-encoder reranking
├── reasoning/          # Versioned prompt and typed summary extraction
├── assessment/         # No-op measurements
├── decision/           # Unconditional baseline acceptance
├── rendering/          # Deterministic general Answer
├── evaluation/         # Offline structural checks and test-suite runner
├── orchestration/      # Registry, composition, context, and linear execution
├── infrastructure/     # Typed YAML and Aqueduct adapter/credentials
└── application/        # Service and CLI
configs/architectures/baseline_v0.yaml
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

See [`docs/architecture.md`](docs/architecture.md) for contracts, a complete file
map, prompt semantics, migration notes, and extension boundaries.

### Optional local reranking

Query pipelines can keep the retrieval order with `type: identity`, truncate it
with `provider: top_k`, or score candidates locally with
`provider: cross_encoder`:

```yaml
reranker:
  provider: cross_encoder
  retrieval_top_k: 20
  rerank_top_k: 5
  model_name: cross-encoder/ms-marco-MiniLM-L6-v2
  device: cpu
  batch_size: 16
```

Install the optional dependency with `uv sync --extra reranking`. The model is
loaded only on the first reranking request and cached in
`~/.cache/rag-eval/models`, or in `RAG_EVAL_MODEL_CACHE_DIR` when set. In Docker,
mount that directory as a persistent volume if the application container should
reuse downloaded weights across restarts. No API key or model server is needed.

Persistent project corpora and Basic RAG v1 are documented in
[`docs/phase1_persistent_corpus.md`](docs/phase1_persistent_corpus.md). Baseline 0
continues to use its original single-PDF CLI syntax.

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
non-secret. The API defaults to `http://127.0.0.1:8000/api`; change
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

## Limitations and next phase

Reading order is a positional heuristic, and OCR is not implemented. Whole-document
input can exceed the model context window; this baseline does not silently truncate
or chunk it. Keyword count (5–8) and summary length (20–500 characters) are schema
constraints; 1–3 words per keyword and 2–3 summary sentences are prompt instructions.
The model is nondeterministic and its remote alias can change.

Later work can introduce additional parsers, substantive assessors, decision
policies, and scientific evaluation datasets behind the existing protocols.
Cross-encoder scores reorder candidates but do not establish claim entailment or
correctness.
# LLM Grounding

The project includes a native `/eval/v1` adapter for reproducible retrieval and
grounded-answer evaluation. Its canonical observations are defined in
`src/llm_grounding/observability` and can be persisted and recovered by request
identity. See [docs/eval_protocol.md](docs/eval_protocol.md) for the protocol.
