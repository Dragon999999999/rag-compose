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

## Architecture

```text
src/llm_grounding/
├── domain/             # Sources, evidence, claims, outputs, and run records
├── parsing/            # Structured PDF blocks with provenance
├── knowledge/          # In-memory canonical evidence
├── retrieval/          # Whole-document selection and identity reranking
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

## Development checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

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

Later work can introduce evidence selection, additional parsers, substantive
assessors, decision policies, and scientific evaluation datasets behind the existing
protocols. Embeddings, databases, real reranking, claim verification, and multimodal
analysis are deliberately outside Baseline 0.
