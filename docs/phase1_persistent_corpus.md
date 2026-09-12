# Phase 1: persistent corpus and Basic RAG v1

Phase 1 adds a project-scoped corpus without changing the Baseline 0 PDF-summary
command. PostgreSQL is the canonical store for projects, content-addressed document
metadata, project/document associations, immutable ingestion representations,
evidence, chunks, chats, messages, and RAG run metadata. Original PDFs are canonical
MinIO artifacts. Qdrant holds only a derived vector index and can be rebuilt from
ready PostgreSQL retrieval units and their configured embedding representation.

## Setup and migration

Copy the safe development defaults, start the three local services, install the
optional local embedder, and run the migration:

```bash
cp .env.example .env
docker compose up -d
uv sync --extra embeddings
uv run alembic upgrade head
```

The first document embedding operation may download the configured Sentence
Transformers model into the Hugging Face cache. Query embedding is explicitly
cache-only and never requires Hub authentication. Each one-shot CLI invocation
loads cached weights into memory; within a longer-lived application, ingestion and
query services share one lazily initialized model instance.

The host CLI reads `DATABASE_URL`, `QDRANT_URL`, and `MINIO_*` from its environment.
The checked-in `.env.example` provides Compose-compatible local values; load those into
your shell or use another environment/configuration mechanism. `MinioArtifactStore` is
the S3-compatible artifact implementation; the `ArtifactStore` protocol preserves the
option of a local or cloud adapter.

Qdrant points contain vector data plus retrieval-unit and project-filtering identifiers,
not canonical chunk text or evidence provenance. Project filtering happens within the
Qdrant query. Search results are hydrated from PostgreSQL. Removing Qdrant data therefore
does not delete the corpus: a rebuild can re-embed ready retrieval units using their
stored ingestion configuration and re-index the resulting vectors.

## Commands

```bash
llm-grounding project create "Photonics"
llm-grounding project add-document PROJECT_ID paper.pdf
llm-grounding chat create PROJECT_ID "Literature review"
llm-grounding chat ask CHAT_ID "What method did the paper evaluate?"
llm-grounding chat ask CHAT_ID "Summarize the corpus" --summary
```

Add `--grounding-diagnostics summary` to `chat ask` to emit retrieved IDs,
assessments, and the decision to stderr. `--grounding-diagnostics verbose` also
emits document text, chat history, and raw LLM output, so use it only for local
debugging with data that is appropriate to log.

`configs/ingestion/default.yaml` controls parsing, knowledge building, chunking, and
embedding identity. The identity includes parser, knowledge, chunker, embedding model,
revision, dimension, normalization, and index versions. Adding the same PDF with the
same identity reuses the ready representation. Changing a representation-defining value
creates a new ingestion and keeps the old one intact. Attaching a ready ingestion to a
second project reuses its stored derived vectors under the new project association; it
does not parse or embed the PDF again.

Basic RAG v1 is described by `configs/architectures/basic_rag_v1.yaml`. Query retrieval
filters by project inside the vector-store operation before scoring. It can narrow to
document IDs when supplied by a request. Retrieval output carries chunk ID, document,
page span, evidence IDs, and character spans.

The grounded reasoner receives source blocks labelled by actual chunk IDs. Conversation
history is labelled untrusted and is never corpus evidence. A deterministic assessor
rejects model-supplied IDs absent from the retrieved context. The renderer creates source
markers and source listings itself, so a model cannot invent a citation. If no matching
chunks exist or the model marks evidence insufficient, the answer states that clearly.

The implementation deliberately has no lexical, hybrid, or Qdrant backend yet. Those
remain extension points behind retrieval/vector protocols after quality and operational
requirements are defined.
