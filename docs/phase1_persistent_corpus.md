# Phase 1: persistent corpus and Basic RAG v1

Phase 1 adds a project-scoped corpus without changing the Baseline 0 PDF-summary
command. PostgreSQL is the canonical store for projects, content-addressed document
metadata, project/document associations, immutable ingestion representations,
evidence, chunks, chats, messages, and RAG run metadata. PDFs are copied to artifact
storage. The `retrieval_embeddings` pgvector table is derived state and can be rebuilt
from canonical chunks and the configured embedding model.

## Setup and migration

Set `DATABASE_URL` to an asyncpg PostgreSQL URL. The database user needs permission to
create the `vector` extension. Install the optional local embedder, make the configured
Sentence Transformers model available in the local cache, and run the migration:

```bash
uv sync --extra embeddings
export DATABASE_URL=postgresql+asyncpg://user:password@localhost/llm_grounding
uv run alembic upgrade head
```

`ARTIFACT_STORAGE_PATH` defaults to `.artifacts`. It is intended for local development;
the `ArtifactStore` protocol is the boundary for an S3-compatible implementation.

## Commands

```bash
llm-grounding project create "Photonics"
llm-grounding project add-document PROJECT_ID paper.pdf
llm-grounding chat create PROJECT_ID "Literature review"
llm-grounding chat ask CHAT_ID "What method did the paper evaluate?"
llm-grounding chat ask CHAT_ID "Summarize the corpus" --summary
```

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
