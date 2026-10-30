# Persistent corpus and split pipelines

The application uses separate ingestion and query pipelines. PostgreSQL is the
canonical store for projects, content-addressed document metadata,
project/document associations, immutable ingestion representations, evidence,
chunks, chats, messages, pipeline records, and RAG run observations. Original
PDFs are canonical MinIO artifacts. A configured vector backend stores derived
indexes that can be rebuilt from ready PostgreSQL retrieval units and their
embedding representation.

## Setup and storage

Copy the safe development defaults, start the local services, install optional
in-process embedding dependencies when needed, and run the migration:

```bash
cp .env.example .env
docker compose up -d
uv sync --locked --extra embeddings
uv run alembic upgrade head
```

The application and Alembic load the repository `.env` without overriding
explicit environment values. The host CLI reads `DATABASE_URL`, vector-backend
settings, and `MINIO_*` from that environment. `MinioArtifactStore` is the
S3-compatible artifact implementation; the `ArtifactStore` protocol preserves
the option of another local or cloud adapter.

The default ingestion pipeline uses Qdrant. pgvector and FAISS are also available
through the vector-backend registry. Learned sparse and multivector indexes use
Qdrant because their storage contracts require sparse or token-level vectors.
Qdrant and the other indexes contain derived vector data plus retrieval-unit and
project-filtering identifiers, not canonical chunk text or evidence provenance.
Search results are hydrated from PostgreSQL. Removing an index therefore does not
delete the corpus; a rebuild can re-embed ready retrieval units using their stored
ingestion configuration.

`docker compose down -v` deletes the local PostgreSQL, Qdrant, and MinIO volumes.
Treat it as a destructive reset of the development corpus.

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
emits document text, chat history, and raw structured model output, so use it
only for local debugging with data that is appropriate to log.

`config/pipelines/ingestion_default_v1.yaml` controls parsing, knowledge
building, chunking, embedding, and vector-backend identity. The identity includes
parser, knowledge builder, chunker, embedding model, representation, revision,
dimension, normalization, and index versions. Adding the same PDF with the same
identity reuses the ready representation. Changing a representation-defining
value creates a new ingestion and keeps the old one intact. The minimal ingestion
pipeline omits knowledge building, embeddings, and vector indexing when a
PostgreSQL-only corpus is sufficient.

The current query example is
`config/pipelines/query_basic_v1.yaml`. It uses summary-aware dense retrieval,
identity reranking, source-reference assessment, the basic grounding decision
policy, and grounded-answer rendering. Other registered query components support
PostgreSQL FTS, BM25, learned sparse, ColBERT MaxSim, full-context, hybrid
retrieval, multiple rerankers, additional deterministic assessors, and bounded
branch/alternative execution. See [`pipelines.md`](pipelines.md) for composition
and compatibility rules.

Query retrieval filters by project inside the selected store before scoring. It
can narrow to document IDs when supplied by a request. Retrieval output carries
chunk ID, document, page span, evidence IDs, and character spans.

The grounded reasoner receives source blocks labelled by request-local `chunk_N`
aliases. Conversation history is labelled untrusted and is never corpus
evidence. A deterministic resolver maps aliases back to canonical chunk IDs
before assessment. The renderer creates source markers and source listings
itself, so a model cannot invent a citation. If no matching chunks exist or the
model marks evidence insufficient, the answer states that clearly.

User-facing citations use stable display labels such as `Smith_15`: the PDF
filename stem plus the one-based canonical chunk order. The label is consistent
across chats that use the same project representation, while canonical chunk IDs
remain internal. Every rendered answer lists retrieved chunks; cited chunks are
marked as used and other candidates remain visible as context. The browser
Evidence pane follows the active prompt-answer turn and can open the source PDF
through the API.

New browser chats persist the placeholder name `New chat` with an unset name flag.
The first grounded-answer request also asks the model for a concise conversation
name, which is applied only if the flag is still unset when generation finishes.
Any manual rename finalizes the flag. Per-turn navigation titles remain separate
from the chat name and continue to be generated for prompt-answer pairs.

See [`cli.md`](cli.md) for project inspection, pipeline selection, chat overrides,
and configuration discovery. See [`eval_protocol.md`](eval_protocol.md) for the
isolated evaluation-corpus API and persisted observation contract.
