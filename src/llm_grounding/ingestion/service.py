"""Separate persistent document ingestion workflow, never invoked by query retrieval."""

import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import TypeVar, cast

from llm_grounding.domain.corpus import (
    Document,
    DocumentIngestion,
    IngestionStatus,
    ProjectDocument,
    RetrievalUnit,
)
from llm_grounding.domain.documents import DocumentSource, ParsedDocument
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.model_management.manager import LocalModelManager
from llm_grounding.infrastructure.storage.protocol import ArtifactStore
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import EmbeddingConfig, IngestionConfig
from llm_grounding.knowledge.chunking.protocol import Chunker
from llm_grounding.knowledge.models import KnowledgeBase
from llm_grounding.knowledge.normalization import normalize_parsed_document
from llm_grounding.knowledge.protocol import KnowledgeBuilder
from llm_grounding.observability.ingestion import build_ingestion_observation
from llm_grounding.observability.models import StageType
from llm_grounding.observability.trace import TraceRecorder
from llm_grounding.parsing.protocol import DocumentParser

StageResult = TypeVar("StageResult")

_TEMPORARY_PDF_NAME = re.compile(r"^tmp[a-z0-9]{6,32}\.pdf$", re.IGNORECASE)
_UUID_PDF_NAME = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\.pdf$",
    re.IGNORECASE,
)
_DIGEST_PDF_NAME = re.compile(r"^[0-9a-f]{64}\.pdf$", re.IGNORECASE)


def _is_generated_pdf_name(name: str) -> bool:
    """Return whether a PDF basename is an infrastructure-generated identifier."""
    return any(
        pattern.fullmatch(name)
        for pattern in (_TEMPORARY_PDF_NAME, _UUID_PDF_NAME, _DIGEST_PDF_NAME)
    )


class DocumentIngestionService:
    """Store, parse, chunk, embed, and index one reusable document representation."""

    def __init__(
        self,
        repository: PostgresRepository,
        artifacts: ArtifactStore,
        parser: DocumentParser,
        knowledge_builder: KnowledgeBuilder | None,
        chunker: Chunker,
        embedder: Embedder | None,
        vectors: VectorStore | None,
        *,
        embedder_factory: Callable[[EmbeddingConfig], Embedder] | None = None,
        local_model_manager: LocalModelManager | None = None,
    ) -> None:
        """Inject persistence, extraction, and indexing dependencies.

        The supplied embedder is retained as the compatibility/default instance
        for existing callers. Additional ingestion configurations are materialized
        lazily and cached by their complete embedding identity.
        """
        self._repository = repository
        self._artifacts = artifacts
        self._parser = parser
        self._knowledge_builder = knowledge_builder
        self._chunker = chunker
        self._embedder_factory = (
            (lambda config: cast(Embedder, build_embedder(config, local_model_manager)))
            if local_model_manager is not None
            else embedder_factory
            or cast(Callable[[EmbeddingConfig], Embedder], build_embedder)
        )
        self._embedders = (
            {embedder.model_identity: embedder} if embedder is not None else {}
        )
        self._vectors = vectors

    def _vectors_for(self, config: IngestionConfig) -> VectorStore | None:
        """Resolve the backend selected by the effective project pipeline."""
        if config.embedding is None or self._vectors is None:
            return None
        resolver = getattr(self._vectors, "for_ingestion_config", None)
        return resolver(config) if resolver is not None else self._vectors

    def _embedder_for(self, config: IngestionConfig) -> Embedder:
        """Return the cached provider for one exact embedding configuration."""
        if config.embedding is None:
            raise ValueError("No embedder is available for PostgreSQL-only ingestion")
        identity = config.embedding.identity()
        embedder = self._embedders.get(identity)
        if embedder is None:
            embedder = self._embedder_factory(config.embedding)
            self._embedders[identity] = embedder
        return embedder

    async def add_document(
        self, project_id: str, source: Path, config: IngestionConfig
    ) -> tuple[Document, ProjectDocument, DocumentIngestion]:
        """Attach a PDF to a project and reuse or build the compatible representation.

        Artifact content is deduplicated by SHA-256. A ready ingestion with matching
        full config is reused only while its derived vectors still exist. Missing
        vectors and incomplete lifecycle states rebuild the existing ingestion;
        different configuration produces a separate representation. Reuse also
        replaces legacy generated filenames with the uploaded PDF's real basename.
        """
        if not source.is_file() or source.suffix.lower() != ".pdf":
            raise ValueError(f"Expected an existing PDF: {source}")
        digest = await self._digest(source)
        document = await self._repository.find_document_by_sha256(digest)
        if document is None:
            key = f"pdf/{digest}.pdf"
            await self._artifacts.put_pdf(key, source)
            document = await self._repository.create_document(source.name, digest, key)
        elif _is_generated_pdf_name(document.name) and not _is_generated_pdf_name(
            source.name
        ):
            # Early uploads persisted parser/upload temporary basenames. Repair the
            # display metadata when the same content is supplied with its real name.
            document = await self._repository.update_document_name(
                document.id, source.name
            )
        existing = await self._repository.find_ingestion(document.id, config.identity())
        ingestion = existing or await self._repository.create_ingestion(
            document.id,
            config.identity(),
            config.model_dump_json(),
            config.embedding.vector_identity()
            if config.embedding is not None
            else None,
            config.embedding.dimension if config.embedding is not None else None,
        )
        association = await self._repository.attach_document(
            project_id, document.id, ingestion.id
        )
        vector_store = self._vectors_for(config)
        if existing is not None and existing.status == IngestionStatus.READY:
            if config.embedding is None:
                return document, association, existing
            if vector_store is None:
                raise RuntimeError(
                    "A vector backend is required for dense ingestion but is "
                    "unavailable"
                )
            if await vector_store.associate_ingestion(project_id, existing.id):
                return document, association, existing
        await self.ingest(
            project_id,
            document,
            ingestion,
            config,
            embedder=(
                self._embedder_for(config) if config.embedding is not None else None
            ),
        )
        refreshed = await self._repository.find_ingestion(
            document.id, config.identity()
        )
        if refreshed is None:
            raise RuntimeError("Ingestion disappeared after processing")
        return document, association, refreshed

    async def ingest(
        self,
        project_id: str,
        document: Document,
        ingestion: DocumentIngestion,
        config: IngestionConfig,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        """Run persisted ingestion stages and leave FAILED state on an error.

        Failures retain canonical metadata and a sanitized error type. The caller can
        explicitly re-run this same compatible ingestion later.
        """
        active_embedder = (
            embedder or self._embedder_for(config)
            if config.embedding is not None
            else None
        )
        vector_store = self._vectors_for(config)
        traces = TraceRecorder()
        total_span = traces.start("ingestion", stage_type=StageType.INGESTION)
        started = monotonic()
        evidence: tuple[EvidenceUnit, ...] = ()
        chunks: tuple[RetrievalUnit, ...] = ()
        try:
            if config.embedding is not None:
                if vector_store is None or active_embedder is None:
                    raise RuntimeError(
                        "Dense ingestion requires both an embedder and vector backend"
                    )
                await vector_store.delete_ingestion(ingestion.id)
            await self._repository.clear_ingestion_representation(ingestion.id)
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.PARSING
            )
            path = await self._artifacts.path_for(document.artifact_key)
            parsed = await self._traced(
                traces,
                "parse",
                self._parser.parse(DocumentSource(document.id, path)),
            )
            knowledge = await self._knowledge(parsed, traces)
            evidence = tuple(
                replace(item, evidence_id=f"{ingestion.id}:{item.evidence_id}")
                for item in knowledge.evidence
            )
            await self._repository.store_evidence(ingestion.id, evidence)
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.CHUNKING
            )
            chunks = await self._traced(
                traces,
                "chunk",
                self._chunker.chunk(evidence, ingestion.id, document.name),
            )
            await self._repository.store_chunks(chunks)
            if config.embedding is not None:
                assert vector_store is not None and active_embedder is not None
                await self._repository.set_ingestion_status(
                    ingestion.id, IngestionStatus.EMBEDDING
                )
                vectors = await self._traced(
                    traces,
                    "embed",
                    active_embedder.embed_documents([chunk.text for chunk in chunks]),
                )
                if any(
                    vector.model_identity != ingestion.embedding_hash
                    or vector.dimension != ingestion.dimension
                    for vector in vectors
                ):
                    raise ValueError(
                        "Configured embedder is incompatible with ingestion "
                        "embedding identity"
                    )
                await self._repository.set_ingestion_status(
                    ingestion.id, IngestionStatus.INDEXING
                )
                await self._traced(
                    traces, "index", vector_store.upsert(project_id, chunks, vectors)
                )
                # Re-indexing one shared representation must restore every project's
                # payload association without parsing or embedding its chunks again.
                for (
                    associated_project_id
                ) in await self._repository.project_ids_for_ingestion(ingestion.id):
                    if associated_project_id != project_id:
                        await vector_store.associate_ingestion(
                            associated_project_id, ingestion.id
                        )
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.READY
            )
            traces.end(total_span, status="completed", output_count=len(chunks))
            await self._save_observation(
                build_ingestion_observation(
                    document,
                    ingestion,
                    config,
                    evidence,
                    chunks,
                    status="ready",
                    trace=traces.spans,
                    latency_ms=(monotonic() - started) * 1000,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            traces.end(total_span, status="failed", error=type(exc).__name__)
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.FAILED, type(exc).__name__
            )
            await self._save_observation(
                build_ingestion_observation(
                    document,
                    ingestion,
                    config,
                    evidence,
                    chunks,
                    status="failed",
                    trace=traces.spans,
                    latency_ms=(monotonic() - started) * 1000,
                    error=type(exc).__name__,
                )
            )
            raise

    async def _knowledge(
        self, parsed: ParsedDocument, traces: TraceRecorder
    ) -> KnowledgeBase:
        """Run the optional builder or normalize parser output directly."""
        if self._knowledge_builder is None:
            return normalize_parsed_document(parsed)
        return await self._traced(
            traces, "extract", self._knowledge_builder.build(parsed)
        )

    async def _traced(
        self,
        recorder: TraceRecorder,
        name: str,
        operation: Awaitable[StageResult],
    ) -> StageResult:
        """Await one ingestion operation and preserve its failure span."""
        span_id = recorder.start(name, stage_type=StageType.INGESTION)
        try:
            result = await operation
        except Exception as exc:
            recorder.end(span_id, status="failed", error=type(exc).__name__)
            raise
        recorder.end(span_id, status="completed")
        return result

    async def _save_observation(self, observation: object) -> None:
        """Persist diagnostics when the repository implements the observability API."""
        saver = getattr(self._repository, "save_ingestion_observation", None)
        if saver is not None:
            await saver(observation)

    async def _digest(self, source: Path) -> str:
        """Hash file content off the event loop without loading all data."""
        import asyncio

        def compute() -> str:
            hasher = sha256()
            with source.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    hasher.update(block)
            return hasher.hexdigest()

        return await asyncio.to_thread(compute)
