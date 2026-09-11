"""Separate persistent document ingestion workflow, never invoked by query retrieval."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from llm_grounding.domain.corpus import (
    Document,
    DocumentIngestion,
    IngestionStatus,
    ProjectDocument,
)
from llm_grounding.domain.documents import DocumentSource
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.storage.protocol import ArtifactStore
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.chunking.protocol import Chunker
from llm_grounding.parsing.protocol import DocumentParser


class DocumentIngestionService:
    """Store, parse, chunk, embed, and index one reusable document representation."""

    def __init__(
        self,
        repository: PostgresRepository,
        artifacts: ArtifactStore,
        parser: DocumentParser,
        knowledge_builder: BasicKnowledgeBuilder,
        chunker: Chunker,
        embedder: Embedder,
        vectors: VectorStore,
    ) -> None:
        """Inject independently replaceable persistence, extraction, and indexing dependencies."""
        self._repository = repository
        self._artifacts = artifacts
        self._parser = parser
        self._knowledge_builder = knowledge_builder
        self._chunker = chunker
        self._embedder = embedder
        self._vectors = vectors

    async def add_document(
        self, project_id: str, source: Path, config: IngestionConfig
    ) -> tuple[Document, ProjectDocument, DocumentIngestion]:
        """Attach a PDF to a project and reuse or build the compatible representation.

        Artifact content is deduplicated by SHA-256. A ready ingestion with matching
        full config is selected without reparsing/re-embedding. Different config
        produces a separate ingestion, preserving prior representations.
        """
        if not source.is_file() or source.suffix.lower() != ".pdf":
            raise ValueError(f"Expected an existing PDF: {source}")
        digest = await self._digest(source)
        document = await self._repository.find_document_by_sha256(digest)
        if document is None:
            key = f"pdf/{digest}.pdf"
            await self._artifacts.put_pdf(key, source)
            document = await self._repository.create_document(source.name, digest, key)
        existing = await self._repository.find_ingestion(document.id, config.identity())
        if existing is not None and existing.status == IngestionStatus.READY:
            association = await self._repository.attach_document(
                project_id, document.id, existing.id
            )
            await self._vectors.associate_ingestion(project_id, existing.id)
            return document, association, existing
        ingestion = existing or await self._repository.create_ingestion(
            document.id,
            config.identity(),
            config.model_dump_json(),
            config.embedding.identity(),
            config.embedding.dimension,
        )
        association = await self._repository.attach_document(
            project_id, document.id, ingestion.id
        )
        if existing is None or existing.status != IngestionStatus.READY:
            await self.ingest(project_id, document, ingestion, config)
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
    ) -> None:
        """Run persisted ingestion stages and leave FAILED state on an error.

        Failures retain canonical metadata and a sanitized error type. The caller can
        explicitly re-run this same compatible ingestion later.
        """
        try:
            await self._vectors.delete_ingestion(ingestion.id)
            await self._repository.clear_ingestion_representation(ingestion.id)
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.PARSING
            )
            path = await self._artifacts.path_for(document.artifact_key)
            parsed = await self._parser.parse(DocumentSource(document.id, path))
            knowledge = await self._knowledge_builder.build(parsed)
            evidence = tuple(
                replace(item, evidence_id=f"{ingestion.id}:{item.evidence_id}")
                for item in knowledge.evidence
            )
            await self._repository.store_evidence(ingestion.id, evidence)
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.CHUNKING
            )
            chunks = await self._chunker.chunk(evidence, ingestion.id, document.name)
            await self._repository.store_chunks(chunks)
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.EMBEDDING
            )
            vectors = await self._embedder.embed_documents(
                [chunk.text for chunk in chunks]
            )
            if any(
                vector.model_identity != ingestion.embedding_hash
                or vector.dimension != ingestion.dimension
                for vector in vectors
            ):
                raise ValueError(
                    "Configured embedder is incompatible with ingestion embedding identity"
                )
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.INDEXING
            )
            await self._vectors.upsert(project_id, chunks, vectors)
            # Re-indexing one shared representation must restore every project's
            # payload association without parsing or embedding its chunks again.
            for (
                associated_project_id
            ) in await self._repository.project_ids_for_ingestion(ingestion.id):
                if associated_project_id != project_id:
                    await self._vectors.associate_ingestion(
                        associated_project_id, ingestion.id
                    )
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.READY
            )
        except (OSError, RuntimeError, ValueError) as exc:
            await self._repository.set_ingestion_status(
                ingestion.id, IngestionStatus.FAILED, type(exc).__name__
            )
            raise

    async def _digest(self, source: Path) -> str:
        """Hash file content off the event loop without loading project documents into memory."""
        import asyncio

        def compute() -> str:
            hasher = sha256()
            with source.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    hasher.update(block)
            return hasher.hexdigest()

        return await asyncio.to_thread(compute)
