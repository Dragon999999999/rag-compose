"""Thin project/document use cases built on persistence and ingestion services."""

from pathlib import Path

from llm_grounding.domain.corpus import Document, DocumentIngestion, Project
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.ingestion.service import DocumentIngestionService


class ProjectService:
    """Expose corpus lifecycle operations without pipeline choreography."""

    def __init__(
        self,
        repository: PostgresRepository,
        ingestion: DocumentIngestionService,
        vectors: VectorStore,
    ) -> None:
        """Inject canonical persistence and the separate ingestion workflow."""
        self._repository, self._ingestion, self._vectors = (
            repository,
            ingestion,
            vectors,
        )

    async def create_project(self, name: str) -> Project:
        """Create an empty reusable project corpus."""
        return await self._repository.create_project(name)

    async def rename_project(self, project_id: str, name: str) -> Project:
        """Rename a project through the canonical persistence boundary."""
        return await self._repository.rename_project(project_id, name)

    async def list_projects(self) -> tuple[Project, ...]:
        """List available project corpora."""
        return await self._repository.list_projects()

    async def delete_project(self, project_id: str) -> None:
        """Remove a project and only its derived Qdrant payload membership."""
        await self._vectors.remove_project(project_id)
        await self._repository.delete_project(project_id)

    async def list_documents(
        self, project_id: str
    ) -> tuple[tuple[Document, DocumentIngestion | None], ...]:
        """List project sources with the selected ingestion lifecycle state."""
        associations = await self._repository.list_project_documents(project_id)
        documents: list[tuple[Document, DocumentIngestion | None]] = []
        for association in associations:
            document = await self._repository.get_document(association.document_id)
            ingestion = None
            if association.ingestion_id is not None:
                # The configuration hash is not available at this boundary, so the
                # selected representation is loaded through its document list below.
                candidates = await self._repository.list_ingestions(document.id)
                ingestion = next(
                    (
                        item
                        for item in candidates
                        if item.id == association.ingestion_id
                    ),
                    None,
                )
            documents.append((document, ingestion))
        return tuple(documents)

    async def remove_document(self, project_id: str, document_id: str) -> None:
        """Detach a source from a project without deleting canonical content."""
        await self._repository.detach_document(project_id, document_id)

    async def add_document(
        self, project_id: str, file: Path, config: IngestionConfig
    ) -> tuple[Document, DocumentIngestion]:
        """Attach and index a PDF with the supplied ingestion configuration."""
        document, _, ingestion = await self._ingestion.add_document(
            project_id, file, config
        )
        return document, ingestion
