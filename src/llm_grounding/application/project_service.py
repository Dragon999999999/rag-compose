"""Thin project/document use cases built on persistence and ingestion services."""

from pathlib import Path

from llm_grounding.domain.corpus import Document, DocumentIngestion, Project
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.ingestion.service import DocumentIngestionService


class ProjectService:
    """Expose simple corpus lifecycle operations without low-level pipeline choreography."""

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

    async def list_projects(self) -> tuple[Project, ...]:
        """List available project corpora."""
        return await self._repository.list_projects()

    async def delete_project(self, project_id: str) -> None:
        """Remove a project and only its derived Qdrant payload membership."""
        await self._vectors.remove_project(project_id)
        await self._repository.delete_project(project_id)

    async def add_document(
        self, project_id: str, file: Path, config: IngestionConfig
    ) -> tuple[Document, DocumentIngestion]:
        """Attach and index a PDF using defaults or an explicit ingestion configuration."""
        document, _, ingestion = await self._ingestion.add_document(
            project_id, file, config
        )
        return document, ingestion
