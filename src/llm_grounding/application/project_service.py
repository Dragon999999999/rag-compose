"""Thin project/document use cases built on persistence and ingestion services."""

from pathlib import Path

from llm_grounding.application.document_summary_service import DocumentSummaryService
from llm_grounding.application.pipeline_compatibility import (
    IndexCapabilities,
    validate_query_compatibility,
)
from llm_grounding.application.pipeline_service import PipelineService
from llm_grounding.domain.corpus import Document, DocumentIngestion, Project
from llm_grounding.infrastructure.config.models import PipelineConfig
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
        summaries: DocumentSummaryService | None = None,
        pipeline_service: PipelineService | None = None,
    ) -> None:
        """Inject canonical persistence and the separate ingestion workflow."""
        self._repository, self._ingestion, self._vectors = (
            repository,
            ingestion,
            vectors,
        )
        self._summaries = summaries
        self._pipeline_service = pipeline_service

    async def create_project(
        self, name: str, ingestion_pipeline_id: str | None = None
    ) -> Project:
        """Create an empty reusable project corpus."""
        if self._pipeline_service is None:
            return await self._repository.create_project(name)
        pipeline = await self._pipeline_service.select_ingestion(ingestion_pipeline_id)
        await self._pipeline_service.resolve_ingestion(pipeline.id)
        return await self._repository.create_project(
            name,
            ingestion_pipeline_id=pipeline.id,
            ingestion_pipeline_version=pipeline.version,
            ingestion_pipeline_config_hash=pipeline.config_hash,
            ingestion_representation_hash=pipeline.representation_hash,
        )

    async def set_ingestion_pipeline(
        self, project_id: str, ingestion_pipeline_id: str
    ) -> Project:
        """Select an ingestion pipeline without silently invalidating an index."""
        if self._pipeline_service is None:
            raise ValueError("Pipeline management is unavailable")
        project = await self._repository.get_project(project_id)
        pipeline = await self._pipeline_service.select_ingestion(ingestion_pipeline_id)
        await self._pipeline_service.resolve_ingestion(pipeline.id)
        if (
            project.index_capabilities
            and project.ingestion_representation_hash is not None
            and project.ingestion_representation_hash != pipeline.representation_hash
        ):
            raise ValueError(
                "The project already has an index for a different ingestion "
                "representation; re-indexing is required before switching."
            )
        return await self._repository.set_project_ingestion_pipeline(
            project_id,
            pipeline.id,
            pipeline_version=pipeline.version,
            pipeline_config_hash=pipeline.config_hash,
            representation_hash=pipeline.representation_hash,
        )

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
        """Attach/index a PDF, then persist its grounded document overview."""
        selected_pipeline = None
        if self._pipeline_service is not None:
            project = await self._repository.get_project(project_id)
            selected_pipeline = await self._pipeline_service.select_ingestion(
                project.ingestion_pipeline_id
            )
            config = await self._pipeline_service.resolve_ingestion(
                selected_pipeline.id
            )
        document, _, ingestion = await self._ingestion.add_document(
            project_id, file, config
        )
        if selected_pipeline is not None:
            capabilities = IndexCapabilities.from_ingestion(
                config,
                ingestion_pipeline_id=selected_pipeline.id,
                ingestion_pipeline_version=selected_pipeline.version,
                ingestion_pipeline_config_hash=selected_pipeline.config_hash,
                representation_hash=selected_pipeline.representation_hash,
            )
            await self._repository.record_project_index(
                project_id,
                pipeline_id=selected_pipeline.id,
                pipeline_version=selected_pipeline.version,
                pipeline_config_hash=selected_pipeline.config_hash,
                representation_hash=selected_pipeline.representation_hash,
                index_capabilities=capabilities.as_dict(),
            )
        if self._summaries is not None:
            await self._summaries.schedule_summary(project_id, document, ingestion)
        return document, ingestion

    async def compatible_query_pipelines(
        self, project_id: str
    ) -> list[dict[str, object]]:
        """Return query pipeline compatibility details for one project index."""
        if self._pipeline_service is None:
            raise ValueError("Pipeline management is unavailable")
        project = await self._repository.get_project(project_id)
        index = IndexCapabilities.from_mapping(project.index_capabilities)
        records = await self._pipeline_service.list(active_only=True, kind="query")
        result: list[dict[str, object]] = []
        for record in records:
            config = PipelineConfig.model_validate(record.config)
            compatibility = validate_query_compatibility(
                config, index, self._pipeline_service.registry
            )
            result.append(
                {
                    "pipeline": self._pipeline_service.summaries((record,))[0],
                    **compatibility.as_dict(),
                }
            )
        return result
