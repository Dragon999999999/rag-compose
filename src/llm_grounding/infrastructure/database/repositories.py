"""Repository boundary mapping canonical SQLAlchemy rows to domain records.

Repositories never return ORM rows to the application. They retain enough state to
rebuild vectors, validate grounding references, and isolate project corpora.
"""

import json
from datetime import UTC, datetime

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.selectable import Exists

from llm_grounding.domain.corpus import (
    Chat,
    Document,
    DocumentIngestion,
    DocumentSummary,
    EvidenceReference,
    IngestionStatus,
    Message,
    Pipeline,
    Project,
    ProjectDocument,
    RetrievalUnit,
    SummaryStatus,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.infrastructure.config.models import PipelineConfig
from llm_grounding.infrastructure.database.models import (
    ChatRow,
    DocumentRow,
    DocumentSummaryRow,
    EvaluationCorpusChunkRow,
    EvaluationCorpusDocumentRow,
    EvaluationCorpusOperationRow,
    EvaluationCorpusRow,
    EvidenceRow,
    IngestionObservationRow,
    IngestionRow,
    MessageRow,
    PipelineRow,
    PipelineVersionRow,
    ProjectDocumentRow,
    ProjectRow,
    RagRunObservationRow,
    RagRunRow,
    RetrievalUnitMemberRow,
    RetrievalUnitRow,
)
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.observability.ingestion import IngestionObservation
from llm_grounding.observability.models import (
    EvaluationCorpusReference,
    RagRunObservation,
)


def _now() -> datetime:
    """Create an explicit UTC fallback for tests without database defaults."""
    return datetime.now(UTC)


def _sent_prompt_exists(chat_id: str | InstrumentedAttribute[str]) -> Exists:
    """Build an existence check correlated to the selected chat when needed."""
    return exists(
        select(MessageRow.id).where(
            MessageRow.chat_id == chat_id,
            MessageRow.role == "user",
        )
    ).correlate(ChatRow)


class PostgresRepository:
    """Persist projects, pipelines, corpora, chats, and RAG-run metadata.

    Pipelines are the canonical composition records. YAML synchronization is
    intentionally implemented through the repository so API-owned rows cannot
    be overwritten by a later import.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        """Inject a factory so each operation owns an isolated async session."""
        self._sessions = sessions

    async def create_pipeline(
        self,
        config: PipelineConfig | dict[str, object],
        *,
        source_kind: str = "api",
        source_path: str | None = None,
        source_fingerprint: str | None = None,
        representation_hash: str | None = None,
        pipeline_id: str | None = None,
        active: bool = True,
    ) -> Pipeline:
        """Create a pipeline owned by an API or another explicit source.

        Args:
            config: A validated pipeline or a mapping accepted by its schema.
            source_kind: Ownership marker, normally ``api`` or ``yaml``.
            source_path: Import path for YAML-owned records.
            source_fingerprint: SHA-256 of the imported source file.
            representation_hash: Resolved identity of the document index contract.
            pipeline_id: Optional stable identifier used by an import process.
            active: Whether the new pipeline may be selected for execution.

        Returns:
            The persisted pipeline domain record.

        Raises:
            ValueError: If the pipeline name or source marker is invalid.
        """
        pipeline = self._validate_pipeline_config(config)
        if not source_kind.strip():
            raise ValueError("Pipeline source_kind must not be blank")
        async with self._sessions.begin() as session:
            effective_representation_hash = (
                representation_hash or pipeline.representation_hash()
            )
            row = PipelineRow(
                id=pipeline_id,
                name=pipeline.name,
                version=pipeline.version,
                schema_version=pipeline.schema_version,
                description=pipeline.description,
                kind=pipeline.kind,
                config_json=pipeline.model_dump(mode="json"),
                config_hash=pipeline.config_hash,
                representation_hash=effective_representation_hash,
                source_kind=source_kind.strip(),
                source_path=source_path,
                source_fingerprint=source_fingerprint,
                active=active,
                is_default=pipeline.is_default,
            )
            session.add(row)
            if row.is_default:
                await session.execute(
                    update(PipelineRow)
                    .where(
                        PipelineRow.name != row.name,
                        PipelineRow.kind == row.kind,
                    )
                    .values(is_default=False)
                )
            await session.flush()
            await session.refresh(row)
            await self._record_pipeline_version(
                session, row, representation_hash=effective_representation_hash
            )
            return self._pipeline(row)

    async def get_pipeline(self, pipeline_id: str) -> Pipeline:
        """Retrieve one pipeline by stable database identifier."""
        async with self._sessions() as session:
            row = await session.get(PipelineRow, pipeline_id)
            if row is None:
                raise KeyError(f"Unknown pipeline: {pipeline_id}")
            return self._pipeline(row)

    async def get_pipeline_by_name(
        self, name: str, *, kind: str | None = None
    ) -> Pipeline | None:
        """Retrieve the current pipeline record for a stable name."""
        async with self._sessions() as session:
            conditions = [PipelineRow.name == name.strip()]
            if kind is not None:
                conditions.append(PipelineRow.kind == kind)
            row = await session.scalar(select(PipelineRow).where(*conditions))
            return None if row is None else self._pipeline(row)

    async def list_pipelines(
        self, *, active_only: bool = False, kind: str | None = None
    ) -> tuple[Pipeline, ...]:
        """List pipelines deterministically by default status, name, and version."""
        async with self._sessions() as session:
            query = select(PipelineRow)
            if active_only:
                query = query.where(PipelineRow.active.is_(True))
            if kind is not None:
                query = query.where(PipelineRow.kind == kind)
            rows = (
                await session.scalars(
                    query.order_by(
                        PipelineRow.is_default.desc(),
                        PipelineRow.name,
                        PipelineRow.version.desc(),
                    )
                )
            ).all()
            return tuple(self._pipeline(row) for row in rows)

    async def update_pipeline(
        self,
        pipeline_id: str,
        config: PipelineConfig | dict[str, object] | None = None,
        *,
        description: str | None = None,
        active: bool | None = None,
        is_default: bool | None = None,
        representation_hash: str | None = None,
        expected_config_hash: str | None = None,
    ) -> Pipeline:
        """Update a pipeline and mark it API-owned so imports cannot replace it."""
        pipeline = (
            self._validate_pipeline_config(config) if config is not None else None
        )
        async with self._sessions.begin() as session:
            row = await session.get(PipelineRow, pipeline_id, with_for_update=True)
            if row is None:
                raise KeyError(f"Unknown pipeline: {pipeline_id}")
            if (
                expected_config_hash is not None
                and row.config_hash != expected_config_hash
            ):
                raise ValueError(
                    "Pipeline changed while the configuration was being edited"
                )
            if pipeline is not None:
                dependent_count = await session.scalar(
                    select(func.count(ChatRow.id)).where(
                        ChatRow.pipeline_id == pipeline_id
                    )
                )
                if dependent_count:
                    raise ValueError(
                        "Pipeline has associated conversations; explicit migration "
                        "is required"
                    )
            if pipeline is not None:
                row.name = pipeline.name
                row.version = pipeline.version
                row.schema_version = pipeline.schema_version
                row.description = pipeline.description
                row.kind = pipeline.kind
                row.config_json = pipeline.model_dump(mode="json")
                row.config_hash = pipeline.config_hash
                row.representation_hash = (
                    representation_hash or pipeline.representation_hash()
                )
                row.is_default = pipeline.is_default
            elif description is not None:
                row.description = description
            if active is not None:
                row.active = active
            if is_default is not None:
                row.is_default = is_default
            row.source_kind = "api"
            row.source_path = None
            row.source_fingerprint = None
            if row.is_default:
                await session.execute(
                    update(PipelineRow)
                    .where(
                        PipelineRow.id != row.id,
                        PipelineRow.kind == row.kind,
                    )
                    .values(is_default=False)
                )
            await session.flush()
            await session.refresh(row)
            if pipeline is not None:
                await self._record_pipeline_version(
                    session,
                    row,
                    representation_hash=row.representation_hash,
                )
            return self._pipeline(row)

    async def update_pipeline_and_migrate(
        self,
        pipeline_id: str,
        config: PipelineConfig | dict[str, object],
        *,
        representation_hash: str,
        expected_config_hash: str,
    ) -> Pipeline:
        """Update a pipeline and migrate all dependent chats atomically.

        The pipeline row and every chat selection snapshot are locked in one
        transaction. A stale editor hash or representation mismatch aborts the
        transaction before any configuration or conversation is changed.
        """
        pipeline = self._validate_pipeline_config(config)
        async with self._sessions.begin() as session:
            row = await session.get(PipelineRow, pipeline_id, with_for_update=True)
            if row is None:
                raise KeyError(f"Unknown pipeline: {pipeline_id}")
            if row.config_hash != expected_config_hash:
                raise ValueError(
                    "Pipeline changed while the configuration was being edited"
                )
            if row.representation_hash != representation_hash:
                raise ValueError(
                    "The updated representation is incompatible with existing "
                    "conversations"
                )
            chats = (
                await session.scalars(
                    select(ChatRow)
                    .where(ChatRow.pipeline_id == pipeline_id)
                    .with_for_update()
                )
            ).all()
            row.name = pipeline.name
            row.version = pipeline.version
            row.schema_version = pipeline.schema_version
            row.description = pipeline.description
            row.kind = pipeline.kind
            row.config_json = pipeline.model_dump(mode="json")
            row.config_hash = pipeline.config_hash
            row.representation_hash = representation_hash
            row.is_default = pipeline.is_default
            row.source_kind = "api"
            row.source_path = None
            row.source_fingerprint = None
            if row.is_default:
                await session.execute(
                    update(PipelineRow)
                    .where(
                        PipelineRow.id != row.id,
                        PipelineRow.kind == row.kind,
                    )
                    .values(is_default=False)
                )
            for chat in chats:
                chat.pipeline_version = row.version
                chat.pipeline_config_hash = row.config_hash
                if row.kind != "query":
                    chat.pipeline_representation_hash = row.representation_hash
            await session.flush()
            await session.refresh(row)
            await self._record_pipeline_version(
                session, row, representation_hash=representation_hash
            )
            return self._pipeline(row)

    async def count_pipeline_conversations(self, pipeline_id: str) -> int:
        """Count conversations whose current selection references a pipeline."""
        async with self._sessions() as session:
            count = await session.scalar(
                select(func.count(ChatRow.id)).where(ChatRow.pipeline_id == pipeline_id)
            )
            return int(count or 0)

    async def count_pipeline_projects(self, pipeline_id: str) -> int:
        """Count projects whose indexed data selects an ingestion pipeline."""
        async with self._sessions() as session:
            count = await session.scalar(
                select(func.count(ProjectRow.id)).where(
                    ProjectRow.ingestion_pipeline_id == pipeline_id
                )
            )
            return int(count or 0)

    async def _record_pipeline_version(
        self,
        session: AsyncSession,
        row: PipelineRow,
        *,
        representation_hash: str,
    ) -> None:
        """Insert an immutable snapshot and reject version identity reuse."""
        version = await session.get(
            PipelineVersionRow, (row.id, row.version), with_for_update=True
        )
        values = {
            "schema_version": row.schema_version,
            "description": row.description,
            "config_json": dict(row.config_json),
            "config_hash": row.config_hash,
            "representation_hash": representation_hash,
        }
        if version is None:
            session.add(
                PipelineVersionRow(
                    pipeline_id=row.id,
                    version=row.version,
                    **values,
                )
            )
        elif version.config_hash != row.config_hash:
            raise ValueError(
                f"Pipeline {row.id} version {row.version} already contains a "
                f"different configuration (stored hash {version.config_hash}, "
                f"incoming hash {row.config_hash}); increment the pipeline version"
            )

    async def deactivate_pipeline(self, pipeline_id: str) -> Pipeline:
        """Mark a pipeline inactive without deleting its reproducibility record."""
        async with self._sessions.begin() as session:
            row = await session.get(PipelineRow, pipeline_id)
            if row is None:
                raise KeyError(f"Unknown pipeline: {pipeline_id}")
            row.active = False
            await session.flush()
            await session.refresh(row)
            return self._pipeline(row)

    async def upsert_pipeline_from_yaml(
        self,
        config: PipelineConfig,
        *,
        source_path: str,
        source_fingerprint: str,
        representation_hash: str | None = None,
    ) -> Pipeline:
        """Import one YAML pipeline unless its stable name is API-owned.

        Existing YAML-owned rows are updated only when the source fingerprint or
        effective configuration changed. API-owned rows remain untouched.
        """
        async with self._sessions.begin() as session:
            row = await session.scalar(
                select(PipelineRow).where(
                    PipelineRow.name == config.name,
                    PipelineRow.kind == config.kind,
                )
            )
            if row is not None and row.source_kind != "yaml":
                return self._pipeline(row)
            effective_representation_hash = (
                representation_hash or config.representation_hash()
            )
            values = {
                "name": config.name,
                "version": config.version,
                "schema_version": config.schema_version,
                "description": config.description,
                "kind": config.kind,
                "config_json": config.model_dump(mode="json"),
                "config_hash": config.config_hash,
                "representation_hash": effective_representation_hash,
                "source_kind": "yaml",
                "source_path": source_path,
                "source_fingerprint": source_fingerprint,
                "active": True,
                "is_default": config.is_default,
            }
            if row is None:
                row = PipelineRow(**values)
                session.add(row)
            elif (
                row.source_fingerprint != source_fingerprint
                or row.config_hash != config.config_hash
                or row.representation_hash != effective_representation_hash
                or dict(row.config_json) != config.model_dump(mode="json")
            ):
                for key, value in values.items():
                    setattr(row, key, value)
            if config.is_default:
                await session.execute(
                    update(PipelineRow)
                    .where(
                        PipelineRow.name != config.name,
                        PipelineRow.kind == config.kind,
                    )
                    .values(is_default=False)
                )
            await session.flush()
            await session.refresh(row)
            await self._record_pipeline_version(
                session, row, representation_hash=row.representation_hash
            )
            return self._pipeline(row)

    @staticmethod
    def _validate_pipeline_config(
        config: PipelineConfig | dict[str, object],
    ) -> PipelineConfig:
        """Normalize repository inputs at the persistence boundary."""
        if isinstance(config, PipelineConfig):
            return config
        return PipelineConfig.model_validate(config)

    @staticmethod
    def _pipeline(row: PipelineRow) -> Pipeline:
        """Map one ORM pipeline row to a detached domain record."""
        return Pipeline(
            row.id,
            row.name,
            row.version,
            row.schema_version,
            row.description,
            dict(row.config_json),
            row.config_hash,
            row.representation_hash,
            row.source_kind,
            row.source_path,
            row.source_fingerprint,
            row.active,
            row.is_default,
            row.created_at or _now(),
            row.updated_at or row.created_at or _now(),
            row.kind,
        )

    @staticmethod
    def _project(row: ProjectRow) -> Project:
        """Map a project row with its ingestion and indexed-capability snapshot."""
        return Project(
            row.id,
            row.name,
            row.created_at or _now(),
            row.ingestion_pipeline_id,
            row.ingestion_pipeline_version,
            row.ingestion_pipeline_config_hash,
            row.ingestion_representation_hash,
            dict(row.index_capabilities_json or {}),
        )

    @staticmethod
    def _chat(
        row: ChatRow,
        *,
        last_answer_at: datetime | None = None,
        has_sent_prompt: bool = False,
    ) -> Chat:
        """Map one ORM chat row while retaining its pipeline selection snapshot."""
        return Chat(
            row.id,
            row.project_id,
            row.name,
            row.created_at or _now(),
            row.chat_name_set,
            last_answer_at,
            has_sent_prompt,
            row.pipeline_id,
            row.pipeline_version,
            row.pipeline_config_hash,
            row.pipeline_representation_hash,
        )

    async def create_project(
        self,
        name: str,
        *,
        ingestion_pipeline_id: str | None = None,
        ingestion_pipeline_version: int | None = None,
        ingestion_pipeline_config_hash: str | None = None,
        ingestion_representation_hash: str | None = None,
        index_capabilities: dict[str, object] | None = None,
    ) -> Project:
        """Create a named reusable corpus.

        Raises:
            ValueError: If name is blank.
        """
        if not name.strip():
            raise ValueError("Project name must not be blank")
        async with self._sessions.begin() as session:
            pipeline = None
            if ingestion_pipeline_id is not None:
                pipeline = await session.get(PipelineRow, ingestion_pipeline_id)
                if pipeline is None:
                    raise KeyError(
                        f"Unknown ingestion pipeline: {ingestion_pipeline_id}"
                    )
                if pipeline.kind != "ingestion":
                    raise ValueError(
                        "Project ingestion selection requires an ingestion pipeline"
                    )
                if not pipeline.active:
                    raise ValueError(f"Pipeline {ingestion_pipeline_id!r} is inactive")
                if ingestion_pipeline_version is None:
                    ingestion_pipeline_version = pipeline.version
                if ingestion_pipeline_config_hash is None:
                    ingestion_pipeline_config_hash = pipeline.config_hash
                if ingestion_representation_hash is None:
                    ingestion_representation_hash = pipeline.representation_hash
            row = ProjectRow(
                name=name.strip(),
                ingestion_pipeline_id=pipeline.id if pipeline is not None else None,
                ingestion_pipeline_version=ingestion_pipeline_version,
                ingestion_pipeline_config_hash=ingestion_pipeline_config_hash,
                ingestion_representation_hash=ingestion_representation_hash,
                index_capabilities_json=index_capabilities or {},
            )
            session.add(row)
            await session.flush()
            return self._project(row)

    async def list_projects(self) -> tuple[Project, ...]:
        """Return projects in creation order."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ProjectRow).order_by(ProjectRow.created_at)
                )
            ).all()
            return tuple(self._project(row) for row in rows)

    async def get_project(self, project_id: str) -> Project:
        """Load one project or raise KeyError."""
        async with self._sessions() as session:
            row = await session.get(ProjectRow, project_id)
            if row is None:
                raise KeyError(f"Unknown project: {project_id}")
            return self._project(row)

    async def rename_project(self, project_id: str, name: str) -> Project:
        """Rename one project while preserving its canonical corpus identity."""
        if not name.strip():
            raise ValueError("Project name must not be blank")
        async with self._sessions.begin() as session:
            row = await session.get(ProjectRow, project_id)
            if row is None:
                raise KeyError(f"Unknown project: {project_id}")
            row.name = name.strip()
            await session.flush()
            return self._project(row)

    async def set_project_ingestion_pipeline(
        self,
        project_id: str,
        pipeline_id: str,
        *,
        pipeline_version: int,
        pipeline_config_hash: str,
        representation_hash: str,
    ) -> Project:
        """Atomically select an active ingestion pipeline for a project."""
        async with self._sessions.begin() as session:
            row = await session.get(ProjectRow, project_id, with_for_update=True)
            if row is None:
                raise KeyError(f"Unknown project: {project_id}")
            pipeline = await session.get(PipelineRow, pipeline_id)
            if pipeline is None:
                raise KeyError(f"Unknown pipeline: {pipeline_id}")
            if pipeline.kind != "ingestion":
                raise ValueError(
                    "Project ingestion selection requires an ingestion pipeline"
                )
            if not pipeline.active:
                raise ValueError(f"Pipeline {pipeline_id!r} is inactive")
            if (
                pipeline.version != pipeline_version
                or pipeline.config_hash != pipeline_config_hash
                or pipeline.representation_hash != representation_hash
            ):
                raise ValueError(
                    "Ingestion pipeline changed while the project was being updated"
                )
            row.ingestion_pipeline_id = pipeline.id
            row.ingestion_pipeline_version = pipeline.version
            row.ingestion_pipeline_config_hash = pipeline.config_hash
            row.ingestion_representation_hash = pipeline.representation_hash
            await session.flush()
            return self._project(row)

    async def record_project_index(
        self,
        project_id: str,
        *,
        pipeline_id: str,
        pipeline_version: int,
        pipeline_config_hash: str,
        representation_hash: str,
        index_capabilities: dict[str, object],
    ) -> Project:
        """Persist the exact ingestion/index identity used by project documents."""
        async with self._sessions.begin() as session:
            row = await session.get(ProjectRow, project_id, with_for_update=True)
            if row is None:
                raise KeyError(f"Unknown project: {project_id}")
            row.ingestion_pipeline_id = pipeline_id
            row.ingestion_pipeline_version = pipeline_version
            row.ingestion_pipeline_config_hash = pipeline_config_hash
            row.ingestion_representation_hash = representation_hash
            row.index_capabilities_json = index_capabilities
            await session.flush()
            return self._project(row)

    async def delete_project(self, project_id: str) -> None:
        """Delete project associations/chats while preserving reusable documents."""
        async with self._sessions.begin() as session:
            row = await session.get(ProjectRow, project_id)
            if row is None:
                raise KeyError(f"Unknown project: {project_id}")
            await session.delete(row)

    async def find_document_by_sha256(self, digest: str) -> Document | None:
        """Find canonical document content by its immutable SHA-256 identity."""
        async with self._sessions() as session:
            row = await session.scalar(
                select(DocumentRow).where(DocumentRow.sha256 == digest)
            )
            return (
                None
                if row is None
                else Document(
                    row.id, row.name, row.sha256, row.artifact_key, row.created_at
                )
            )

    async def create_document(
        self, name: str, digest: str, artifact_key: str
    ) -> Document:
        """Insert one canonical document; callers deduplicate by digest first."""
        async with self._sessions.begin() as session:
            row = DocumentRow(name=name, sha256=digest, artifact_key=artifact_key)
            session.add(row)
            await session.flush()
            return Document(
                row.id, row.name, row.sha256, row.artifact_key, row.created_at or _now()
            )

    async def update_document_name(self, document_id: str, name: str) -> Document:
        """Repair a document display name and its persisted chunk provenance.

        Canonical content and retrieval identities remain unchanged. Existing chunks
        carry a denormalized document name, so both records are updated in the same
        transaction to keep source lists, citations, and evidence views consistent.

        Args:
            document_id: Canonical document whose display metadata should be repaired.
            name: Original user-facing PDF basename.

        Returns:
            Document metadata with the repaired display name.

        Raises:
            KeyError: If the document does not exist.
            ValueError: If the replacement name is blank.
        """
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Document name must not be blank")
        async with self._sessions.begin() as session:
            row = await session.get(DocumentRow, document_id)
            if row is None:
                raise KeyError(f"Unknown document: {document_id}")
            row.name = normalized_name
            await session.execute(
                update(RetrievalUnitRow)
                .where(RetrievalUnitRow.document_id == document_id)
                .values(document_name=normalized_name)
            )
            await session.flush()
            return Document(
                row.id, row.name, row.sha256, row.artifact_key, row.created_at or _now()
            )

    async def attach_document(
        self, project_id: str, document_id: str, ingestion_id: str | None = None
    ) -> ProjectDocument:
        """Attach reusable content and optionally select its ready ingestion."""
        async with self._sessions.begin() as session:
            if await session.get(ProjectRow, project_id) is None:
                raise KeyError(f"Unknown project: {project_id}")
            if await session.get(DocumentRow, document_id) is None:
                raise KeyError(f"Unknown document: {document_id}")
            row = await session.get(
                ProjectDocumentRow,
                {"project_id": project_id, "document_id": document_id},
            )
            if row is None:
                row = ProjectDocumentRow(
                    project_id=project_id,
                    document_id=document_id,
                    ingestion_id=ingestion_id,
                )
                session.add(row)
            elif ingestion_id is not None:
                row.ingestion_id = ingestion_id
            return ProjectDocument(project_id, document_id, row.ingestion_id)

    async def list_project_documents(
        self, project_id: str
    ) -> tuple[ProjectDocument, ...]:
        """Return project associations and their selected ingestion IDs."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ProjectDocumentRow).where(
                        ProjectDocumentRow.project_id == project_id
                    )
                )
            ).all()
            return tuple(
                ProjectDocument(row.project_id, row.document_id, row.ingestion_id)
                for row in rows
            )

    async def detach_document(self, project_id: str, document_id: str) -> None:
        """Remove one document association without deleting reusable content."""
        async with self._sessions.begin() as session:
            row = await session.get(
                ProjectDocumentRow,
                {"project_id": project_id, "document_id": document_id},
            )
            if row is None:
                raise KeyError(f"Document {document_id} is not in project {project_id}")
            await session.delete(row)

    async def get_document(self, document_id: str) -> Document:
        """Load canonical document metadata or raise KeyError."""
        async with self._sessions() as session:
            row = await session.get(DocumentRow, document_id)
            if row is None:
                raise KeyError(f"Unknown document: {document_id}")
            return Document(
                row.id, row.name, row.sha256, row.artifact_key, row.created_at
            )

    async def find_ingestion(
        self, document_id: str, config_hash: str
    ) -> DocumentIngestion | None:
        """Find a compatible existing representation before expensive work starts."""
        async with self._sessions() as session:
            row = await session.scalar(
                select(IngestionRow).where(
                    IngestionRow.document_id == document_id,
                    IngestionRow.config_hash == config_hash,
                )
            )
            return None if row is None else self._ingestion(row)

    async def list_ingestions(self, document_id: str) -> tuple[DocumentIngestion, ...]:
        """List immutable representations for document-detail views."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(IngestionRow)
                    .where(IngestionRow.document_id == document_id)
                    .order_by(IngestionRow.created_at.desc())
                )
            ).all()
            return tuple(self._ingestion(row) for row in rows)

    async def create_ingestion(
        self,
        document_id: str,
        config_hash: str,
        config_json: str,
        embedding_hash: str | None,
        dimension: int | None,
    ) -> DocumentIngestion:
        """Create a pending immutable representation before parsing and indexing."""
        async with self._sessions.begin() as session:
            row = IngestionRow(
                document_id=document_id,
                config_hash=config_hash,
                config_json=config_json,
                embedding_hash=embedding_hash,
                dimension=dimension,
                status=IngestionStatus.PENDING,
            )
            session.add(row)
            await session.flush()
            return self._ingestion(row)

    async def set_ingestion_status(
        self, ingestion_id: str, status: IngestionStatus, error: str | None = None
    ) -> None:
        """Advance/retry lifecycle state; store only sanitized error text."""
        async with self._sessions.begin() as session:
            row = await session.get(IngestionRow, ingestion_id)
            if row is None:
                raise KeyError(f"Unknown ingestion: {ingestion_id}")
            row.status, row.error = status, error

    async def document_summary(
        self, document_id: str, ingestion_id: str
    ) -> DocumentSummary | None:
        """Load a persisted document overview for one ingestion representation."""
        async with self._sessions() as session:
            row = await session.get(
                DocumentSummaryRow,
                {"document_id": document_id, "ingestion_id": ingestion_id},
            )
            return None if row is None else self._document_summary(row)

    async def begin_document_summary(
        self, document_id: str, ingestion_id: str, page_count: int | None
    ) -> DocumentSummary:
        """Create or restart a summary lifecycle after a successful ingestion."""
        async with self._sessions.begin() as session:
            row = await session.get(
                DocumentSummaryRow,
                {"document_id": document_id, "ingestion_id": ingestion_id},
                with_for_update=True,
            )
            if row is None:
                row = DocumentSummaryRow(
                    document_id=document_id,
                    ingestion_id=ingestion_id,
                    status=SummaryStatus.GENERATING,
                    page_count=page_count,
                )
                session.add(row)
            else:
                row.status = SummaryStatus.GENERATING
                row.rag_run_id = None
                row.error = None
                row.page_count = page_count
            await session.flush()
            # SQL onupdate timestamps can expire during flush even with
            # expire_on_commit=False. Load them before synchronous DTO mapping.
            await session.refresh(row)
            return self._document_summary(row)

    async def complete_document_summary(
        self, document_id: str, ingestion_id: str, rag_run_id: str
    ) -> DocumentSummary:
        """Mark a summary ready after its grounded RAG run has been persisted."""
        async with self._sessions.begin() as session:
            row = await session.get(
                DocumentSummaryRow,
                {"document_id": document_id, "ingestion_id": ingestion_id},
                with_for_update=True,
            )
            if row is None:
                raise KeyError("Document summary does not exist")
            row.status = SummaryStatus.READY
            row.rag_run_id = rag_run_id
            row.error = None
            await session.flush()
            await session.refresh(row)
            return self._document_summary(row)

    async def fail_document_summary(
        self, document_id: str, ingestion_id: str, error: str
    ) -> DocumentSummary:
        """Persist a summary-generation failure reason for the document view."""
        async with self._sessions.begin() as session:
            row = await session.get(
                DocumentSummaryRow,
                {"document_id": document_id, "ingestion_id": ingestion_id},
                with_for_update=True,
            )
            if row is None:
                raise KeyError("Document summary does not exist")
            row.status = SummaryStatus.FAILED
            row.rag_run_id = None
            row.error = error
            await session.flush()
            await session.refresh(row)
            return self._document_summary(row)

    async def page_count_for_ingestion(self, ingestion_id: str) -> int | None:
        """Return the highest parsed PDF page number for summary presentation."""
        async with self._sessions() as session:
            return await session.scalar(
                select(func.max(EvidenceRow.page_number)).where(
                    EvidenceRow.ingestion_id == ingestion_id
                )
            )

    def _ingestion(self, row: IngestionRow) -> DocumentIngestion:
        """Map persistent ingestion state to its domain representation."""
        return DocumentIngestion(
            row.id,
            row.document_id,
            row.config_hash,
            row.config_json,
            row.embedding_hash,
            row.dimension,
            IngestionStatus(row.status),
            row.created_at,
            row.error,
        )

    @staticmethod
    def _document_summary(row: DocumentSummaryRow) -> DocumentSummary:
        """Map persisted document-summary state to its domain representation."""
        return DocumentSummary(
            row.document_id,
            row.ingestion_id,
            SummaryStatus(row.status),
            row.rag_run_id,
            row.error,
            row.page_count,
            row.created_at or _now(),
            row.updated_at or _now(),
        )

    async def store_evidence(
        self, ingestion_id: str, evidence: tuple[EvidenceUnit, ...]
    ) -> None:
        """Persist source evidence before chunks/vectors so the index is rebuildable."""
        async with self._sessions.begin() as session:
            for item in evidence:
                session.add(
                    EvidenceRow(
                        id=item.evidence_id,
                        ingestion_id=ingestion_id,
                        document_id=item.document_id,
                        evidence_type=item.evidence_type,
                        source_uri=item.source_uri,
                        ordering=item.order,
                        page_number=item.page_number,
                        block_index=item.block_index,
                        bounding_box=(
                            list(item.bounding_box)
                            if item.bounding_box is not None
                            else None
                        ),
                        text=item.text,
                        metadata_json=dict(item.metadata),
                    )
                )

    async def clear_ingestion_representation(self, ingestion_id: str) -> None:
        """Clear partial chunks/evidence before explicitly retrying ingestion."""
        async with self._sessions.begin() as session:
            await session.execute(
                delete(RetrievalUnitRow).where(
                    RetrievalUnitRow.ingestion_id == ingestion_id
                )
            )
            await session.execute(
                delete(EvidenceRow).where(EvidenceRow.ingestion_id == ingestion_id)
            )

    async def store_chunks(self, chunks: tuple[RetrievalUnit, ...]) -> None:
        """Persist chunks and each evidence span in a single canonical transaction."""
        async with self._sessions.begin() as session:
            for chunk in chunks:
                session.add(
                    RetrievalUnitRow(
                        id=chunk.chunk_id,
                        ingestion_id=chunk.ingestion_id,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        text=chunk.text,
                        ordering=chunk.order,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        metadata_json=chunk.metadata,
                    )
                )

            # These mappings intentionally use IDs instead of ORM relationships.
            # Flush parents explicitly so SQLAlchemy cannot batch member inserts
            # ahead of retrieval_units and violate the database foreign key.
            await session.flush()

            for chunk in chunks:
                for reference in chunk.evidence_refs:
                    session.add(
                        RetrievalUnitMemberRow(
                            chunk_id=chunk.chunk_id,
                            evidence_id=reference.evidence_id,
                            start=reference.start,
                            end=reference.end,
                        )
                    )

    async def chunks_for_project(
        self, project_id: str, document_ids: tuple[str, ...] = ()
    ) -> tuple[RetrievalUnit, ...]:
        """Load ready, selected project chunks for rebuilding or source validation."""
        async with self._sessions() as session:
            query = (
                select(RetrievalUnitRow)
                .join(
                    ProjectDocumentRow,
                    ProjectDocumentRow.ingestion_id == RetrievalUnitRow.ingestion_id,
                )
                .join(IngestionRow, IngestionRow.id == RetrievalUnitRow.ingestion_id)
                .where(
                    ProjectDocumentRow.project_id == project_id,
                    IngestionRow.status == IngestionStatus.READY,
                )
            )
            if document_ids:
                query = query.where(RetrievalUnitRow.document_id.in_(document_ids))
            rows = (
                await session.scalars(
                    query.order_by(
                        RetrievalUnitRow.document_id, RetrievalUnitRow.ordering
                    )
                )
            ).all()
            chunks: list[RetrievalUnit] = []
            for row in rows:
                chunks.append(await self._chunk(session, row))
            return tuple(chunks)

    async def chunks_by_ids(
        self, chunk_ids: tuple[str, ...]
    ) -> dict[str, RetrievalUnit]:
        """Hydrate canonical chunks returned by a derived vector index.

        The index supplies identifiers and scores only. This method deliberately
        reconstructs text and evidence provenance from PostgreSQL, keeping Qdrant
        disposable and preventing payload data from becoming a second authority.
        """
        if not chunk_ids:
            return {}
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(RetrievalUnitRow).where(RetrievalUnitRow.id.in_(chunk_ids))
                )
            ).all()
            chunks = [await self._chunk(session, row) for row in rows]
            return {chunk.chunk_id: chunk for chunk in chunks}

    async def chunks_for_ingestion(
        self, ingestion_id: str
    ) -> tuple[RetrievalUnit, ...]:
        """Load every chunk and source mapping for one ingestion in order."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(RetrievalUnitRow)
                    .where(RetrievalUnitRow.ingestion_id == ingestion_id)
                    .order_by(RetrievalUnitRow.ordering)
                )
            ).all()
            return tuple(await self._chunk(session, row) for row in rows)

    async def evidence_by_ids(
        self, evidence_ids: tuple[str, ...]
    ) -> dict[str, EvidenceUnit]:
        """Hydrate canonical evidence for browser PDF location rendering."""
        if not evidence_ids:
            return {}
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(EvidenceRow).where(EvidenceRow.id.in_(evidence_ids))
                )
            ).all()
            return {
                row.id: EvidenceUnit(
                    row.id,
                    row.document_id,
                    EvidenceType(row.evidence_type),
                    row.source_uri,
                    row.ordering,
                    row.page_number,
                    row.block_index,
                    (
                        (
                            row.bounding_box[0],
                            row.bounding_box[1],
                            row.bounding_box[2],
                            row.bounding_box[3],
                        )
                        if row.bounding_box is not None and len(row.bounding_box) == 4
                        else None
                    ),
                    row.text,
                    dict(row.metadata_json),
                )
                for row in rows
            }

    async def evidence_for_ingestion(
        self, ingestion_id: str
    ) -> tuple[EvidenceUnit, ...]:
        """Load all canonical evidence units for one ingestion in parser order."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(EvidenceRow)
                    .where(EvidenceRow.ingestion_id == ingestion_id)
                    .order_by(EvidenceRow.ordering)
                )
            ).all()
            return tuple(
                EvidenceUnit(
                    row.id,
                    row.document_id,
                    EvidenceType(row.evidence_type),
                    row.source_uri,
                    row.ordering,
                    row.page_number,
                    row.block_index,
                    tuple(row.bounding_box) if row.bounding_box else None,
                    row.text,
                    dict(row.metadata_json),
                )
                for row in rows
            )

    async def project_ids_for_ingestion(self, ingestion_id: str) -> tuple[str, ...]:
        """Return every project selecting an ingestion for index-membership repair."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ProjectDocumentRow.project_id).where(
                        ProjectDocumentRow.ingestion_id == ingestion_id
                    )
                )
            ).all()
            return tuple(rows)

    async def _chunk(
        self, session: AsyncSession, row: RetrievalUnitRow
    ) -> RetrievalUnit:
        """Load one chunk's canonical evidence member mappings."""
        members = (
            await session.scalars(
                select(RetrievalUnitMemberRow)
                .where(RetrievalUnitMemberRow.chunk_id == row.id)
                .order_by(
                    RetrievalUnitMemberRow.evidence_id, RetrievalUnitMemberRow.start
                )
            )
        ).all()
        return RetrievalUnit(
            row.id,
            row.document_id,
            row.ingestion_id,
            row.document_name,
            row.text,
            row.ordering,
            row.page_start,
            row.page_end,
            tuple(
                EvidenceReference(member.evidence_id, member.start, member.end)
                for member in members
            ),
            dict(row.metadata_json),
        )

    async def create_chat(
        self,
        project_id: str,
        name: str,
        chat_name_set: bool = False,
        *,
        pipeline_id: str | None = None,
        pipeline_version: int | None = None,
        pipeline_config_hash: str | None = None,
        pipeline_representation_hash: str | None = None,
    ) -> Chat:
        """Create a project-scoped conversation.

        Raises:
            KeyError: If project is absent.
            ValueError: If name is blank.
        """
        if not name.strip():
            raise ValueError("Chat name must not be blank")
        async with self._sessions.begin() as session:
            if await session.get(ProjectRow, project_id) is None:
                raise KeyError(f"Unknown project: {project_id}")
            pipeline = None
            if pipeline_id is not None:
                pipeline = await session.get(PipelineRow, pipeline_id)
                if pipeline is None:
                    raise KeyError(f"Unknown pipeline: {pipeline_id}")
                if not pipeline.active:
                    raise ValueError(f"Pipeline {pipeline_id!r} is inactive")
                if pipeline_version is None:
                    pipeline_version = pipeline.version
                if pipeline_config_hash is None:
                    pipeline_config_hash = pipeline.config_hash
                if pipeline_representation_hash is None:
                    pipeline_representation_hash = pipeline.representation_hash
                if (
                    pipeline_version != pipeline.version
                    or pipeline_config_hash != pipeline.config_hash
                    or (
                        pipeline.kind != "query"
                        and pipeline_representation_hash != pipeline.representation_hash
                    )
                ):
                    raise ValueError(
                        "Pipeline selection snapshot does not match the active pipeline"
                    )
            elif any(
                value is not None
                for value in (
                    pipeline_version,
                    pipeline_config_hash,
                    pipeline_representation_hash,
                )
            ):
                raise ValueError("Pipeline selection metadata requires pipeline_id")
            row = ChatRow(
                project_id=project_id,
                name=name.strip(),
                chat_name_set=chat_name_set,
                pipeline_id=pipeline.id if pipeline is not None else None,
                pipeline_version=pipeline_version,
                pipeline_config_hash=pipeline_config_hash,
                pipeline_representation_hash=pipeline_representation_hash,
            )
            session.add(row)
            await session.flush()
            return self._chat(row)

    async def get_chat(self, chat_id: str) -> Chat:
        """Load a chat and its project scope."""
        async with self._sessions() as session:
            row = await session.get(ChatRow, chat_id)
            if row is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            has_sent_prompt = await session.scalar(select(_sent_prompt_exists(chat_id)))
            return self._chat(row, has_sent_prompt=bool(has_sent_prompt))

    async def rename_chat(self, chat_id: str, name: str) -> Chat:
        """Apply a user-selected name and prevent later automatic replacement."""
        if not name.strip():
            raise ValueError("Chat name must not be blank")
        async with self._sessions.begin() as session:
            row = await session.get(ChatRow, chat_id)
            if row is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            row.name = name.strip()
            row.chat_name_set = True
            await session.flush()
            return self._chat(row)

    async def set_generated_chat_name(self, chat_id: str, name: str) -> Chat:
        """Set the first generated name unless a user already finalized the name.

        The flag is checked and updated in one transaction so a manual rename
        completed during answer generation cannot be overwritten afterward.
        """
        if not name.strip():
            raise ValueError("Chat name must not be blank")
        async with self._sessions.begin() as session:
            row = await session.get(ChatRow, chat_id, with_for_update=True)
            if row is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            if not row.chat_name_set:
                row.name = name.strip()
                row.chat_name_set = True
                await session.flush()
            return self._chat(row)

    async def set_chat_pipeline(
        self,
        chat_id: str,
        pipeline_id: str,
        *,
        pipeline_version: int,
        pipeline_config_hash: str,
        pipeline_representation_hash: str,
    ) -> Chat:
        """Atomically select an active pipeline and persist its exact snapshot."""
        async with self._sessions.begin() as session:
            row = await session.get(ChatRow, chat_id, with_for_update=True)
            if row is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            pipeline = await session.get(PipelineRow, pipeline_id)
            if pipeline is None:
                raise KeyError(f"Unknown pipeline: {pipeline_id}")
            if not pipeline.active:
                raise ValueError(f"Pipeline {pipeline_id!r} is inactive")
            if (
                pipeline.version != pipeline_version
                or pipeline.config_hash != pipeline_config_hash
                or (
                    pipeline.kind != "query"
                    and pipeline.representation_hash != pipeline_representation_hash
                )
            ):
                raise ValueError(
                    "Pipeline changed while the conversation was being updated"
                )
            row.pipeline_id = pipeline.id
            row.pipeline_version = pipeline.version
            row.pipeline_config_hash = pipeline.config_hash
            row.pipeline_representation_hash = pipeline.representation_hash
            await session.flush()
            return self._chat(row)

    async def assign_default_pipeline_to_unassigned_chats(self) -> int:
        """Backfill legacy chats after startup imports the active default pipeline."""
        async with self._sessions.begin() as session:
            pipeline = await session.scalar(
                select(PipelineRow)
                .where(
                    PipelineRow.active.is_(True),
                    PipelineRow.is_default.is_(True),
                    PipelineRow.kind == "query",
                )
                .order_by(PipelineRow.name, PipelineRow.version.desc())
                .limit(1)
            )
            if pipeline is None:
                return 0
            result = await session.execute(
                update(ChatRow)
                .where(ChatRow.pipeline_id.is_(None))
                .values(
                    pipeline_id=pipeline.id,
                    pipeline_version=pipeline.version,
                    pipeline_config_hash=pipeline.config_hash,
                    pipeline_representation_hash=pipeline.representation_hash,
                )
            )
            return int(result.rowcount or 0)

    async def assign_default_ingestion_to_unassigned_projects(self) -> int:
        """Assign the active default ingestion pipeline to unassigned projects."""
        async with self._sessions.begin() as session:
            pipeline = await session.scalar(
                select(PipelineRow)
                .where(
                    PipelineRow.active.is_(True),
                    PipelineRow.kind == "ingestion",
                    PipelineRow.is_default.is_(True),
                )
                .order_by(PipelineRow.name, PipelineRow.version.desc())
                .limit(1)
            )
            if pipeline is None:
                return 0
            result = await session.execute(
                update(ProjectRow)
                .where(ProjectRow.ingestion_pipeline_id.is_(None))
                .values(
                    ingestion_pipeline_id=pipeline.id,
                    ingestion_pipeline_version=pipeline.version,
                    ingestion_pipeline_config_hash=pipeline.config_hash,
                    ingestion_representation_hash=pipeline.representation_hash,
                )
            )
            return int(result.rowcount or 0)

    async def backfill_project_index_capabilities(self) -> int:
        """Recover safe index capability snapshots from legacy ready ingestions.

        Projects created before pipeline associations were introduced have no
        capability JSON. A snapshot is reconstructed only when every selected
        ready document uses the same effective ingestion identity; mixed legacy
        representations remain unavailable until explicitly rebuilt.
        """
        updated = 0
        async with self._sessions.begin() as session:
            projects = (await session.scalars(select(ProjectRow))).all()
            for project in projects:
                if project.index_capabilities_json:
                    continue
                rows = (
                    await session.scalars(
                        select(IngestionRow)
                        .join(
                            ProjectDocumentRow,
                            ProjectDocumentRow.ingestion_id == IngestionRow.id,
                        )
                        .where(
                            ProjectDocumentRow.project_id == project.id,
                            IngestionRow.status == IngestionStatus.READY,
                        )
                    )
                ).all()
                if not rows:
                    continue
                configs = [
                    IngestionConfig.model_validate(json.loads(row.config_json))
                    for row in rows
                ]
                identities = {config.identity() for config in configs}
                if len(identities) != 1:
                    continue
                config = configs[0]
                capability_names = ["full_text", "metadata_filtering"]
                if config.embedding is not None and config.vector_backend == "qdrant":
                    capability_names.append("chunk_vectors")
                capabilities = {
                    "capabilities": capability_names,
                    "embedding_identity": (
                        config.embedding.identity()
                        if config.embedding is not None
                        else None
                    ),
                    "dimension": (
                        config.embedding.dimension
                        if config.embedding is not None
                        else None
                    ),
                    "ingestion_pipeline_id": project.ingestion_pipeline_id,
                    "ingestion_pipeline_version": project.ingestion_pipeline_version,
                    "ingestion_pipeline_config_hash": (
                        project.ingestion_pipeline_config_hash
                    ),
                    "representation_hash": project.ingestion_representation_hash,
                }
                project.index_capabilities_json = capabilities
                updated += 1
        return updated

    async def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and its messages without changing project documents."""
        async with self._sessions.begin() as session:
            row = await session.get(ChatRow, chat_id)
            if row is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            await session.delete(row)

    async def list_chats(self, project_id: str) -> tuple[Chat, ...]:
        """List chats with prompt state and the latest completed answer first.

        Chats without an assistant response use their creation time as their
        activity time. `has_sent_prompt` distinguishes an unopened chat from a
        chat whose first answer has not completed yet.
        """
        async with self._sessions() as session:
            last_answer_at = (
                select(func.max(MessageRow.created_at))
                .where(
                    MessageRow.chat_id == ChatRow.id,
                    MessageRow.role == "assistant",
                )
                .scalar_subquery()
            )
            has_sent_prompt = _sent_prompt_exists(ChatRow.id)
            activity_at = func.coalesce(last_answer_at, ChatRow.created_at)
            rows = (
                await session.execute(
                    select(
                        ChatRow,
                        last_answer_at.label("last_answer_at"),
                        has_sent_prompt.label("has_sent_prompt"),
                    )
                    .where(ChatRow.project_id == project_id)
                    .order_by(
                        activity_at.desc(),
                        ChatRow.created_at.desc(),
                    )
                )
            ).all()
            return tuple(
                self._chat(
                    row,
                    last_answer_at=answer_at,
                    has_sent_prompt=bool(sent_prompt),
                )
                for row, answer_at, sent_prompt in rows
            )

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        rag_run_id: str | None = None,
        turn_title: str | None = None,
    ) -> Message:
        """Persist a user or assistant message, separately from source evidence."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Message role must be user, assistant, or system")
        async with self._sessions.begin() as session:
            if await session.get(ChatRow, chat_id) is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            row = MessageRow(
                chat_id=chat_id,
                role=role,
                content=content,
                rag_run_id=rag_run_id,
                turn_title=turn_title,
            )
            session.add(row)
            await session.flush()
            return Message(
                row.id,
                row.chat_id,
                row.role,
                row.content,
                row.created_at or _now(),
                row.rag_run_id,
                row.turn_title,
            )

    async def recent_messages(self, chat_id: str, limit: int) -> tuple[Message, ...]:
        """Return recent conversation in chronological order, not as source evidence."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(MessageRow)
                    .where(MessageRow.chat_id == chat_id)
                    .order_by(MessageRow.created_at.desc())
                    .limit(limit)
                )
            ).all()
            return tuple(
                Message(
                    row.id,
                    row.chat_id,
                    row.role,
                    row.content,
                    row.created_at,
                    row.rag_run_id,
                    row.turn_title,
                )
                for row in reversed(rows)
            )

    async def record_run(
        self,
        run_id: str,
        project_id: str | None,
        architecture_name: str,
        architecture_version: int,
        status: str,
        metadata: dict[str, object],
        *,
        pipeline_id: str | None = None,
        pipeline_version: int | None = None,
        pipeline_config_hash: str | None = None,
        pipeline_representation_hash: str | None = None,
    ) -> None:
        """Persist a run with the immutable pipeline snapshot used for execution."""
        async with self._sessions.begin() as session:
            session.add(
                RagRunRow(
                    id=run_id,
                    project_id=project_id,
                    architecture_name=architecture_name,
                    architecture_version=architecture_version,
                    status=status,
                    pipeline_id=pipeline_id,
                    pipeline_version=pipeline_version,
                    pipeline_config_hash=pipeline_config_hash,
                    pipeline_representation_hash=pipeline_representation_hash,
                    metadata_json=metadata,
                )
            )

    async def run_metadata(self, run_id: str) -> dict[str, object]:
        """Load persisted run metadata for structured browser provenance views."""
        async with self._sessions() as session:
            row = await session.get(RagRunRow, run_id)
            if row is None:
                raise KeyError(f"Unknown RAG run: {run_id}")
            return dict(row.metadata_json)

    async def save_observation(self, observation: RagRunObservation) -> None:
        """Persist or replace one canonical observation by run identity."""
        payload = observation.model_dump(mode="json")
        async with self._sessions.begin() as session:
            row = await session.get(RagRunObservationRow, observation.run_id)
            if row is None:
                session.add(
                    RagRunObservationRow(
                        run_id=observation.run_id,
                        request_id=observation.request_id,
                        benchmark_case_id=observation.benchmark_case_id,
                        experiment_id=observation.experiment_id,
                        status=observation.status,
                        observation_json=payload,
                    )
                )
            else:
                row.request_id = observation.request_id
                row.benchmark_case_id = observation.benchmark_case_id
                row.experiment_id = observation.experiment_id
                row.status = observation.status
                row.observation_json = payload

    async def get_observation(self, run_id: str) -> RagRunObservation | None:
        """Load a canonical observation by its RAG run identifier."""
        async with self._sessions() as session:
            row = await session.get(RagRunObservationRow, run_id)
            return (
                None
                if row is None
                else RagRunObservation.model_validate(row.observation_json)
            )

    async def observation_by_request_id(
        self, request_id: str
    ) -> RagRunObservation | None:
        """Load a canonical observation by idempotency/request identity."""
        async with self._sessions() as session:
            row = await session.scalar(
                select(RagRunObservationRow).where(
                    RagRunObservationRow.request_id == request_id
                )
            )
            return (
                None
                if row is None
                else RagRunObservation.model_validate(row.observation_json)
            )

    async def observations_by_benchmark_case(
        self, benchmark_case_id: str
    ) -> tuple[RagRunObservation, ...]:
        """Load all persisted observations for one benchmark case."""
        return await self._observations_where(
            RagRunObservationRow.benchmark_case_id == benchmark_case_id
        )

    async def observations_by_experiment(
        self, experiment_id: str
    ) -> tuple[RagRunObservation, ...]:
        """Load all persisted observations for one statistical experiment."""
        return await self._observations_where(
            RagRunObservationRow.experiment_id == experiment_id
        )

    async def observations_by_status(
        self, status: str
    ) -> tuple[RagRunObservation, ...]:
        """Load observations with one explicit completed/abstained/failed status."""
        return await self._observations_where(RagRunObservationRow.status == status)

    async def _observations_where(
        self, predicate: object
    ) -> tuple[RagRunObservation, ...]:
        """Hydrate canonical observations from an indexed observation predicate."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(select(RagRunObservationRow).where(predicate))
            ).all()
            return tuple(
                RagRunObservation.model_validate(row.observation_json) for row in rows
            )

    async def save_ingestion_observation(
        self, observation: IngestionObservation
    ) -> None:
        """Persist or replace the inventory associated with an ingestion."""
        payload = observation.model_dump(mode="json")
        async with self._sessions.begin() as session:
            row = await session.get(IngestionObservationRow, observation.ingestion_id)
            if row is None:
                session.add(
                    IngestionObservationRow(
                        ingestion_id=observation.ingestion_id,
                        observation_json=payload,
                    )
                )
            else:
                row.observation_json = payload

    async def get_ingestion_observation(
        self, ingestion_id: str
    ) -> IngestionObservation | None:
        """Load a persisted ingestion inventory by ingestion identity."""
        async with self._sessions() as session:
            row = await session.get(IngestionObservationRow, ingestion_id)
            return (
                None
                if row is None
                else IngestionObservation.model_validate(row.observation_json)
            )

    async def save_evaluation_corpus(self, corpus: EvaluationCorpusReference) -> None:
        """Persist an isolated evaluation corpus without project-table writes."""
        async with self._sessions.begin() as session:
            row = await session.get(EvaluationCorpusRow, corpus.corpus_id)
            values = {
                "parent_corpus_id": corpus.parent_corpus_id,
                "corpus_hash": corpus.corpus_hash,
                "version": corpus.version,
                "documents_json": corpus.documents,
                "chunks_json": corpus.chunks,
                "perturbation_metadata": corpus.perturbation_metadata,
            }
            if row is None:
                session.add(EvaluationCorpusRow(id=corpus.corpus_id, **values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)
            await session.execute(
                delete(EvaluationCorpusDocumentRow).where(
                    EvaluationCorpusDocumentRow.corpus_id == corpus.corpus_id
                )
            )
            await session.execute(
                delete(EvaluationCorpusChunkRow).where(
                    EvaluationCorpusChunkRow.corpus_id == corpus.corpus_id
                )
            )
            session.add_all(
                EvaluationCorpusDocumentRow(
                    id=str(item.get("id", f"document-{index}")),
                    corpus_id=corpus.corpus_id,
                    document_json=item,
                )
                for index, item in enumerate(corpus.documents)
            )
            session.add_all(
                EvaluationCorpusChunkRow(
                    id=str(item.get("id", f"chunk-{index}")),
                    corpus_id=corpus.corpus_id,
                    chunk_json=item,
                )
                for index, item in enumerate(corpus.chunks)
            )

    async def get_evaluation_corpus(
        self, corpus_id: str
    ) -> EvaluationCorpusReference | None:
        """Load an isolated evaluation corpus inventory."""
        async with self._sessions() as session:
            row = await session.get(EvaluationCorpusRow, corpus_id)
            if row is None:
                return None
            return EvaluationCorpusReference(
                corpus_id=row.id,
                parent_corpus_id=row.parent_corpus_id,
                corpus_hash=row.corpus_hash,
                version=row.version,
                documents=list(row.documents_json),
                chunks=list(row.chunks_json),
                perturbation_metadata=dict(row.perturbation_metadata),
            )

    async def delete_evaluation_corpus(self, corpus_id: str) -> bool:
        """Delete only isolated evaluation tables, never production projects."""
        async with self._sessions.begin() as session:
            row = await session.get(EvaluationCorpusRow, corpus_id)
            if row is None:
                return False
            await session.delete(row)
            return True

    async def save_evaluation_operation(
        self,
        operation_id: str,
        corpus_id: str,
        status: str,
        payload: dict[str, object],
    ) -> None:
        """Persist one isolated corpus lifecycle operation."""
        async with self._sessions.begin() as session:
            session.add(
                EvaluationCorpusOperationRow(
                    id=operation_id,
                    corpus_id=corpus_id,
                    status=status,
                    operation_json=payload,
                )
            )

    async def get_evaluation_operation(
        self, operation_id: str
    ) -> dict[str, object] | None:
        """Load a corpus-operation status payload."""
        async with self._sessions() as session:
            row = await session.get(EvaluationCorpusOperationRow, operation_id)
            return None if row is None else dict(row.operation_json)
