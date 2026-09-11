"""Repository boundary mapping canonical SQLAlchemy rows to domain records.

Repositories never return ORM rows to the application. They retain enough state to
rebuild vectors, validate grounding references, and isolate project corpora.
"""

from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_grounding.domain.corpus import (
    Chat,
    Document,
    DocumentIngestion,
    EvidenceReference,
    IngestionStatus,
    Message,
    Project,
    ProjectDocument,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.infrastructure.database.models import (
    ChatRow,
    DocumentRow,
    EvidenceRow,
    IngestionRow,
    MessageRow,
    ProjectDocumentRow,
    ProjectRow,
    RagRunRow,
    RetrievalUnitMemberRow,
    RetrievalUnitRow,
)


def _now() -> datetime:
    """Create an explicit UTC fallback for tests without database defaults."""
    return datetime.now(UTC)


class PostgresRepository:
    """Persist and load projects, corpus representations, chats, and RAG run metadata."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        """Inject a factory so each operation owns an isolated async session."""
        self._sessions = sessions

    async def create_project(self, name: str) -> Project:
        """Create a named reusable corpus.

        Raises:
            ValueError: If name is blank.
        """
        if not name.strip():
            raise ValueError("Project name must not be blank")
        async with self._sessions.begin() as session:
            row = ProjectRow(name=name.strip())
            session.add(row)
            await session.flush()
            return Project(row.id, row.name, row.created_at or _now())

    async def list_projects(self) -> tuple[Project, ...]:
        """Return projects in creation order."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ProjectRow).order_by(ProjectRow.created_at)
                )
            ).all()
            return tuple(Project(row.id, row.name, row.created_at) for row in rows)

    async def get_project(self, project_id: str) -> Project:
        """Load one project or raise KeyError."""
        async with self._sessions() as session:
            row = await session.get(ProjectRow, project_id)
            if row is None:
                raise KeyError(f"Unknown project: {project_id}")
            return Project(row.id, row.name, row.created_at)

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

    async def attach_document(
        self, project_id: str, document_id: str, ingestion_id: str | None = None
    ) -> ProjectDocument:
        """Attach reusable content to a project and optionally select its ready ingestion."""
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

    async def create_ingestion(
        self,
        document_id: str,
        config_hash: str,
        config_json: str,
        embedding_hash: str,
        dimension: int,
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
                        text=item.text,
                        metadata_json=dict(item.metadata),
                    )
                )

    async def clear_ingestion_representation(self, ingestion_id: str) -> None:
        """Clear partial canonical chunks/evidence before explicitly retrying an ingestion."""
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

    async def create_chat(self, project_id: str, name: str) -> Chat:
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
            row = ChatRow(project_id=project_id, name=name.strip())
            session.add(row)
            await session.flush()
            return Chat(row.id, row.project_id, row.name, row.created_at or _now())

    async def get_chat(self, chat_id: str) -> Chat:
        """Load a chat and its project scope."""
        async with self._sessions() as session:
            row = await session.get(ChatRow, chat_id)
            if row is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            return Chat(row.id, row.project_id, row.name, row.created_at)

    async def list_chats(self, project_id: str) -> tuple[Chat, ...]:
        """List project conversations in creation order."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ChatRow)
                    .where(ChatRow.project_id == project_id)
                    .order_by(ChatRow.created_at)
                )
            ).all()
            return tuple(
                Chat(row.id, row.project_id, row.name, row.created_at) for row in rows
            )

    async def add_message(
        self, chat_id: str, role: str, content: str, rag_run_id: str | None = None
    ) -> Message:
        """Persist a user or assistant message, separately from source evidence."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Message role must be user, assistant, or system")
        async with self._sessions.begin() as session:
            if await session.get(ChatRow, chat_id) is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            row = MessageRow(
                chat_id=chat_id, role=role, content=content, rag_run_id=rag_run_id
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
    ) -> None:
        """Persist a lightweight RAG run record linked to its active project."""
        async with self._sessions.begin() as session:
            session.add(
                RagRunRow(
                    id=run_id,
                    project_id=project_id,
                    architecture_name=architecture_name,
                    architecture_version=architecture_version,
                    status=status,
                    metadata_json=metadata,
                )
            )
