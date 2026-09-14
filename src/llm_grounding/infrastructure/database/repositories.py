"""Repository boundary mapping canonical SQLAlchemy rows to domain records.

Repositories never return ORM rows to the application. They retain enough state to
rebuild vectors, validate grounding references, and isolate project corpora.
"""

from datetime import UTC, datetime

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_grounding.domain.corpus import (
    Chat,
    Document,
    DocumentIngestion,
    DocumentSummary,
    EvidenceReference,
    IngestionStatus,
    Message,
    Project,
    ProjectDocument,
    RetrievalUnit,
    SummaryStatus,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.infrastructure.database.models import (
    ChatRow,
    DocumentRow,
    DocumentSummaryRow,
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
    """Persist and load projects, corpora, chats, and RAG-run metadata."""

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
            return Project(row.id, row.name, row.created_at or _now())

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
        self, project_id: str, name: str, chat_name_set: bool = False
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
            row = ChatRow(
                project_id=project_id,
                name=name.strip(),
                chat_name_set=chat_name_set,
            )
            session.add(row)
            await session.flush()
            return Chat(
                row.id,
                row.project_id,
                row.name,
                row.created_at or _now(),
                row.chat_name_set,
            )

    async def get_chat(self, chat_id: str) -> Chat:
        """Load a chat and its project scope."""
        async with self._sessions() as session:
            row = await session.get(ChatRow, chat_id)
            if row is None:
                raise KeyError(f"Unknown chat: {chat_id}")
            has_sent_prompt = await session.scalar(
                select(
                    exists().where(
                        MessageRow.chat_id == chat_id,
                        MessageRow.role == "user",
                    )
                )
            )
            return Chat(
                row.id,
                row.project_id,
                row.name,
                row.created_at,
                row.chat_name_set,
                has_sent_prompt=bool(has_sent_prompt),
            )

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
            return Chat(
                row.id,
                row.project_id,
                row.name,
                row.created_at or _now(),
                row.chat_name_set,
            )

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
            return Chat(
                row.id,
                row.project_id,
                row.name,
                row.created_at or _now(),
                row.chat_name_set,
            )

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
            has_sent_prompt = (
                select(
                    exists().where(
                        MessageRow.chat_id == ChatRow.id,
                        MessageRow.role == "user",
                    )
                )
                .scalar_subquery()
            )
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
                Chat(
                    row.id,
                    row.project_id,
                    row.name,
                    row.created_at,
                    row.chat_name_set,
                    answer_at,
                    bool(sent_prompt),
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

    async def run_metadata(self, run_id: str) -> dict[str, object]:
        """Load persisted run metadata for structured browser provenance views."""
        async with self._sessions() as session:
            row = await session.get(RagRunRow, run_id)
            if row is None:
                raise KeyError(f"Unknown RAG run: {run_id}")
            return dict(row.metadata_json)
