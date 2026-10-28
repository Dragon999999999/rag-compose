"""Regression tests for canonical repository transaction ordering."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import MissingGreenlet
from sqlalchemy.sql.dml import Update

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.infrastructure.database.models import (
    ChatRow,
    DocumentRow,
    PipelineVersionRow,
    RetrievalUnitMemberRow,
    RetrievalUnitRow,
)
from llm_grounding.infrastructure.database.repositories import (
    PostgresRepository,
    _sent_prompt_exists,
)
from llm_grounding.infrastructure.database.session import create_session_factory


def test_sent_prompt_state_is_correlated_to_each_chat() -> None:
    """An existing prompt in one chat must not mark every chat as used."""
    statement = select(
        ChatRow.id,
        _sent_prompt_exists(ChatRow.id).label("has_sent_prompt"),
    )

    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "FROM messages, chats" not in compiled
    assert "messages.chat_id = chats.id" in compiled


class RecordingSession:
    """Reject child rows added before a simulated parent-row flush."""

    def __init__(self) -> None:
        """Start with no rows and no flush boundary."""
        self.rows: list[object] = []
        self.parents_flushed = False

    def add(self, row: object) -> None:
        """Record rows while enforcing the retrieval-unit foreign-key ordering."""
        if isinstance(row, RetrievalUnitMemberRow) and not self.parents_flushed:
            raise AssertionError("member added before retrieval-unit flush")
        self.rows.append(row)

    async def flush(self) -> None:
        """Mark parent retrieval-unit rows as available to foreign keys."""
        assert self.rows
        assert all(isinstance(row, RetrievalUnitRow) for row in self.rows)
        self.parents_flushed = True


class TransactionContext:
    """Provide the async context-manager shape used by SQLAlchemy sessions."""

    def __init__(self, session: RecordingSession) -> None:
        """Retain the session returned by the transaction context."""
        self._session = session

    async def __aenter__(self) -> RecordingSession:
        """Return the recording session."""
        return self._session

    async def __aexit__(self, *args: object) -> None:
        """Complete the simulated transaction without suppressing errors."""


class SessionFactory:
    """Return one recording transaction for a repository operation."""

    def __init__(self) -> None:
        """Expose the recording session for assertions."""
        self.session = RecordingSession()

    def begin(self) -> TransactionContext:
        """Open the simulated transaction context."""
        return TransactionContext(self.session)


class ExpiringPipelineRow:
    """Simulate an ORM pipeline row whose update timestamp expires on flush."""

    def __init__(self, config, updated_at: datetime) -> None:
        """Initialize fields accessed by the repository's pipeline mapper."""
        self.id = "pipeline-id"
        self.name = config.name
        self.version = config.version
        self.schema_version = config.schema_version
        self.description = config.description
        self.kind = config.kind
        self.config_json = config.model_dump(mode="json")
        self.config_hash = config.config_hash
        self.representation_hash = config.representation_hash()
        self.source_kind = "yaml"
        self.source_path = "config/pipelines/query_basic_v1.yaml"
        self.source_fingerprint = "fingerprint"
        self.active = True
        self.is_default = config.is_default
        self.created_at = updated_at
        self._updated_at = updated_at
        self._timestamp_loaded = False

    @property
    def updated_at(self) -> datetime:
        """Raise like an async ORM access to an expired attribute."""
        if not self._timestamp_loaded:
            raise MissingGreenlet("Pipeline updated_at was not refreshed")
        return self._updated_at

    def mark_refreshed(self) -> None:
        """Mark the generated timestamp as loaded by an awaited refresh."""
        self._timestamp_loaded = True


class ExpiringPipelineSession:
    """Provide the async session operations used by YAML pipeline upsert."""

    def __init__(self, row: ExpiringPipelineRow) -> None:
        """Retain the simulated existing pipeline row."""
        self.row = row
        self.refreshed = False

    async def scalar(self, statement: object) -> ExpiringPipelineRow:
        """Return the existing row selected by the upsert."""
        return self.row

    async def get(
        self,
        model: type[object],
        identity: tuple[str, int],
        *,
        with_for_update: bool,
    ) -> None:
        """Report that no immutable version snapshot needs inserting."""
        assert model is PipelineVersionRow
        assert identity == (self.row.id, self.row.version)
        assert with_for_update
        return None

    async def execute(self, statement: object) -> None:
        """Accept default-pipeline update statements without database access."""

    async def flush(self) -> None:
        """Keep the simulated expired timestamp until explicit refresh."""

    async def refresh(self, row: ExpiringPipelineRow) -> None:
        """Load the generated timestamp through the awaited ORM boundary."""
        assert row is self.row
        self.refreshed = True
        row.mark_refreshed()

    def add(self, row: object) -> None:
        """Accept immutable snapshot inserts for the repository contract."""


class ExpiringPipelineSessionFactory:
    """Open the simulated transaction used by the pipeline refresh test."""

    def __init__(self, session: ExpiringPipelineSession) -> None:
        """Retain the session for refresh assertions."""
        self.session = session

    def begin(self) -> TransactionContext:
        """Open a transaction around the simulated pipeline session."""
        return TransactionContext(self.session)  # type: ignore[arg-type]


async def test_pipeline_upsert_refreshes_expired_timestamp_before_mapping() -> None:
    """Pipeline synchronization maps rows only after generated values are loaded."""
    from pathlib import Path

    from llm_grounding.infrastructure.config.loader import (
        load_model_catalog,
        load_pipeline_config,
    )

    catalog = load_model_catalog()
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    timestamp = datetime.now(UTC)
    session = ExpiringPipelineSession(ExpiringPipelineRow(config, timestamp))
    repository = PostgresRepository(ExpiringPipelineSessionFactory(session))  # type: ignore[arg-type]

    record = await repository.upsert_pipeline_from_yaml(
        config,
        source_path="config/pipelines/query_basic_v1.yaml",
        source_fingerprint="fingerprint",
        representation_hash=config.representation_hash(catalog),
    )

    assert session.refreshed
    assert record.updated_at == timestamp


async def test_store_chunks_flushes_parents_before_foreign_key_members() -> None:
    """Chunk member rows are inserted only after their parent chunks are flushed."""
    sessions = SessionFactory()
    repository = PostgresRepository(sessions)  # type: ignore[arg-type]
    chunk = RetrievalUnit(
        "ingestion:c000000",
        "document",
        "ingestion",
        "paper.pdf",
        "Grounded text",
        0,
        1,
        1,
        (EvidenceReference("evidence", 0, 13),),
    )

    await repository.store_chunks((chunk,))

    assert sessions.session.parents_flushed
    assert [type(row) for row in sessions.session.rows] == [
        RetrievalUnitRow,
        RetrievalUnitMemberRow,
    ]


class DocumentRenameSession:
    """Record a document metadata repair and its chunk update statement."""

    def __init__(self, document: DocumentRow) -> None:
        """Retain the document row returned by the simulated lookup."""
        self.document = document
        self.statement: Update | None = None
        self.flushed = False

    async def get(self, model: type[object], key: str) -> DocumentRow | None:
        """Return the configured document for its canonical identifier."""
        if model is DocumentRow and key == self.document.id:
            return self.document
        return None

    async def execute(self, statement: Update) -> None:
        """Capture the denormalized retrieval-unit name update."""
        self.statement = statement

    async def flush(self) -> None:
        """Record the transaction flush."""
        self.flushed = True


class DocumentRenameTransaction:
    """Provide the repository's async transaction context for a name repair."""

    def __init__(self, session: DocumentRenameSession) -> None:
        """Retain the simulated session."""
        self._session = session

    async def __aenter__(self) -> DocumentRenameSession:
        """Return the simulated session."""
        return self._session

    async def __aexit__(self, *args: object) -> None:
        """Close the simulated transaction without suppressing errors."""


class DocumentRenameSessionFactory:
    """Open one recording transaction for a document-name repair."""

    def __init__(self, document: DocumentRow) -> None:
        """Expose the recording session for assertions."""
        self.session = DocumentRenameSession(document)

    def begin(self) -> DocumentRenameTransaction:
        """Open the simulated transaction context."""
        return DocumentRenameTransaction(self.session)


async def test_update_document_name_synchronizes_chunk_provenance() -> None:
    """A repaired PDF name is also written to all existing retrieval units."""
    row = DocumentRow(
        id="document",
        name="tmpjluizqrw.pdf",
        sha256="a" * 64,
        artifact_key=f"pdf/{'a' * 64}.pdf",
        created_at=datetime.now(UTC),
    )
    sessions = DocumentRenameSessionFactory(row)
    repository = PostgresRepository(sessions)  # type: ignore[arg-type]

    document = await repository.update_document_name(
        row.id, "Original Research Paper.pdf"
    )

    assert document.name == "Original Research Paper.pdf"
    assert row.name == "Original Research Paper.pdf"
    assert sessions.session.flushed
    assert sessions.session.statement is not None
    assert "UPDATE retrieval_units" in str(sessions.session.statement)
    assert sessions.session.statement.compile().params["document_name"] == (
        "Original Research Paper.pdf"
    )


async def test_async_session_factory_disables_commit_expiry() -> None:
    """Inspect the configured session without connecting to PostgreSQL."""
    engine, sessions = create_session_factory(
        "postgresql+asyncpg://test:test@localhost/test"
    )
    try:
        async with sessions() as session:
            assert session.sync_session.expire_on_commit is False
    finally:
        await engine.dispose()
