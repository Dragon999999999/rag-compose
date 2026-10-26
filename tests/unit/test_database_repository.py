"""Regression tests for canonical repository transaction ordering."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import MissingGreenlet
from sqlalchemy.orm import Session, make_transient_to_detached
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.sql.dml import Update

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit, SummaryStatus
from llm_grounding.infrastructure.database.models import (
    ChatRow,
    DocumentRow,
    DocumentSummaryRow,
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


class SummaryMutationSession:
    """Expire real ORM attributes and reject implicit database loads offline."""

    def __init__(self, row: DocumentSummaryRow, *, exists: bool = True) -> None:
        """Retain the row that a document-summary mutation updates."""
        self.row = row
        self.exists = exists
        self.committed = False
        self.flushed = False
        self.refreshed = False
        self.refreshed_at = datetime(2025, 1, 2, tzinfo=UTC)
        self.orm = Session(expire_on_commit=False)
        if exists:
            make_transient_to_detached(row)
            self.orm.add(row)
        event.listen(self.orm, "do_orm_execute", self._reject_implicit_load)

    @staticmethod
    def _reject_implicit_load(*args: object) -> None:
        """Simulate the async driver's error when mapping loads expired state."""
        raise MissingGreenlet("Implicit ORM load outside an awaited session method")

    def begin(self) -> "SummaryTransaction":
        """Open the transaction whose exit detaches the ORM row."""
        return SummaryTransaction(self)

    def add(self, row: DocumentSummaryRow) -> None:
        """Attach a new row with simulated database timestamp defaults."""
        row.created_at = self.refreshed_at
        row.updated_at = self.refreshed_at
        row.rag_run_id = None
        row.error = None
        make_transient_to_detached(row)
        self.orm.add(row)
        self.row = row
        self.exists = True

    async def get(
        self,
        model: type[object],
        identity: dict[str, str],
        *,
        with_for_update: bool,
    ) -> DocumentSummaryRow | None:
        """Return the configured summary row for its canonical composite key."""
        assert model is DocumentSummaryRow
        assert identity == {
            "document_id": self.row.document_id,
            "ingestion_id": self.row.ingestion_id,
        }
        assert with_for_update
        return self.row if self.exists else None

    async def flush(self) -> None:
        """Simulate SQL onupdate expiry even with expire_on_commit disabled."""
        self.flushed = True
        self.orm.expire(self.row, ["updated_at"])

    async def refresh(self, row: DocumentSummaryRow) -> None:
        """Supply the database timestamp while the transaction is still active."""
        assert self.flushed
        assert not self.committed
        assert row is self.row
        self.refreshed = True
        set_committed_value(row, "updated_at", self.refreshed_at)


class SummaryTransaction:
    """Commit the simulated document-summary mutation on context exit."""

    def __init__(self, session: SummaryMutationSession) -> None:
        """Retain the session whose transaction state must be advanced."""
        self._session = session

    async def __aenter__(self) -> SummaryMutationSession:
        """Expose the active session to the repository."""
        return self._session

    async def __aexit__(self, *args: object) -> None:
        """Expire and detach the row to catch mapping after transaction exit."""
        self._session.committed = True
        self._session.orm.expire_all()
        self._session.orm.close()


class SummarySessionFactory:
    """Return an isolated transaction for each summary mutation test."""

    def __init__(self, row: DocumentSummaryRow, *, exists: bool = True) -> None:
        """Expose the mutable summary session for assertions."""
        self.session = SummaryMutationSession(row, exists=exists)

    def begin(self) -> SummaryTransaction:
        """Create the transaction context used by a repository operation."""
        return SummaryTransaction(self.session)


def _document_summary_row() -> DocumentSummaryRow:
    """Build one generating row with stable timestamps for mutation tests."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return DocumentSummaryRow(
        document_id="document",
        ingestion_id="ingestion",
        status=SummaryStatus.GENERATING,
        rag_run_id=None,
        error=None,
        page_count=24,
        created_at=now,
        updated_at=now,
    )


async def test_completed_summary_refreshes_before_mapping() -> None:
    """A completed summary maps loaded timestamps without lazy async ORM I/O."""
    sessions = SummarySessionFactory(_document_summary_row())
    repository = PostgresRepository(sessions)  # type: ignore[arg-type]

    summary = await repository.complete_document_summary("document", "ingestion", "run")

    assert sessions.session.flushed
    assert sessions.session.refreshed
    assert summary.status == SummaryStatus.READY
    assert summary.rag_run_id == "run"
    assert summary.updated_at == sessions.session.refreshed_at


async def test_failed_summary_refreshes_before_mapping() -> None:
    """A failed summary maps loaded timestamps without lazy async ORM I/O."""
    sessions = SummarySessionFactory(_document_summary_row())
    repository = PostgresRepository(sessions)  # type: ignore[arg-type]

    summary = await repository.fail_document_summary(
        "document", "ingestion", "network unavailable"
    )

    assert sessions.session.flushed
    assert sessions.session.refreshed
    assert summary.status == SummaryStatus.FAILED
    assert summary.error == "network unavailable"
    assert summary.updated_at == sessions.session.refreshed_at


@pytest.mark.parametrize("status", [SummaryStatus.READY, SummaryStatus.FAILED])
async def test_regenerate_summary_refreshes_generating_state(
    status: SummaryStatus,
) -> None:
    """Restarting a persisted summary clears prior results and loads its timestamp."""
    row = _document_summary_row()
    row.status = status
    row.rag_run_id = "old-run"
    row.error = "previous failure"
    sessions = SummarySessionFactory(row)
    repository = PostgresRepository(sessions)  # type: ignore[arg-type]

    summary = await repository.begin_document_summary("document", "ingestion", 30)

    assert summary.status == SummaryStatus.GENERATING
    assert summary.rag_run_id is None
    assert summary.error is None
    assert summary.page_count == 30
    assert summary.updated_at == sessions.session.refreshed_at
    assert sessions.session.committed


async def test_summary_mapping_without_refresh_reproduces_missing_greenlet() -> None:
    """Verify the regression harness rejects the original implicit-load behavior."""
    row = _document_summary_row()
    sessions = SummarySessionFactory(row)
    async with sessions.begin() as session:
        await session.flush()
        with pytest.raises(MissingGreenlet, match="Implicit ORM load"):
            PostgresRepository._document_summary(row)


async def test_new_summary_refreshes_before_mapping() -> None:
    """Creating a missing summary returns loaded database-generated timestamps."""
    sessions = SummarySessionFactory(_document_summary_row(), exists=False)
    repository = PostgresRepository(sessions)  # type: ignore[arg-type]

    summary = await repository.begin_document_summary("document", "ingestion", 24)

    assert summary.status == SummaryStatus.GENERATING
    assert summary.created_at == sessions.session.refreshed_at
    assert summary.updated_at == sessions.session.refreshed_at
    assert summary.page_count == 24


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
