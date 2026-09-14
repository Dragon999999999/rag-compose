"""Regression tests for canonical repository transaction ordering."""

from datetime import UTC, datetime

from sqlalchemy.sql.dml import Update

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit, SummaryStatus
from llm_grounding.infrastructure.database.models import (
    DocumentRow,
    DocumentSummaryRow,
    RetrievalUnitMemberRow,
    RetrievalUnitRow,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository


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
    """Simulate expiry at commit and require an explicit async refresh."""

    def __init__(self, row: DocumentSummaryRow) -> None:
        """Retain the row that a document-summary mutation updates."""
        self.row = row
        self.committed = False
        self.flushed = False
        self.refreshed = False
        self.refreshed_at = datetime(2025, 1, 2, tzinfo=UTC)

    def begin(self) -> "SummaryTransaction":
        """Open the transaction whose exit expires server-managed fields."""
        return SummaryTransaction(self)

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
        return self.row

    async def flush(self) -> None:
        """Record that pending summary fields were sent to the database."""
        self.flushed = True

    async def refresh(self, row: DocumentSummaryRow) -> None:
        """Reload expired timestamps only after the transaction commits."""
        assert self.committed
        assert row is self.row
        self.refreshed = True
        row.updated_at = self.refreshed_at


class SummaryTransaction:
    """Commit the simulated document-summary mutation on context exit."""

    def __init__(self, session: SummaryMutationSession) -> None:
        """Retain the session whose transaction state must be advanced."""
        self._session = session

    async def __aenter__(self) -> SummaryMutationSession:
        """Expose the active session to the repository."""
        return self._session

    async def __aexit__(self, *args: object) -> None:
        """Model the commit boundary for a successful mutation."""
        self._session.committed = True


class SummarySessionContext:
    """Provide a non-transactional session context around one mutation."""

    def __init__(self, session: SummaryMutationSession) -> None:
        """Retain the session returned by the factory."""
        self._session = session

    async def __aenter__(self) -> SummaryMutationSession:
        """Open the session before its nested transaction begins."""
        return self._session

    async def __aexit__(self, *args: object) -> None:
        """Close the simulated session without suppressing errors."""


class SummarySessionFactory:
    """Return a session context compatible with post-commit refreshes."""

    def __init__(self, row: DocumentSummaryRow) -> None:
        """Expose the mutable summary session for assertions."""
        self.session = SummaryMutationSession(row)

    def __call__(self) -> SummarySessionContext:
        """Create the session context used by a repository operation."""
        return SummarySessionContext(self.session)


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


async def test_fail_document_summary_refreshes_after_commit_before_mapping() -> None:
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
