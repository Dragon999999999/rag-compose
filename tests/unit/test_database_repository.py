"""Regression tests for canonical repository transaction ordering."""

from datetime import UTC, datetime

from sqlalchemy.sql.dml import Update

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.infrastructure.database.models import (
    DocumentRow,
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
