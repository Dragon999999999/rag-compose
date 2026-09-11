"""Regression tests for canonical repository transaction ordering."""

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.infrastructure.database.models import (
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
