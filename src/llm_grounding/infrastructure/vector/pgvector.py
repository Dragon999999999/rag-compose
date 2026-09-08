"""PostgreSQL pgvector derived-index adapter with database-level project filtering.

The retrieval_embeddings table is rebuildable from canonical retrieval_units and
their selected project association. This adapter never parses, chunks, or embeds.
It uses cosine distance; pgvector's vector extension must be installed by migration.
"""

from sqlalchemy import literal, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    EvidenceReference,
    RetrievalUnit,
)
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.database.models import (
    RetrievalEmbeddingRow,
    RetrievalUnitMemberRow,
    RetrievalUnitRow,
)


class PgVectorStore:
    """Index/search pgvector rows while retaining generic retrieval results at the boundary."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        """Inject the canonical database session factory."""
        self._sessions = sessions

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[EmbeddingVector, ...],
    ) -> None:
        """Upsert a compatible vector batch in PostgreSQL.

        Raises:
            ValueError: If counts, dimensions, or model identities differ.
        """
        if len(chunks) != len(vectors):
            raise ValueError("Each retrieval chunk needs exactly one vector")
        if not vectors:
            return
        first = vectors[0]
        if any(
            vector.dimension != first.dimension
            or vector.model_identity != first.model_identity
            for vector in vectors
        ):
            raise ValueError("One vector upsert batch must use one embedding space")
        async with self._sessions.begin() as session:
            for chunk, vector in zip(chunks, vectors, strict=True):
                statement = insert(RetrievalEmbeddingRow).values(
                    project_id=project_id,
                    chunk_id=chunk.chunk_id,
                    ingestion_id=chunk.ingestion_id,
                    model_identity=vector.model_identity,
                    dimension=vector.dimension,
                    embedding=list(vector.values),
                )
                statement = statement.on_conflict_do_update(
                    index_elements=["project_id", "chunk_id"],
                    set_={
                        "ingestion_id": chunk.ingestion_id,
                        "model_identity": vector.model_identity,
                        "dimension": vector.dimension,
                        "embedding": list(vector.values),
                    },
                )
                await session.execute(statement)

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Delete rebuildable index rows for one ingestion."""
        from sqlalchemy import delete

        async with self._sessions.begin() as session:
            await session.execute(
                delete(RetrievalEmbeddingRow).where(
                    RetrievalEmbeddingRow.ingestion_id == ingestion_id
                )
            )

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> None:
        """Create a project-scoped copy of existing vectors without re-embedding."""
        async with self._sessions.begin() as session:
            existing = await session.scalar(
                select(RetrievalEmbeddingRow.chunk_id)
                .where(RetrievalEmbeddingRow.ingestion_id == ingestion_id)
                .limit(1)
            )
            if existing is None:
                raise KeyError(f"No indexed vectors for ingestion: {ingestion_id}")
            source = select(
                literal(project_id),
                RetrievalEmbeddingRow.chunk_id,
                RetrievalEmbeddingRow.ingestion_id,
                RetrievalEmbeddingRow.model_identity,
                RetrievalEmbeddingRow.dimension,
                RetrievalEmbeddingRow.embedding,
            ).where(RetrievalEmbeddingRow.ingestion_id == ingestion_id)
            statement = insert(RetrievalEmbeddingRow).from_select(
                [
                    "project_id",
                    "chunk_id",
                    "ingestion_id",
                    "model_identity",
                    "dimension",
                    "embedding",
                ],
                source,
            )
            statement = statement.on_conflict_do_update(
                index_elements=["project_id", "chunk_id"],
                set_={
                    "ingestion_id": statement.excluded.ingestion_id,
                    "model_identity": statement.excluded.model_identity,
                    "dimension": statement.excluded.dimension,
                    "embedding": statement.excluded.embedding,
                },
            )
            await session.execute(statement)

    async def search(
        self,
        project_id: str,
        query: EmbeddingVector,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search cosine similarity with project/document predicates in SQL."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        distance = RetrievalEmbeddingRow.embedding.cosine_distance(
            list(query.values)
        ).label("distance")
        statement = (
            select(RetrievalUnitRow, distance)
            .join(
                RetrievalEmbeddingRow,
                RetrievalEmbeddingRow.chunk_id == RetrievalUnitRow.id,
            )
            .where(
                RetrievalEmbeddingRow.project_id == project_id,
                RetrievalEmbeddingRow.model_identity == query.model_identity,
                RetrievalEmbeddingRow.dimension == query.dimension,
            )
        )
        if document_ids:
            statement = statement.where(RetrievalUnitRow.document_id.in_(document_ids))
        statement = statement.order_by(distance).limit(top_k)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
            output: list[ChunkRetrievalResult] = []
            for rank, (row, value) in enumerate(rows, 1):
                members = (
                    await session.scalars(
                        select(RetrievalUnitMemberRow)
                        .where(RetrievalUnitMemberRow.chunk_id == row.id)
                        .order_by(
                            RetrievalUnitMemberRow.evidence_id,
                            RetrievalUnitMemberRow.start,
                        )
                    )
                ).all()
                chunk = RetrievalUnit(
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
                output.append(
                    ChunkRetrievalResult(chunk, rank, "vector", 1.0 - float(value))
                )
            return tuple(output)
