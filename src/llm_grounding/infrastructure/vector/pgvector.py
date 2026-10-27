"""PostgreSQL pgvector adapter for the canonical vector-store protocol.

The adapter keeps chunk text and provenance in the existing repository and stores
only derived vectors plus filter metadata in PostgreSQL. The extension and table
are created lazily on first use; approximate indexes are created only after a
vector dimension is known.
"""

from collections.abc import Iterable
from hashlib import sha256

from sqlalchemy import text

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.config.models import VectorBackendConfig
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.observability.embedding import EmbeddingRecord

_TABLE = "llm_grounding_pgvector"


def _literal_vector(values: Iterable[float]) -> str:
    """Serialize values in the format accepted by pgvector's text input."""
    return "[" + ",".join(f"{value:.17g}" for value in values) + "]"


class PgVectorStore:
    """Persist project-scoped vectors in the application's PostgreSQL server."""

    def __init__(
        self, config: VectorBackendConfig, repository: PostgresRepository
    ) -> None:
        """Configure a lazy pgvector store without opening database resources."""
        self._config = config
        self._repository = repository
        self._schema_ready = False
        self._indexes: set[tuple[int, str]] = set()

    async def _ensure_schema(self) -> None:
        """Create the derived vector table when this backend is first used."""
        if self._schema_ready:
            return
        async with self._repository.session_factory.begin() as session:
            try:
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await session.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {_TABLE} (
                            project_id VARCHAR(255) NOT NULL,
                            chunk_id VARCHAR(255) NOT NULL,
                            ingestion_id VARCHAR(36) NOT NULL,
                            document_id VARCHAR(36) NOT NULL,
                            model_identity VARCHAR(255) NOT NULL,
                            dimension INTEGER NOT NULL,
                            embedding vector NOT NULL,
                            PRIMARY KEY (project_id, chunk_id, model_identity)
                        )
                        """
                    )
                )
                await session.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS ix_{_TABLE}_project "
                        f"ON {_TABLE} (project_id, model_identity, dimension)"
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "pgvector is unavailable. Run the database migrations and "
                    "ensure the PostgreSQL service uses a pgvector-enabled image."
                ) from exc
        self._schema_ready = True

    async def _ensure_index(self, vector: EmbeddingVector) -> None:
        """Create one configured approximate index for a known dimension."""
        if self._config.index_type == "flat":
            return
        key = (vector.dimension, self._config.similarity_metric)
        if key in self._indexes:
            return
        digest = sha256(
            f"{vector.dimension}:{self._config.similarity_metric}:"
            f"{self._config.index_type}".encode()
        ).hexdigest()[:16]
        index_name = f"ix_{_TABLE}_{self._config.index_type}_{digest}"
        operator = {
            "cosine": "vector_cosine_ops",
            "dot": "vector_ip_ops",
            "euclidean": "vector_l2_ops",
        }[self._config.similarity_metric]
        method = {"hnsw": "hnsw", "ivf": "ivfflat"}[self._config.index_type]
        options = (
            f" WITH (m = {self._config.hnsw_m}, "
            f"ef_construction = {self._config.hnsw_ef_construction})"
            if self._config.index_type == "hnsw"
            else f" WITH (lists = {self._config.nlist})"
        )
        statement = text(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON {_TABLE} USING {method} "
            f"((embedding::vector({vector.dimension})) {operator}){options} "
            f"WHERE dimension = {vector.dimension}"
        )
        async with self._repository.session_factory.begin() as session:
            try:
                await session.execute(statement)
            except Exception as exc:
                raise RuntimeError(
                    "pgvector could not create the configured "
                    f"{self._config.index_type} "
                    f"index for dimension {vector.dimension}"
                ) from exc
        self._indexes.add(key)

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[EmbeddingVector, ...],
    ) -> None:
        """Insert or replace one compatible project-scoped vector batch."""
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
        await self._ensure_schema()
        async with self._repository.session_factory.begin() as session:
            for chunk, vector in zip(chunks, vectors, strict=True):
                await session.execute(
                    text(
                        f"""
                        INSERT INTO {_TABLE}
                            (project_id, chunk_id, ingestion_id, document_id,
                             model_identity, dimension, embedding)
                        VALUES (:project_id, :chunk_id, :ingestion_id, :document_id,
                                :model_identity, :dimension, CAST(:embedding AS vector))
                        ON CONFLICT (project_id, chunk_id, model_identity)
                        DO UPDATE SET ingestion_id = EXCLUDED.ingestion_id,
                                      document_id = EXCLUDED.document_id,
                                      dimension = EXCLUDED.dimension,
                                      embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "project_id": project_id,
                        "chunk_id": chunk.chunk_id,
                        "ingestion_id": chunk.ingestion_id,
                        "document_id": chunk.document_id,
                        "model_identity": vector.model_identity,
                        "dimension": vector.dimension,
                        "embedding": _literal_vector(vector.values),
                    },
                )
        await self._ensure_index(first)

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Delete all derived pgvector rows for one ingestion."""
        await self._ensure_schema()
        async with self._repository.session_factory.begin() as session:
            await session.execute(
                text(f"DELETE FROM {_TABLE} WHERE ingestion_id = :ingestion_id"),
                {"ingestion_id": ingestion_id},
            )

    async def delete_document(self, document_id: str) -> None:
        """Delete all derived pgvector rows for one canonical document."""
        await self._ensure_schema()
        async with self._repository.session_factory.begin() as session:
            await session.execute(
                text(f"DELETE FROM {_TABLE} WHERE document_id = :document_id"),
                {"document_id": document_id},
            )

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> bool:
        """Reuse every vector row for an ingestion under another project."""
        await self._ensure_schema()
        async with self._repository.session_factory.begin() as session:
            result = await session.execute(
                text(
                    f"""
                    INSERT INTO {_TABLE}
                        (project_id, chunk_id, ingestion_id, document_id,
                         model_identity, dimension, embedding)
                    SELECT :project_id, chunk_id, ingestion_id, document_id,
                           model_identity, dimension, embedding
                    FROM {_TABLE}
                    WHERE ingestion_id = :ingestion_id
                    ON CONFLICT (project_id, chunk_id, model_identity) DO NOTHING
                    """
                ),
                {"project_id": project_id, "ingestion_id": ingestion_id},
            )
            return bool(getattr(result, "rowcount", 0))

    async def remove_project(self, project_id: str) -> None:
        """Remove one project's derived rows without affecting other projects."""
        await self._ensure_schema()
        async with self._repository.session_factory.begin() as session:
            await session.execute(
                text(f"DELETE FROM {_TABLE} WHERE project_id = :project_id"),
                {"project_id": project_id},
            )

    async def search(
        self,
        project_id: str,
        query: EmbeddingVector,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Run exact or indexed KNN and hydrate canonical PostgreSQL chunks."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        await self._ensure_schema()
        distance_expression = {
            "cosine": "embedding <=> CAST(:embedding AS vector)",
            "dot": "embedding <#> CAST(:embedding AS vector)",
            "euclidean": "embedding <-> CAST(:embedding AS vector)",
        }[self._config.similarity_metric]
        document_clause = ""
        parameters: dict[str, object] = {
            "project_id": project_id,
            "model_identity": query.model_identity,
            "dimension": query.dimension,
            "embedding": _literal_vector(query.values),
            "top_k": top_k,
        }
        if document_ids:
            document_clause = " AND document_id = ANY(:document_ids)"
            parameters["document_ids"] = list(document_ids)
        async with self._repository.session_factory() as session:
            if self._config.index_type == "hnsw":
                await session.execute(
                    text(f"SET LOCAL hnsw.ef_search = {self._config.hnsw_ef_search}")
                )
            elif self._config.index_type == "ivf":
                await session.execute(
                    text(f"SET LOCAL ivfflat.probes = {self._config.nprobe}")
                )
            rows = (
                await session.execute(
                    text(
                        f"""
                        SELECT chunk_id, {distance_expression} AS distance
                        FROM {_TABLE}
                        WHERE project_id = :project_id
                          AND model_identity = :model_identity
                          AND dimension = :dimension{document_clause}
                        ORDER BY distance ASC, chunk_id ASC
                        LIMIT :top_k
                        """
                    ),
                    parameters,
                )
            ).mappings().all()
        ids = tuple(str(row["chunk_id"]) for row in rows)
        chunks = await self._repository.chunks_by_ids(ids)
        return tuple(
            ChunkRetrievalResult(
                chunks[chunk_id],
                rank,
                "pgvector",
                self._score(float(row["distance"])),
            )
            for rank, (row, chunk_id) in enumerate(zip(rows, ids, strict=True), 1)
            if chunk_id in chunks
        )

    def _score(self, distance: float) -> float:
        """Convert pgvector's distance ordering into the shared score ordering."""
        if self._config.similarity_metric == "cosine":
            return 1.0 - distance
        return -distance

    async def fetch_vectors(
        self, chunk_ids: tuple[str, ...], *, limit: int = 128
    ) -> tuple[EmbeddingRecord, ...]:
        """Fetch bounded vectors for evaluation diagnostics."""
        if not chunk_ids:
            return ()
        if limit < 1 or limit > 128:
            raise ValueError("Embedding sampling limit must be in [1, 128]")
        await self._ensure_schema()
        async with self._repository.session_factory() as session:
            rows = (
                await session.execute(
                    text(
                        f"SELECT chunk_id, embedding, model_identity FROM {_TABLE} "
                        "WHERE chunk_id = ANY(:chunk_ids) LIMIT :limit"
                    ),
                    {
                        "chunk_ids": list(dict.fromkeys(chunk_ids))[:limit],
                        "limit": limit,
                    },
                )
            ).mappings().all()
        output: list[EmbeddingRecord] = []
        for row in rows:
            raw_values = row["embedding"]
            if isinstance(raw_values, str):
                raw_values = raw_values.strip("[]").split(",")
            values = tuple(float(value) for value in raw_values)
            vector = EmbeddingVector(values, str(row["model_identity"]))
            output.append(
                EmbeddingRecord(
                    vector=list(values),
                    vector_norm=vector.vector_norm,
                    embedding_id=vector.embedding_id,
                    source_id=str(row["chunk_id"]),
                    source_type="document_chunk",
                    model_id=vector.model_identity,
                )
            )
        return tuple(output)

    async def close(self) -> None:
        """Release no dedicated resources; PostgreSQL remains application-owned."""
