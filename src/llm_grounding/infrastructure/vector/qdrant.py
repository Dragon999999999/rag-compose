"""Qdrant-derived vector index adapter with in-index project filtering.

Qdrant point payloads contain only identifiers and filterable representation
metadata. Search results are hydrated from canonical PostgreSQL chunks, so the
index can be deleted and rebuilt without losing corpus text or provenance.
"""

from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import AsyncQdrantClient, models

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.config.settings import QdrantSettings
from llm_grounding.infrastructure.database.repositories import PostgresRepository

_PROJECT_IDS = "project_ids"
_DOCUMENT_ID = "document_id"
_INGESTION_ID = "ingestion_id"
_CHUNK_ID = "retrieval_unit_id"
_MODEL_IDENTITY = "model_identity"
_DIMENSION = "dimension"


class QdrantVectorStore:
    """Maintain one versioned Qdrant collection for derived retrieval vectors."""

    def __init__(
        self, settings: QdrantSettings, repository: PostgresRepository
    ) -> None:
        """Configure the client and canonical hydration repository.

        Collection creation is deferred to the first vector batch because its
        dimension is part of the immutable collection contract.
        """
        self._collection = settings.collection
        self._repository = repository
        self._client = AsyncQdrantClient(url=settings.url, api_key=settings.api_key)

    async def _ensure_collection(self, vector: EmbeddingVector) -> None:
        """Create or validate the collection's embedding-space dimensions."""
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    self._collection,
                    vectors_config=models.VectorParams(
                        size=vector.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
                for field in (
                    _PROJECT_IDS,
                    _DOCUMENT_ID,
                    _INGESTION_ID,
                    _MODEL_IDENTITY,
                ):
                    await self._client.create_payload_index(
                        self._collection, field, models.PayloadSchemaType.KEYWORD
                    )
                return
            collection = await self._client.get_collection(self._collection)
        except Exception as exc:
            raise RuntimeError(
                "Qdrant is unavailable or collection setup failed"
            ) from exc
        configured = collection.config.params.vectors
        if (
            not isinstance(configured, models.VectorParams)
            or configured.size != vector.dimension
        ):
            raise ValueError(
                "Qdrant collection dimension is incompatible with the configured "
                "embedder"
            )
        if configured.distance != models.Distance.COSINE:
            raise ValueError("Qdrant collection must use cosine distance")

    @staticmethod
    def _point_id(chunk: RetrievalUnit) -> str:
        """Generate a stable UUID acceptable to Qdrant for one ingestion chunk."""
        return str(uuid5(NAMESPACE_URL, f"{chunk.ingestion_id}:{chunk.chunk_id}"))

    @staticmethod
    def _filter(
        project_id: str | None = None,
        ingestion_id: str | None = None,
        document_ids: tuple[str, ...] = (),
        model_identity: str | None = None,
        dimension: int | None = None,
    ) -> models.Filter:
        """Build provider-private filter predicates for payload metadata."""
        conditions: list[models.Condition] = []
        if project_id is not None:
            conditions.append(
                models.FieldCondition(
                    key=_PROJECT_IDS, match=models.MatchValue(value=project_id)
                )
            )
        if ingestion_id is not None:
            conditions.append(
                models.FieldCondition(
                    key=_INGESTION_ID, match=models.MatchValue(value=ingestion_id)
                )
            )
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key=_DOCUMENT_ID, match=models.MatchAny(any=list(document_ids))
                )
            )
        if model_identity is not None:
            conditions.append(
                models.FieldCondition(
                    key=_MODEL_IDENTITY,
                    match=models.MatchValue(value=model_identity),
                )
            )
        if dimension is not None:
            conditions.append(
                models.FieldCondition(
                    key=_DIMENSION, match=models.MatchValue(value=dimension)
                )
            )
        return models.Filter(must=conditions)

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[EmbeddingVector, ...],
    ) -> None:
        """Upsert one compatible chunk/vector batch and its project association."""
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
        await self._ensure_collection(first)
        points = [
            models.PointStruct(
                id=self._point_id(chunk),
                vector=list(vector.values),
                payload={
                    _CHUNK_ID: chunk.chunk_id,
                    _INGESTION_ID: chunk.ingestion_id,
                    _DOCUMENT_ID: chunk.document_id,
                    _PROJECT_IDS: [project_id],
                    _MODEL_IDENTITY: vector.model_identity,
                    _DIMENSION: vector.dimension,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            await self._client.upsert(self._collection, points, wait=True)
        except Exception as exc:
            raise RuntimeError("Qdrant could not index document vectors") from exc

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Delete derived ingestion vectors, leaving PostgreSQL untouched."""
        try:
            if await self._client.collection_exists(self._collection):
                await self._client.delete(
                    self._collection,
                    points_selector=self._filter(ingestion_id=ingestion_id),
                    wait=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "Qdrant could not delete stale ingestion vectors"
            ) from exc

    async def delete_document(self, document_id: str) -> None:
        """Delete derived vectors for a canonical document that is being removed."""
        try:
            if await self._client.collection_exists(self._collection):
                await self._client.delete(
                    self._collection,
                    points_selector=self._filter(document_ids=(document_id,)),
                    wait=True,
                )
        except Exception as exc:
            raise RuntimeError("Qdrant could not delete document vectors") from exc

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> None:
        """Add project payload membership to existing points without re-embedding."""
        try:
            records, _ = await self._client.scroll(
                self._collection,
                scroll_filter=self._filter(ingestion_id=ingestion_id),
                with_payload=[_PROJECT_IDS],
                with_vectors=False,
                limit=10_000,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qdrant could not load reusable ingestion vectors"
            ) from exc
        if not records:
            raise KeyError(f"No indexed vectors for ingestion: {ingestion_id}")
        for record in records:
            payload = record.payload or {}
            project_ids = set(payload.get(_PROJECT_IDS, []))
            project_ids.add(project_id)
            await self._client.set_payload(
                self._collection,
                payload={_PROJECT_IDS: sorted(project_ids)},
                points=[record.id],
                wait=True,
            )

    async def remove_project(self, project_id: str) -> None:
        """Remove project membership while preserving vectors shared elsewhere."""
        try:
            if not await self._client.collection_exists(self._collection):
                return
            records, _ = await self._client.scroll(
                self._collection,
                scroll_filter=self._filter(project_id=project_id),
                with_payload=[_PROJECT_IDS],
                with_vectors=False,
                limit=10_000,
            )
            delete_ids: list[int | str | UUID] = []
            for record in records:
                project_ids = set((record.payload or {}).get(_PROJECT_IDS, []))
                project_ids.discard(project_id)
                if project_ids:
                    await self._client.set_payload(
                        self._collection,
                        payload={_PROJECT_IDS: sorted(project_ids)},
                        points=[record.id],
                        wait=True,
                    )
                else:
                    delete_ids.append(record.id)
            if delete_ids:
                await self._client.delete(
                    self._collection, points_selector=delete_ids, wait=True
                )
        except Exception as exc:
            raise RuntimeError(
                "Qdrant could not remove project index membership"
            ) from exc

    async def search(
        self,
        project_id: str,
        query: EmbeddingVector,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search within Qdrant filters and hydrate canonical PostgreSQL chunks."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        await self._ensure_collection(query)
        try:
            response = await self._client.query_points(
                self._collection,
                query=list(query.values),
                query_filter=self._filter(
                    project_id=project_id,
                    document_ids=document_ids,
                    model_identity=query.model_identity,
                    dimension=query.dimension,
                ),
                limit=top_k,
                with_payload=[_CHUNK_ID],
                with_vectors=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qdrant could not search the project vector index"
            ) from exc
        matched = tuple(
            (point, point.payload[_CHUNK_ID])
            for point in response.points
            if point.payload is not None
            and isinstance(point.payload.get(_CHUNK_ID), str)
        )
        ids = tuple(chunk_id for _, chunk_id in matched)
        chunks = await self._repository.chunks_by_ids(ids)
        return tuple(
            ChunkRetrievalResult(chunks[chunk_id], rank, "qdrant", point.score)
            for rank, (point, chunk_id) in enumerate(matched, 1)
            if chunk_id in chunks
        )

    async def close(self) -> None:
        """Release Qdrant HTTP/gRPC client resources during application shutdown."""
        await self._client.close()
