"""Qdrant stores for learned sparse and token-level representations.

These stores intentionally use collections separate from the existing dense
collection.  A dense collection cannot be changed from a single unnamed vector
to sparse or multivector configuration without invalidating existing indexes.
Canonical chunks remain in PostgreSQL and are hydrated after every search.
"""

from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import AsyncQdrantClient, models

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import SparseEmbedding, TokenEmbedding
from llm_grounding.infrastructure.config.settings import QdrantSettings
from llm_grounding.infrastructure.database.repositories import PostgresRepository

_PROJECT_IDS = "project_ids"
_DOCUMENT_ID = "document_id"
_INGESTION_ID = "ingestion_id"
_CHUNK_ID = "retrieval_unit_id"
_MODEL_IDENTITY = "model_identity"
_DIMENSION = "dimension"


class QdrantSparseVectorStore:
    """Persist and search learned sparse vectors in a dedicated Qdrant collection."""

    def __init__(
        self,
        settings: QdrantSettings,
        repository: PostgresRepository,
        model_identity: str,
    ) -> None:
        """Create a lazy client; collection setup waits for the first upsert."""
        self._collection = f"{settings.collection}_sparse"
        self._repository = repository
        self._model_identity = model_identity
        self._client = AsyncQdrantClient(url=settings.url, api_key=settings.api_key)

    async def _ensure_collection(self) -> None:
        """Create the sparse collection only when learned sparse retrieval is used."""
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            self._collection,
            vectors_config={},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
        )
        for field in (_PROJECT_IDS, _DOCUMENT_ID, _INGESTION_ID, _MODEL_IDENTITY):
            await self._client.create_payload_index(
                self._collection, field, models.PayloadSchemaType.KEYWORD
            )

    @staticmethod
    def _point_id(chunk: RetrievalUnit, model_identity: str) -> str:
        """Generate a collection- and model-specific stable Qdrant UUID."""
        return str(
            uuid5(
                NAMESPACE_URL,
                f"sparse:{model_identity}:{chunk.ingestion_id}:{chunk.chunk_id}",
            )
        )

    def _filter(
        self,
        *,
        project_id: str | None = None,
        ingestion_id: str | None = None,
        document_ids: tuple[str, ...] = (),
    ) -> models.Filter:
        """Build project, document, ingestion, and model predicates."""
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key=_MODEL_IDENTITY,
                match=models.MatchValue(value=self._model_identity),
            )
        ]
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
        return models.Filter(must=conditions)

    async def has_project_model(self, project_id: str) -> bool:
        """Return whether this model already has at least one project vector."""
        if not await self._client.collection_exists(self._collection):
            return False
        records, _ = await self._client.scroll(
            self._collection,
            scroll_filter=self._filter(project_id=project_id),
            with_payload=False,
            with_vectors=False,
            limit=1,
        )
        return bool(records)

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[SparseEmbedding, ...],
    ) -> None:
        """Persist sparse vectors and project-scoped metadata."""
        if len(chunks) != len(vectors):
            raise ValueError("Each retrieval chunk needs exactly one sparse vector")
        if not vectors:
            return
        if any(vector.model_identity != self._model_identity for vector in vectors):
            raise ValueError("Sparse vector model identity does not match the store")
        await self._ensure_collection()
        points = [
            models.PointStruct(
                id=self._point_id(chunk, self._model_identity),
                vector={
                    "sparse": models.SparseVector(
                        indices=list(vector.indices), values=list(vector.weights)
                    )
                },
                payload={
                    _CHUNK_ID: chunk.chunk_id,
                    _INGESTION_ID: chunk.ingestion_id,
                    _DOCUMENT_ID: chunk.document_id,
                    _PROJECT_IDS: [project_id],
                    _MODEL_IDENTITY: self._model_identity,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(self._collection, points, wait=True)

    async def search(
        self,
        project_id: str,
        query: SparseEmbedding,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search learned sparse vectors and hydrate canonical chunks."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if query.model_identity != self._model_identity:
            raise ValueError("Sparse query identity is incompatible with the index")
        if not await self._client.collection_exists(self._collection):
            return ()
        response = await self._client.query_points(
            self._collection,
            query=models.SparseVector(
                indices=list(query.indices), values=list(query.weights)
            ),
            using="sparse",
            query_filter=self._filter(project_id=project_id, document_ids=document_ids),
            limit=top_k,
            with_payload=[_CHUNK_ID],
            with_vectors=False,
        )
        matched = tuple(
            (point, point.payload[_CHUNK_ID])
            for point in response.points
            if point.payload is not None
            and isinstance(point.payload.get(_CHUNK_ID), str)
        )
        chunks = await self._repository.chunks_by_ids(
            tuple(item[1] for item in matched)
        )
        return tuple(
            ChunkRetrievalResult(chunks[chunk_id], rank, "qdrant_sparse", point.score)
            for rank, (point, chunk_id) in enumerate(matched, 1)
            if chunk_id in chunks
        )

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Delete sparse vectors derived from one ingestion."""
        if await self._client.collection_exists(self._collection):
            await self._client.delete(
                self._collection,
                points_selector=self._filter(ingestion_id=ingestion_id),
                wait=True,
            )

    async def delete_document(self, document_id: str) -> None:
        """Delete sparse vectors belonging to one document."""
        if await self._client.collection_exists(self._collection):
            await self._client.delete(
                self._collection,
                points_selector=self._filter(document_ids=(document_id,)),
                wait=True,
            )

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> bool:
        """Associate reusable sparse vectors with another project."""
        if not await self._client.collection_exists(self._collection):
            return False
        records, _ = await self._client.scroll(
            self._collection,
            scroll_filter=self._filter(ingestion_id=ingestion_id),
            with_payload=True,
            with_vectors=False,
            limit=10_000,
        )
        for record in records:
            project_ids = set((record.payload or {}).get(_PROJECT_IDS, []))
            project_ids.add(project_id)
            await self._client.set_payload(
                self._collection,
                payload={_PROJECT_IDS: sorted(project_ids)},
                points=[record.id],
                wait=True,
            )
        return bool(records)

    async def remove_project(self, project_id: str) -> None:
        """Remove one project's sparse membership without deleting shared points."""
        if not await self._client.collection_exists(self._collection):
            return
        records, _ = await self._client.scroll(
            self._collection,
            scroll_filter=self._filter(project_id=project_id),
            with_payload=True,
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

    async def close(self) -> None:
        """Close the lazy Qdrant client."""
        await self._client.close()


class QdrantMultiVectorStore:
    """Persist token embeddings for MaxSim late-interaction retrieval."""

    def __init__(
        self,
        settings: QdrantSettings,
        repository: PostgresRepository,
        model_identity: str,
    ) -> None:
        """Create a separate lazy multivector collection for one token model."""
        self._collection = f"{settings.collection}_multivector"
        self._repository = repository
        self._model_identity = model_identity
        self._client = AsyncQdrantClient(url=settings.url, api_key=settings.api_key)

    async def _ensure_collection(self, dimension: int) -> None:
        """Create or validate a MaxSim collection with the model token dimension."""
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                self._collection,
                vectors_config={
                    "tokens": models.VectorParams(
                        size=dimension,
                        distance=models.Distance.DOT,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    )
                },
            )
            for field in (_PROJECT_IDS, _DOCUMENT_ID, _INGESTION_ID, _MODEL_IDENTITY):
                await self._client.create_payload_index(
                    self._collection, field, models.PayloadSchemaType.KEYWORD
                )
            return
        configured = (
            await self._client.get_collection(self._collection)
        ).config.params.vectors
        if not isinstance(configured, dict) or "tokens" not in configured:
            raise ValueError("Qdrant multivector collection has an incompatible schema")
        if configured["tokens"].size != dimension:
            raise ValueError("Qdrant multivector dimension is incompatible")

    @staticmethod
    def _point_id(chunk: RetrievalUnit, model_identity: str) -> str:
        """Generate a stable token-index point ID."""
        return str(
            uuid5(
                NAMESPACE_URL,
                f"multivector:{model_identity}:{chunk.ingestion_id}:{chunk.chunk_id}",
            )
        )

    @staticmethod
    def _filter(
        model_identity: str,
        project_id: str,
        document_ids: tuple[str, ...],
    ) -> models.Filter:
        """Build the model and project scope filter."""
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key=_MODEL_IDENTITY, match=models.MatchValue(value=model_identity)
            ),
            models.FieldCondition(
                key=_PROJECT_IDS, match=models.MatchValue(value=project_id)
            ),
        ]
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key=_DOCUMENT_ID, match=models.MatchAny(any=list(document_ids))
                )
            )
        return models.Filter(must=conditions)

    async def has_project_model(self, project_id: str) -> bool:
        """Return whether any token index points exist for a project."""
        if not await self._client.collection_exists(self._collection):
            return False
        records, _ = await self._client.scroll(
            self._collection,
            scroll_filter=self._filter(self._model_identity, project_id, ()),
            with_payload=False,
            with_vectors=False,
            limit=1,
        )
        return bool(records)

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[TokenEmbedding, ...],
    ) -> None:
        """Persist token matrices and project metadata."""
        if len(chunks) != len(vectors):
            raise ValueError("Each retrieval chunk needs exactly one token embedding")
        if not vectors:
            return
        dimension = vectors[0].dimension
        if dimension < 1 or any(
            vector.model_identity != self._model_identity
            or vector.dimension != dimension
            for vector in vectors
        ):
            raise ValueError("Token vectors must share identity and dimension")
        await self._ensure_collection(dimension)
        points = [
            models.PointStruct(
                id=self._point_id(chunk, self._model_identity),
                vector={"tokens": [list(token) for token in vector.values]},
                payload={
                    _CHUNK_ID: chunk.chunk_id,
                    _INGESTION_ID: chunk.ingestion_id,
                    _DOCUMENT_ID: chunk.document_id,
                    _PROJECT_IDS: [project_id],
                    _MODEL_IDENTITY: self._model_identity,
                    _DIMENSION: dimension,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(self._collection, points, wait=True)

    async def search(
        self,
        project_id: str,
        query: TokenEmbedding,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search token matrices using Qdrant's MaxSim comparator."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if query.model_identity != self._model_identity or not query.values:
            raise ValueError("Token query is incompatible with the index")
        if not await self._client.collection_exists(self._collection):
            return ()
        response = await self._client.query_points(
            self._collection,
            query=[list(token) for token in query.values],
            using="tokens",
            query_filter=self._filter(self._model_identity, project_id, document_ids),
            limit=top_k,
            with_payload=[_CHUNK_ID],
            with_vectors=False,
        )
        matched = tuple(
            (point, point.payload[_CHUNK_ID])
            for point in response.points
            if point.payload is not None
            and isinstance(point.payload.get(_CHUNK_ID), str)
        )
        chunks = await self._repository.chunks_by_ids(
            tuple(item[1] for item in matched)
        )
        return tuple(
            ChunkRetrievalResult(
                chunks[chunk_id], rank, "qdrant_multivector", point.score
            )
            for rank, (point, chunk_id) in enumerate(matched, 1)
            if chunk_id in chunks
        )

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Delete token vectors from one ingestion."""
        if await self._client.collection_exists(self._collection):
            await self._client.delete(
                self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=_INGESTION_ID,
                                match=models.MatchValue(value=ingestion_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )

    async def delete_document(self, document_id: str) -> None:
        """Delete token vectors from one document."""
        if await self._client.collection_exists(self._collection):
            await self._client.delete(
                self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=_DOCUMENT_ID,
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> bool:
        """Associate reusable token vectors with a project."""
        if not await self._client.collection_exists(self._collection):
            return False
        records, _ = await self._client.scroll(
            self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=_INGESTION_ID, match=models.MatchValue(value=ingestion_id)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=10_000,
        )
        for record in records:
            project_ids = set((record.payload or {}).get(_PROJECT_IDS, []))
            project_ids.add(project_id)
            await self._client.set_payload(
                self._collection,
                {_PROJECT_IDS: sorted(project_ids)},
                [record.id],
                wait=True,
            )
        return bool(records)

    async def remove_project(self, project_id: str) -> None:
        """Remove token vectors only when no other project uses them."""
        if not await self._client.collection_exists(self._collection):
            return
        records, _ = await self._client.scroll(
            self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=_PROJECT_IDS, match=models.MatchValue(value=project_id)
                    )
                ]
            ),
            with_payload=True,
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
                    {_PROJECT_IDS: sorted(project_ids)},
                    [record.id],
                    wait=True,
                )
            else:
                delete_ids.append(record.id)
        if delete_ids:
            await self._client.delete(
                self._collection, points_selector=delete_ids, wait=True
            )

    async def close(self) -> None:
        """Close the lazy Qdrant client."""
        await self._client.close()
