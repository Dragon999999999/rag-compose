"""Resolve configured vector backends into concrete infrastructure adapters."""

import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.config.models import VectorBackendConfig
from llm_grounding.infrastructure.config.settings import FaissSettings, QdrantSettings
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.faiss import FaissVectorStore
from llm_grounding.infrastructure.vector.learned import (
    QdrantMultiVectorStore,
    QdrantSparseVectorStore,
)
from llm_grounding.infrastructure.vector.pgvector import PgVectorStore
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.infrastructure.vector.qdrant import QdrantVectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.observability.embedding import EmbeddingRecord

VectorBackendConstructor = Callable[
    [PostgresRepository, VectorBackendConfig], VectorStore
]


class VectorStoreFactory:
    """Construct registered vector stores without coupling callers to Qdrant."""

    def __init__(self) -> None:
        """Register available concrete backend constructors."""
        self._factories: dict[str, VectorBackendConstructor] = {
            "qdrant": self._create_qdrant,
            "pgvector": self._create_pgvector,
            "faiss": self._create_faiss,
        }

    def register(
        self,
        implementation: str,
        constructor: VectorBackendConstructor
        | Callable[[PostgresRepository], VectorStore],
    ) -> None:
        """Register a backend adapter without changing pipeline orchestration."""
        if not implementation.strip():
            raise ValueError("Vector backend implementation name must be non-empty")
        if len(inspect.signature(constructor).parameters) == 1:
            legacy_constructor = cast(
                Callable[[PostgresRepository], VectorStore], constructor
            )
            self._factories[implementation] = lambda repository, config: (
                legacy_constructor(repository)
            )
        else:
            self._factories[implementation] = cast(
                VectorBackendConstructor, constructor
            )

    def create(
        self, config: VectorBackendConfig, repository: PostgresRepository
    ) -> VectorStore:
        """Construct the selected backend and surface unavailable infrastructure."""
        try:
            factory = self._factories[config.implementation]
        except KeyError as exc:
            raise ValueError(
                "No vector backend implementation is registered for "
                f"{config.implementation!r}"
            ) from exc
        return factory(repository, config)

    @property
    def implementations(self) -> frozenset[str]:
        """Return registered vector backend identifiers."""
        return frozenset(self._factories)

    @staticmethod
    def _create_qdrant(
        repository: PostgresRepository, config: VectorBackendConfig
    ) -> VectorStore:
        """Construct Qdrant only after the effective config selected it."""
        try:
            settings = QdrantSettings.from_environment()
            return QdrantVectorStore(settings, repository)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Configured vector backend 'qdrant' is unavailable; "
                "set QDRANT_URL and verify the service is reachable"
            ) from exc

    @staticmethod
    def _create_pgvector(
        repository: PostgresRepository, config: VectorBackendConfig
    ) -> VectorStore:
        """Construct the lazy PostgreSQL adapter without creating its table yet."""
        return PgVectorStore(config, repository)

    @staticmethod
    def _create_faiss(
        repository: PostgresRepository, config: VectorBackendConfig
    ) -> VectorStore:
        """Construct lazy FAISS storage without importing FAISS or loading indexes."""
        settings = FaissSettings.from_environment()
        root = Path(settings.root).expanduser()
        if config.storage_path is not None:
            configured = Path(config.storage_path).expanduser()
            root = configured if configured.is_absolute() else root / configured
        return FaissVectorStore(config, repository, root)


class VectorStoreRouter:
    """Resolve one lazily constructed vector store per effective ingestion config."""

    def __init__(
        self,
        factory: VectorStoreFactory,
        repository: PostgresRepository,
        default_config: IngestionConfig,
    ) -> None:
        """Configure backend routing without contacting any vector service."""
        self._factory = factory
        self._repository = repository
        self._default_config = default_config
        self._stores: dict[str, VectorStore] = {}

    @staticmethod
    def _backend_config(config: IngestionConfig) -> VectorBackendConfig:
        """Convert resolved ingestion settings into the typed backend contract."""
        if config.vector_backend is None:
            raise ValueError("A vector backend is required for dense ingestion")
        values = dict(config.vector_backend_config)
        values["implementation"] = config.vector_backend
        values.setdefault("index_version", config.index_version)
        return VectorBackendConfig.model_validate(values)

    def for_ingestion_config(self, config: IngestionConfig) -> VectorStore:
        """Return a cached backend selected by the effective ingestion pipeline."""
        backend = self._backend_config(config)
        cache_key = repr(backend.model_dump(mode="json"))
        store = self._stores.get(cache_key)
        if store is None:
            if (
                config.embedding is not None
                and backend.implementation == "qdrant"
                and config.embedding.representation in {"sparse", "multivector"}
            ):
                settings = QdrantSettings.from_environment()
                model_identity = config.embedding.vector_identity()
                store = cast(
                    VectorStore,
                    (
                        QdrantSparseVectorStore(
                            settings, self._repository, model_identity
                        )
                        if config.embedding.representation == "sparse"
                        else QdrantMultiVectorStore(
                            settings, self._repository, model_identity
                        )
                    ),
                )
            else:
                store = self._factory.create(backend, self._repository)
            assert store is not None
            self._stores[cache_key] = store
        return store

    def _default_store(self) -> VectorStore:
        """Return the store used by legacy callers without effective config."""
        return self.for_ingestion_config(self._default_config)

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[EmbeddingVector, ...],
    ) -> None:
        """Delegate protocol calls from legacy/default callers."""
        await self._default_store().upsert(project_id, chunks, vectors)

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Delete an ingestion from every instantiated backend."""
        for store in tuple(self._stores.values()):
            await store.delete_ingestion(ingestion_id)

    async def delete_document(self, document_id: str) -> None:
        """Delete a document from every instantiated backend."""
        for store in tuple(self._stores.values()):
            await store.delete_document(document_id)

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> bool:
        """Delegate legacy association calls to the default backend."""
        return await self._default_store().associate_ingestion(project_id, ingestion_id)

    async def remove_project(self, project_id: str) -> None:
        """Remove project membership from every instantiated backend."""
        for store in tuple(self._stores.values()):
            await store.remove_project(project_id)

    async def search(
        self,
        project_id: str,
        query: EmbeddingVector,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Delegate legacy searches to the default backend."""
        return await self._default_store().search(
            project_id, query, top_k, document_ids
        )

    async def fetch_vectors(
        self, chunk_ids: tuple[str, ...], *, limit: int = 128
    ) -> tuple[EmbeddingRecord, ...]:
        """Fetch diagnostics from the default backend when available."""
        store = self._default_store()
        fetcher = getattr(store, "fetch_vectors", None)
        if fetcher is None:
            return ()
        typed_fetcher = cast(
            Callable[..., Awaitable[tuple[EmbeddingRecord, ...]]], fetcher
        )
        return await typed_fetcher(chunk_ids, limit=limit)

    async def close(self) -> None:
        """Close all instantiated stores while preserving external services."""
        for store in tuple(self._stores.values()):
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    async def health_check(self) -> None:
        """Check the configured default backend without executing a search."""
        store = self._default_store()
        check = getattr(store, "health_check", None)
        if check is None:
            raise RuntimeError("Vector backend health check is unavailable")
        await check()
