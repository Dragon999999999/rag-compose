"""Resolve configured vector backends into concrete infrastructure adapters."""

from collections.abc import Callable

from llm_grounding.infrastructure.config.models import VectorBackendConfig
from llm_grounding.infrastructure.config.settings import QdrantSettings
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.infrastructure.vector.qdrant import QdrantVectorStore


class VectorStoreFactory:
    """Construct registered vector stores without coupling callers to Qdrant."""

    def __init__(self) -> None:
        """Register available concrete backend constructors."""
        self._factories: dict[
            str, Callable[[PostgresRepository], VectorStore]
        ] = {"qdrant": self._create_qdrant}

    def create(
        self, config: VectorBackendConfig, repository: PostgresRepository
    ) -> VectorStore:
        """Construct the selected backend and surface unavailable infrastructure."""
        try:
            factory = self._factories[config.type]
        except KeyError as exc:
            raise ValueError(
                f"No vector backend implementation is registered for {config.type!r}"
            ) from exc
        return factory(repository)

    @staticmethod
    def _create_qdrant(repository: PostgresRepository) -> VectorStore:
        """Construct Qdrant only after the effective config selected it."""
        try:
            settings = QdrantSettings.from_environment()
            return QdrantVectorStore(settings, repository)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Configured vector backend 'qdrant' is unavailable; "
                "set QDRANT_URL and verify the service is reachable"
            ) from exc
