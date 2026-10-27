"""Resolve configured vector backends into concrete infrastructure adapters."""

from collections.abc import Callable

from llm_grounding.infrastructure.config.models import VectorBackendConfig
from llm_grounding.infrastructure.config.settings import QdrantSettings
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.infrastructure.vector.qdrant import QdrantVectorStore

VectorBackendConstructor = Callable[[PostgresRepository], VectorStore]


class VectorStoreFactory:
    """Construct registered vector stores without coupling callers to Qdrant."""

    def __init__(self) -> None:
        """Register available concrete backend constructors."""
        self._factories: dict[str, VectorBackendConstructor] = {
            "qdrant": self._create_qdrant
        }

    def register(
        self, implementation: str, constructor: VectorBackendConstructor
    ) -> None:
        """Register a backend adapter without changing pipeline orchestration."""
        if not implementation.strip():
            raise ValueError("Vector backend implementation name must be non-empty")
        self._factories[implementation] = constructor

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
        return factory(repository)

    @property
    def implementations(self) -> frozenset[str]:
        """Return registered vector backend identifiers."""
        return frozenset(self._factories)

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
