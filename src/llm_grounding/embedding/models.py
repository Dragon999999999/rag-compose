"""Embedding values and identity metadata shared by ingestion and retrieval."""

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt


@dataclass(frozen=True)
class EmbeddingVector:
    """Represent one vector in a named compatible embedding space.

    ``model_identity`` is retained as the serialized field name for backward
    compatibility. For explicitly declared spaces it is a provider-independent
    space identity; legacy configurations continue to use their old identity.
    """

    values: tuple[float, ...]
    model_identity: str

    @property
    def dimension(self) -> int:
        """Return vector dimensionality for store compatibility validation."""
        return len(self.values)

    @property
    def vector_norm(self) -> float:
        """Return the Euclidean norm without normalizing or mutating the vector."""
        return sqrt(sum(value * value for value in self.values))

    @property
    def embedding_id(self) -> str:
        """Return a stable content identity for this model-space vector."""
        content = f"{self.model_identity}:{self.dimension}:".encode()
        values = ",".join(f"{value:.17g}" for value in self.values).encode()
        return sha256(content + values).hexdigest()
