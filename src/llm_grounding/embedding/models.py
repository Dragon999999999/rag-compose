"""Embedding values and identity metadata shared by ingestion and retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingVector:
    """Represent one dense vector in a named compatible embedding space."""

    values: tuple[float, ...]
    model_identity: str

    @property
    def dimension(self) -> int:
        """Return vector dimensionality for store compatibility validation."""
        return len(self.values)
