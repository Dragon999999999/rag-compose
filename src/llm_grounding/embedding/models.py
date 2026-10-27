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


@dataclass(frozen=True)
class SparseEmbedding:
    """Represent a learned sparse vector by non-zero token weights.

    Indices and weights are kept in matching order.  Sparse providers may use a
    model vocabulary rather than a fixed dense dimension, so ``dimension`` is
    optional metadata and is never inferred from the number of non-zero values.
    """

    indices: tuple[int, ...]
    weights: tuple[float, ...]
    model_identity: str
    dimension: int | None = None

    def __post_init__(self) -> None:
        """Reject malformed sparse vectors before they reach an index."""
        if len(self.indices) != len(self.weights):
            raise ValueError("Sparse vector indices and weights must have equal length")
        if any(index < 0 for index in self.indices):
            raise ValueError("Sparse vector indices must be non-negative")
        if self.dimension is not None and any(
            index >= self.dimension for index in self.indices
        ):
            raise ValueError("Sparse vector index exceeds the declared dimension")

    @property
    def nnz(self) -> int:
        """Return the number of non-zero entries represented by this vector."""
        return len(self.indices)


@dataclass(frozen=True)
class TokenEmbedding:
    """Represent token-level embeddings and optional source-token offsets."""

    values: tuple[tuple[float, ...], ...]
    model_identity: str
    offsets: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        """Validate optional offsets without breaking legacy token providers."""
        if self.offsets and len(self.offsets) != len(self.values):
            raise ValueError("Token embedding offsets must match token values")
        if any(start < 0 or end < start for start, end in self.offsets):
            raise ValueError("Token embedding offsets must be non-negative spans")

    @property
    def token_count(self) -> int:
        """Return the number of encoded tokens."""
        return len(self.values)

    @property
    def dimension(self) -> int:
        """Return token embedding dimensionality."""
        return len(self.values[0]) if self.values else 0
