"""Deterministic, offline embedding provider used by tests and local demonstrations."""

import math
import re
from collections import Counter
from hashlib import sha256

from llm_grounding.embedding.models import EmbeddingVector


class DeterministicFakeEmbedder:
    """Hash normalized word counts into a deterministic cosine-compatible space."""

    def __init__(self, dimension: int = 32) -> None:
        """Configure a small fixed dimension suitable only for tests."""
        if dimension < 2:
            raise ValueError("Fake embedder dimension must be at least 2")
        self._dimension = dimension

    @property
    def model_identity(self) -> str:
        """Identify the deterministic hashing representation."""
        return f"fake-hash-v1:{self._dimension}:normalized"

    def _embed(self, text: str) -> EmbeddingVector:
        """Hash word counts, L2-normalize them, and retain the configured identity."""
        values = [0.0] * self._dimension
        for token, count in Counter(re.findall(r"[\w]+", text.lower())).items():
            index = (
                int.from_bytes(sha256(token.encode("utf-8")).digest()[:8], "big")
                % self._dimension
            )
            values[index] += count
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
        return EmbeddingVector(tuple(values), self.model_identity)

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Embed a batch deterministically without model downloads or network I/O."""
        return tuple(self._embed(text) for text in texts)

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a query in the same hash space as documents."""
        return self._embed(text)
