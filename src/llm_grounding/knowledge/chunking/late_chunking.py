"""Late chunking with token-aligned contextualized pooling."""

from math import sqrt

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.embedding.models import EmbeddingVector, TokenEmbedding
from llm_grounding.embedding.protocol import TokenEmbedder
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.common import (
    EvidenceStream,
    split_token_ranges,
)


class LateChunker:
    """Pool one long-document token pass into reusable chunk embeddings."""

    def __init__(
        self,
        config: ChunkerConfig,
        token_embedder: TokenEmbedder,
        embedding_identity: str,
    ) -> None:
        """Retain a token-capable provider and the dense index identity."""
        if config.model_id is None:
            raise ValueError("LateChunker requires a model_id")
        self._config = config
        self._token_embedder = token_embedder
        self._embedding_identity = embedding_identity
        self._vectors: dict[str, EmbeddingVector] = {}

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Create chunks and pool aligned token representations once."""
        stream = EvidenceStream.from_evidence(evidence)
        if not stream.text.strip():
            return ()
        encoded = await self._token_embedder.embed_token_documents([stream.text])
        if len(encoded) != 1:
            raise RuntimeError(
                "Late-chunking model returned an unexpected vector count"
            )
        token_embedding = encoded[0]
        offsets = token_embedding.offsets
        if not offsets:
            raise RuntimeError(
                "Late chunking requires a token embedding provider with source offsets"
            )
        self._validate_offsets(offsets, len(stream.text), token_embedding)
        ranges = split_token_ranges(
            offsets,
            self._config.target_tokens,
            self._config.max_tokens or self._config.target_tokens,
            self._config.overlap_tokens,
        )
        chunks: list[RetrievalUnit] = []
        for token_start, token_end in ranges:
            start, end = offsets[token_start][0], offsets[token_end - 1][1]
            values = token_embedding.values[token_start:token_end]
            vector = self._pool(values)
            chunk_id = f"{ingestion_id}:c{len(chunks):06d}"
            unit = stream.unit(
                ingestion_id,
                document_name,
                start,
                end,
                len(chunks),
                {
                    "chunking": "late_chunking",
                    "model_id": self._config.model_id,
                    "token_count": token_end - token_start,
                    "embedding_reused": True,
                    "embedding_identity": self._embedding_identity,
                },
                chunk_id=chunk_id,
            )
            if unit is not None:
                chunks.append(unit)
                self._vectors[chunk_id] = EmbeddingVector(
                    vector, self._embedding_identity
                )
        return tuple(chunks)

    async def embed_chunks(
        self, chunks: tuple[RetrievalUnit, ...]
    ) -> tuple[EmbeddingVector, ...]:
        """Return pooled vectors from the preceding chunk call without re-encoding."""
        try:
            return tuple(self._vectors[chunk.chunk_id] for chunk in chunks)
        except KeyError as exc:
            raise RuntimeError(
                "Late-chunk embeddings are unavailable; call chunk before indexing"
            ) from exc

    @staticmethod
    def _validate_offsets(
        offsets: tuple[tuple[int, int], ...],
        text_length: int,
        embedding: TokenEmbedding,
    ) -> None:
        """Reject unsupported or misaligned token providers before indexing."""
        if len(offsets) != len(embedding.values):
            raise RuntimeError("Late-chunking token offsets do not match embeddings")
        previous_end = 0
        for start, end in offsets:
            if start < previous_end or end <= start or end > text_length:
                raise RuntimeError("Late-chunking token offsets are not source aligned")
            previous_end = end

    @staticmethod
    def _pool(values: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        """Mean-pool contextualized token vectors and normalize the dense result."""
        if not values or not values[0]:
            raise RuntimeError("Late-chunking model returned an empty token span")
        dimension = len(values[0])
        if any(len(value) != dimension for value in values):
            raise RuntimeError("Late-chunking token dimensions are inconsistent")
        pooled = tuple(
            sum(value[index] for value in values) / len(values)
            for index in range(dimension)
        )
        norm = sqrt(sum(value * value for value in pooled))
        return tuple(value / norm for value in pooled) if norm else pooled
