"""Verify embedding-, generation-, and token-model chunking contracts."""

import re

from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.embedding.models import EmbeddingVector, TokenEmbedding
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.late_chunking import LateChunker
from llm_grounding.knowledge.chunking.llm_guided import LLMGuidedChunker
from llm_grounding.knowledge.chunking.semantic import SemanticChunker


def _evidence() -> tuple[EvidenceUnit, ...]:
    """Create one source block with three independently meaningful sentences."""
    return (
        EvidenceUnit(
            evidence_id="evidence-1",
            document_id="document-1",
            evidence_type=EvidenceType.TEXT,
            source_uri="file:///document.txt",
            order=0,
            page_number=1,
            text="Alpha topic. Alpha continuation. Beta topic.",
        ),
    )


class _SemanticEmbedder:
    """Return deterministic sentence vectors with one semantic discontinuity."""

    @property
    def model_identity(self) -> str:
        """Return the fake embedding-space identity."""
        return "test-semantic-space"

    async def embed_documents(
        self, texts: list[str]
    ) -> tuple[EmbeddingVector, ...]:
        """Encode the two alpha sentences similarly and beta oppositely."""
        return tuple(
            EmbeddingVector(
                (1.0, 0.0) if text.startswith("Alpha") else (-1.0, 0.0),
                self.model_identity,
            )
            for text in texts
        )

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Provide the protocol's unused query method for completeness."""
        return EmbeddingVector((1.0, 0.0), self.model_identity)


class _BoundaryGenerator:
    """Return a valid boundary proposal or an invalid response on demand."""

    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Return the configured response without using the prompt or token limit."""
        del prompt, max_new_tokens
        return self.response


class _TokenEmbedder:
    """Return source-aligned token vectors and count model invocations."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_identity(self) -> str:
        """Return the token model identity."""
        return "test-late-space"

    async def embed_token_documents(
        self, texts: list[str]
    ) -> tuple[TokenEmbedding, ...]:
        """Encode whitespace tokens with offsets into the original document."""
        self.calls += 1
        text = texts[0]
        spans = tuple(
            (match.start(), match.end()) for match in re.finditer(r"\S+", text)
        )
        offsets = spans
        values = tuple((float(index + 1), 0.0) for index in range(len(offsets)))
        return (TokenEmbedding(values, self.model_identity, offsets),)

    async def embed_token_query(self, text: str) -> TokenEmbedding:
        """Provide the protocol's unused query method for completeness."""
        del text
        return TokenEmbedding(((1.0, 0.0),), self.model_identity, ((0, 1),))


async def test_semantic_chunker_uses_embedding_distance_as_boundary() -> None:
    """Semantic chunking keeps nearby sentences and splits the topic transition."""
    config = ChunkerConfig(
        implementation="semantic",
        model_id="embedding_gemma_300m",
        target_tokens=20,
        max_tokens=20,
        overlap_tokens=0,
        semantic_threshold=0.5,
    )

    chunks = await SemanticChunker(config, _SemanticEmbedder()).chunk(
        _evidence(), "ingestion-1", "document.txt"
    )

    assert [chunk.text for chunk in chunks] == [
        "Alpha topic. Alpha continuation.",
        "Beta topic.",
    ]
    assert all(chunk.evidence_refs for chunk in chunks)


async def test_llm_guided_chunker_validates_boundaries_and_falls_back() -> None:
    """Valid JSON controls segmentation while malformed JSON uses token fallback."""
    config = ChunkerConfig(
        implementation="llm_guided",
        model_id="qwen3_1_7b",
        target_tokens=20,
        max_tokens=20,
        overlap_tokens=0,
    )
    guided = await LLMGuidedChunker(
        config, _BoundaryGenerator('{"boundaries": [0]}')
    ).chunk(_evidence(), "ingestion-1", "document.txt")
    fallback = await LLMGuidedChunker(
        config, _BoundaryGenerator("not json")
    ).chunk(_evidence(), "ingestion-1", "document.txt")

    assert [chunk.text for chunk in guided] == [
        "Alpha topic.",
        "Alpha continuation. Beta topic.",
    ]
    assert all(chunk.metadata["boundary_source"] == "llm" for chunk in guided)
    assert fallback
    assert all(
        chunk.metadata["boundary_source"] == "deterministic_fallback"
        for chunk in fallback
    )


async def test_late_chunker_reuses_pooled_vectors_and_requires_offsets() -> None:
    """Late chunking performs one pass and exposes pooled vectors downstream."""
    embedder = _TokenEmbedder()
    config = ChunkerConfig(
        implementation="late_chunking",
        model_id="embedding_jina_v2_base_en",
        target_tokens=2,
        max_tokens=2,
        overlap_tokens=0,
    )
    chunker = LateChunker(config, embedder, "jina-dense-space")

    chunks = await chunker.chunk(_evidence(), "ingestion-1", "document.txt")
    vectors = await chunker.embed_chunks(chunks)

    assert embedder.calls == 1
    assert len(chunks) == len(vectors) == 3
    assert all(vector.model_identity == "jina-dense-space" for vector in vectors)
    assert all(chunk.metadata["embedding_reused"] for chunk in chunks)
