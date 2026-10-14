"""Tests for provider construction and request-scoped embedding selection."""

from types import SimpleNamespace

import pytest

from llm_grounding.domain.corpus import CorpusRequest, EvidenceReference, RetrievalUnit
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.embedding.openai_compatible import OpenAICompatibleEmbedder
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore
from llm_grounding.ingestion.models import EmbeddingConfig
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.retrieval.vector import VectorRetriever


def test_fake_factory_uses_persisted_embedding_identity() -> None:
    """Factory-created fake vectors match the identity persisted for ingestion."""
    config = EmbeddingConfig(type="fake", model="alternate", dimension=3)

    embedder = build_embedder(config)

    assert embedder.model_identity == config.identity()


async def test_runner_uses_request_embedding_override() -> None:
    """A structured embedding override reaches query-time vector encoding."""
    default_config = EmbeddingConfig(type="fake", model="default", dimension=3)
    override_config = EmbeddingConfig(type="fake", model="alternate", dimension=3)
    override = build_embedder(override_config)
    chunk = RetrievalUnit(
        "chunk",
        "document",
        "ingestion",
        "paper.pdf",
        "The alternate embedding space is indexed.",
        0,
        1,
        1,
        (EvidenceReference("evidence", 0, 40),),
    )
    store = InMemoryVectorStore()
    await store.upsert(
        "project", (chunk,), await override.embed_documents([chunk.text])
    )
    vector = VectorRetriever(
        build_embedder(default_config),
        store,
        embedding_config=default_config,
    )
    runner = BasicRagRunner(
        vector,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        embedding_config=default_config,
    )

    results = await runner.retrieve(
        CorpusRequest("project", "alternate embedding"),
        runtime_config={"config": {"embedding": override_config.model_dump()}},
    )

    assert [item.chunk.chunk_id for item in results] == ["chunk"]
    assert vector.last_query_embedding is not None
    assert vector.last_query_embedding.model_identity == override_config.identity()


async def test_incompatible_override_dimension_fails_clearly() -> None:
    """The vector boundary rejects a query from an incompatible embedding space."""
    indexed_config = EmbeddingConfig(type="fake", model="indexed", dimension=3)
    query_config = EmbeddingConfig(type="fake", model="query", dimension=4)
    indexed = build_embedder(indexed_config)
    chunk = RetrievalUnit(
        "chunk",
        "document",
        "ingestion",
        "paper.pdf",
        "Indexed content.",
        0,
        1,
        1,
        (),
    )
    store = InMemoryVectorStore()
    await store.upsert("project", (chunk,), await indexed.embed_documents([chunk.text]))
    vector = VectorRetriever(build_embedder(indexed_config), store)

    with pytest.raises(ValueError, match="incompatible"):
        await vector.retrieve(
            CorpusRequest("project", "query"),
            embedder=build_embedder(query_config),
        )


async def test_openai_compatible_embedder_normalizes_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote adapter maps ordered provider data into validated vectors."""
    calls: list[dict[str, object]] = []

    class FakeEmbeddings:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[0.0, 2.0]),
                    SimpleNamespace(index=0, embedding=[3.0, 0.0]),
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(
        "llm_grounding.embedding.openai_compatible.AsyncOpenAI", FakeClient
    )
    monkeypatch.setattr(
        "llm_grounding.embedding.openai_compatible.resolve_embedding_key",
        lambda: "test-key",
    )
    config = EmbeddingConfig(
        type="openai_compatible",
        provider="aqueduct",
        model="embedding-model",
        dimension=2,
        base_url="https://example.test/v1",
    )

    embedder = OpenAICompatibleEmbedder(config)
    vectors = await embedder.embed_documents(["first", "second"])

    assert vectors == (
        EmbeddingVector((1.0, 0.0), config.identity()),
        EmbeddingVector((0.0, 1.0), config.identity()),
    )
    assert calls == [{"model": "embedding-model", "input": ["first", "second"]}]
