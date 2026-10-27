"""Tests for provider construction and request-scoped embedding selection."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from llm_grounding.domain.corpus import CorpusRequest, EvidenceReference, RetrievalUnit
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.local_manager import ManagedLocalEmbedder
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.embedding.openai_compatible import OpenAICompatibleEmbedder
from llm_grounding.infrastructure.config.loader import load_model_catalog
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore
from llm_grounding.ingestion.models import EmbeddingConfig
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.retrieval.vector import VectorRetriever


class _FakeLocalLease:
    """Record texts passed to a local inference lease."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        """Return one fixed-size vector per requested input."""
        self.calls.append(texts)
        return tuple(self.values for _ in texts)


class _FakeLocalManager:
    """Provide a context-managed local lease without Docker or network access."""

    def __init__(self, lease: _FakeLocalLease) -> None:
        self.lease = lease

    @asynccontextmanager
    async def acquire(self, _model_id: str):
        """Yield the fake lease used by the managed adapter."""
        yield self.lease


def test_fake_factory_uses_persisted_embedding_identity() -> None:
    """Factory-created fake vectors match the identity persisted for ingestion."""
    config = EmbeddingConfig(type="fake", model="alternate", dimension=3)

    embedder = build_embedder(config)

    assert embedder.model_identity == config.identity()


async def test_managed_local_embedder_applies_document_and_query_prefixes() -> None:
    """Managed Docker-backed embeddings use distinct retrieval input prefixes."""
    config = EmbeddingConfig(
        model_id="embedding_nomic_v2_moe",
        model="nomic-ai/nomic-embed-text-v2-moe",
        dimension=3,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    lease = _FakeLocalLease((1.0, 0.0, 0.0))
    embedder = ManagedLocalEmbedder(config, _FakeLocalManager(lease))

    await embedder.embed_documents(["document text"])
    await embedder.embed_query("question text")

    assert lease.calls == [
        ["search_document: document text"],
        ["search_query: question text"],
    ]


@pytest.mark.parametrize(
    "model_id",
    (
        "embedding_001",
        "embedding_gemma_300m",
        "embedding_nomic_v2_moe",
        "embedding_bge_m3",
    ),
)
async def test_registered_local_models_produce_contract_sized_vectors(
    model_id: str,
) -> None:
    """Every registered local model reaches the managed embedding contract."""
    config = load_model_catalog().embedding(model_id)
    lease = _FakeLocalLease(tuple(0.0 for _ in range(config.dimension)))
    embedder = build_embedder(config, _FakeLocalManager(lease))

    document_vectors = await embedder.embed_documents(["document"])
    query_vector = await embedder.embed_query("question")

    assert len(document_vectors) == 1
    assert document_vectors[0].dimension == config.dimension
    assert query_vector.dimension == config.dimension
    assert lease.calls == [
        [f"{config.document_prefix}document"],
        [f"{config.query_prefix}question"],
    ]


async def test_runner_uses_request_embedding_override() -> None:
    """A structured embedding override reaches query-time vector encoding."""
    default_config = EmbeddingConfig(
        type="fake",
        provider="local",
        model="shared-space",
        embedding_space_id="shared-fake-v1",
        dimension=3,
    )
    override_config = EmbeddingConfig(
        type="fake",
        provider="remote",
        model="shared-space",
        embedding_space_id="shared-fake-v1",
        dimension=3,
    )
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
    assert (
        vector.last_query_embedding.model_identity == override_config.vector_identity()
    )


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
        "llm_grounding.embedding.openai_compatible.resolve_api_key",
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


async def test_openai_compatible_embedder_batches_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote adapter bounds request size and retains batch order."""
    calls: list[list[str]] = []

    class FakeEmbeddings:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            texts = kwargs["input"]
            assert isinstance(texts, list)
            calls.append(texts)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=index, embedding=[float(index + 1), 0.0])
                    for index, _ in enumerate(reversed(texts))
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(
        "llm_grounding.embedding.openai_compatible.AsyncOpenAI", FakeClient
    )
    monkeypatch.setattr(
        "llm_grounding.embedding.openai_compatible.resolve_api_key",
        lambda: "test-key",
    )
    config = EmbeddingConfig(
        type="openai_compatible",
        model="embedding-model",
        dimension=2,
        batch_size=2,
        base_url="https://example.test/v1",
    )

    vectors = await build_embedder(config).embed_documents(["one", "two", "three"])

    assert calls == [["one", "two"], ["three"]]
    assert len(vectors) == 3
