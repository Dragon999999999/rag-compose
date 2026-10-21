"""Test configurable candidate truncation and local cross-encoder reranking."""

from pathlib import Path

import pytest

from llm_grounding.domain.corpus import (
    CorpusRequest,
    EvidenceReference,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, RetrievalResult
from llm_grounding.embedding.fake import DeterministicFakeEmbedder
from llm_grounding.infrastructure.config.models import RerankerConfig
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.retrieval.models import RetrievalResponse
from llm_grounding.retrieval.reranking.cross_encoder import CrossEncoderReranker
from llm_grounding.retrieval.reranking.top_k import TopKReranker
from llm_grounding.retrieval.vector import VectorRetriever


def _response() -> RetrievalResponse:
    """Create candidates with identifiers and metadata that must survive reranking."""
    results = tuple(
        RetrievalResult(
            EvidenceUnit(
                evidence_id=f"evidence-{index}",
                document_id="document",
                evidence_type=EvidenceType.TEXT,
                source_uri="file:///paper.pdf",
                order=index,
                text=text,
                metadata={"source": f"block-{index}"},
            ),
            index,
            "vector",
            score,
            query_variant="original",
            metadata={"candidate": str(index)},
        )
        for index, (text, score) in enumerate(
            (("first passage", 0.9), ("second passage", 0.8), ("third passage", 0.7)),
            1,
        )
    )
    return RetrievalResponse("the question", results)


async def test_top_k_preserves_existing_order_identifiers_and_metadata() -> None:
    """Top-k mode truncates without changing the retrieval ranking semantics."""
    response = await TopKReranker(2).rerank(_response())

    assert [item.evidence.evidence_id for item in response.results] == [
        "evidence-1",
        "evidence-2",
    ]
    assert [item.rank for item in response.results] == [1, 2]
    assert response.results[1].metadata == {"candidate": "2"}
    assert response.results[1].query_variant == "original"


def test_reranker_configuration_accepts_provider_alias_and_validates_limits() -> None:
    """New provider syntax normalizes to the existing component type field."""
    config = RerankerConfig(
        provider="cross_encoder",
        retrieval_top_k=20,
        rerank_top_k=5,
        device="cpu",
        batch_size=16,
    )

    assert config.type == "cross_encoder"
    assert config.model_dump()["type"] == "cross_encoder"

    with pytest.raises(ValueError, match="must not exceed"):
        RerankerConfig(provider="top_k", retrieval_top_k=2, rerank_top_k=3)

    with pytest.raises(ValueError, match="requires"):
        RerankerConfig(type="identity", rerank_top_k=2)


def test_reranking_yaml_section_alias_is_accepted(config) -> None:
    """The descriptive ``reranking`` section remains compatible with ``reranker``."""
    values = config.model_dump(mode="json")
    values.pop("reranker")
    values["reranking"] = {
        "provider": "top_k",
        "retrieval_top_k": 4,
        "rerank_top_k": 2,
    }

    updated = type(config).model_validate(values)

    assert updated.reranker.type == "top_k"
    assert updated.reranker.retrieval_top_k == 4


async def test_cross_encoder_loads_lazily_reuses_model_and_orders_results(
    tmp_path: Path,
) -> None:
    """Mocked local inference proves lazy construction, caching, and score ordering."""
    model_calls: list[dict[str, object]] = []

    class FakeModel:
        def predict(self, pairs: list[list[str]], **kwargs: object) -> list[float]:
            model_calls.append({"pairs": pairs, **kwargs})
            return [0.2, 0.9, 0.4]

    constructors: list[dict[str, object]] = []

    def factory(model_name: str, **kwargs: object) -> FakeModel:
        constructors.append({"model_name": model_name, **kwargs})
        return FakeModel()

    reranker = CrossEncoderReranker(
        batch_size=4,
        top_k=2,
        cache_dir=tmp_path / "models",
        model_factory=factory,
    )
    assert reranker.model is None

    first = await reranker.rerank(_response())
    second = await reranker.rerank(_response())

    assert len(constructors) == 1
    assert constructors[0]["model_name"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert constructors[0]["device"] == "cpu"
    assert constructors[0]["cache_folder"] == str(tmp_path / "models")
    assert len(model_calls) == 2
    assert model_calls[0]["batch_size"] == 4
    assert [item.evidence.evidence_id for item in first.results] == [
        "evidence-2",
        "evidence-3",
    ]
    assert [item.score for item in first.results] == [0.9, 0.4]
    assert first.results[0].metadata == {"candidate": "2"}
    assert [item.rank for item in second.results] == [1, 2]


async def test_cross_encoder_rejects_malformed_score_count(tmp_path: Path) -> None:
    """Provider output must contain exactly one finite score per candidate."""

    class BadModel:
        def predict(self, pairs: list[list[str]], **kwargs: object) -> list[float]:
            return [0.5]

    reranker = CrossEncoderReranker(
        cache_dir=tmp_path,
        model_factory=lambda *_args, **_kwargs: BadModel(),
    )

    with pytest.raises(RuntimeError, match="score count"):
        await reranker.rerank(_response())


async def test_basic_rag_applies_top_k_reranker_to_vector_candidates() -> None:
    """The specialized vector runner uses the shared reranker contract."""
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    chunks = tuple(
        RetrievalUnit(
            f"chunk-{index}",
            "document",
            "ingestion",
            "paper.pdf",
            f"passage {index}",
            index,
            1,
            1,
            (EvidenceReference(f"evidence-{index}", 0, 9),),
        )
        for index in range(3)
    )
    vectors = await embedder.embed_documents([item.text for item in chunks])
    await store.upsert("project", chunks, vectors)
    runner = BasicRagRunner(
        VectorRetriever(embedder, store, top_k=3),
        object(),
        object(),
        object(),
        object(),
        object(),
        reranker=TopKReranker(1),
    )

    results = await runner.retrieve(CorpusRequest("project", "passage"))

    assert len(results) == 1
    assert results[0].rank == 1
