"""Tests for the lazy Sentence Transformers embedding adapter."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from llm_grounding.embedding.sentence_transformers import (
    SentenceTransformersEmbedder,
)
from llm_grounding.ingestion.models import EmbeddingConfig


class _FakeSentenceTransformer:
    """Record query/document encoding without loading external model artifacts."""

    def __init__(self) -> None:
        """Initialize call counters."""
        self.query_calls = 0
        self.document_calls = 0

    def encode_query(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        """Return one correctly sized vector per query."""
        self.query_calls += 1
        return [[0.0] * 384 for _ in texts]

    def encode_document(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        """Return one correctly sized vector per document."""
        self.document_calls += 1
        return [[0.0] * 384 for _ in texts]


def _install_fake_package(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeSentenceTransformer, list[dict[str, object]]]:
    """Install a recording optional dependency module for one test."""
    model = _FakeSentenceTransformer()
    constructor_calls: list[dict[str, object]] = []

    def create_model(model_name: str, **kwargs: object) -> object:
        constructor_calls.append({"model_name": model_name, **kwargs})
        return model

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = create_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    return model, constructor_calls


async def test_query_loading_is_cache_only_and_reuses_in_memory_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries avoid Hub access and reuse one model during an application lifetime."""
    model, constructor_calls = _install_fake_package(monkeypatch)
    embedder = SentenceTransformersEmbedder(EmbeddingConfig(revision="model-revision"))
    run_sync = AsyncMock(
        side_effect=lambda function, *args, **kwargs: function(*args, **kwargs)
    )
    monkeypatch.setattr(
        "llm_grounding.embedding.sentence_transformers.asyncio.to_thread", run_sync
    )

    await embedder.embed_query("first question")
    await embedder.embed_query("second question")

    assert constructor_calls == [
        {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "model-revision",
            "local_files_only": True,
        }
    ]
    assert model.query_calls == 2
    assert model.document_calls == 0


async def test_document_loading_can_populate_empty_model_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingestion may download model artifacts when the local cache is empty."""
    model, constructor_calls = _install_fake_package(monkeypatch)
    embedder = SentenceTransformersEmbedder(EmbeddingConfig())
    run_sync = AsyncMock(
        side_effect=lambda function, *args, **kwargs: function(*args, **kwargs)
    )
    monkeypatch.setattr(
        "llm_grounding.embedding.sentence_transformers.asyncio.to_thread", run_sync
    )

    await embedder.embed_documents(["document chunk"])

    assert constructor_calls[0]["local_files_only"] is False
    assert "token" not in constructor_calls[0]
    assert model.document_calls == 1
    assert model.query_calls == 0
