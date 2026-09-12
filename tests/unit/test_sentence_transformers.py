"""Tests for the lazy Sentence Transformers embedding adapter."""

import sys
from types import ModuleType

import pytest

from llm_grounding.embedding.sentence_transformers import (
    SentenceTransformersEmbedder,
)
from llm_grounding.ingestion.models import EmbeddingConfig


def test_model_loading_permits_first_use_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model loading allows Sentence Transformers to populate an empty cache."""
    constructor_arguments: dict[str, object] = {}
    expected_model = object()

    def create_model(model_name: str, **kwargs: object) -> object:
        constructor_arguments["model_name"] = model_name
        constructor_arguments.update(kwargs)
        return expected_model

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = create_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    embedder = SentenceTransformersEmbedder(EmbeddingConfig(revision="model-revision"))

    assert embedder._model is expected_model
    assert constructor_arguments == {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "model-revision",
        "local_files_only": False,
    }
