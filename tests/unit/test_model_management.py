"""Verify standardized model definitions and local lifecycle boundaries."""

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.config.loader import load_model_registry
from llm_grounding.infrastructure.model_management.manager import LocalModelManager
from llm_grounding.infrastructure.model_management.models import ModelEndpoint
from llm_grounding.infrastructure.model_management.registry import ModelRegistry
from llm_grounding.infrastructure.model_management.storage import ModelStorageManager


def test_qwen_reranker_accepts_batch_encoding_from_chat_template() -> None:
    """Generative reranking handles Transformers BatchEncoding inputs."""
    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    from llm_grounding.infrastructure.model_management.runtime import (
        _TransformersModel,
    )

    definition = load_model_registry().resolve("reranker_qwen3_0_6b").model_copy(
        update={
            "parameters": {
                "reranker_type": "generative",
                "yes_token_id": 1,
                "no_token_id": 0,
            }
        }
    )

    class _Tokenizer:
        def apply_chat_template(
            self, *_args: object, **kwargs: object
        ) -> BatchEncoding:
            assert kwargs["return_dict"] is True
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[4, 5]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }
            )

    class _Model:
        def __call__(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(logits=torch.tensor([[[0.0, 2.0]]]))

    model = _TransformersModel(definition, _Tokenizer(), _Model(), "cpu")

    result = model._qwen_score("What is this?", "A document")

    assert result > 0.5


def test_colbert_token_runtime_applies_checkpoint_projection(tmp_path: Path) -> None:
    """ColBERT token outputs use the checkpoint's 768-to-128 projection."""
    import torch

    from llm_grounding.infrastructure.model_management.runtime import (
        _TransformersModel,
    )

    definition = load_model_registry().resolve("embedding_colbert_v2")

    class _Tokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, object]:
            assert texts == ["document"]
            if kwargs.get("return_offsets_mapping"):
                return {
                    "offset_mapping": torch.tensor([[[0, 4], [4, 8]]]),
                }
            return {
                "input_ids": torch.tensor([[4, 5]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class _Model:
        def __call__(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                last_hidden_state=torch.tensor([[[3.0, 4.0, 0.0], [0.0, 4.0, 3.0]]])
            )

    projection = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    from safetensors.torch import save_file

    save_file({"linear.weight": projection}, str(tmp_path / "model.safetensors"))
    loaded_projection = _TransformersModel._load_token_projection(
        tmp_path, _Model()
    )
    assert torch.equal(loaded_projection, projection)

    model = _TransformersModel(
        definition, _Tokenizer(), _Model(), "cpu", loaded_projection
    )

    response = model.request("/v1/token_embeddings", {"input": ["document"]})
    values = response["data"][0]["token_embeddings"]

    assert values[0] == pytest.approx([0.6, 0.8])
    assert values[1] == pytest.approx([0.0, 1.0])


def test_model_registry_validates_standard_sources_and_excludes_schema() -> None:
    """Canonical model files expose location and nested source/runtime metadata."""
    registry = load_model_registry()

    assert len(registry.list()) == 16
    embedding = registry.resolve("embedding_001")
    assert embedding.location == "local"
    assert embedding.source is not None
    assert embedding.source.type == "huggingface"
    assert embedding.runtime is not None
    assert embedding.runtime.type == "docker"
    assert {
        registry.resolve(model_id).name
        for model_id in (
            "embedding_001",
            "embedding_nomic_v2_moe",
            "embedding_bge_m3",
            "embedding_jina_v2_base_en",
        )
    } == {
        "sentence-transformers/all-MiniLM-L6-v2",
        "nomic-ai/nomic-embed-text-v2-moe",
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v2-base-en",
    }
    assert registry.resolve("qwen3_1_7b").name == "Qwen/Qwen3-1.7B"
    assert registry.resolve("gliner_small").backend == "gliner"
    assert registry.resolve("embedding_nomic_v2_moe").runtime.command[-1] == "8000"
    assert registry.resolve("reasoning_002").location == "remote"
    assert {
        registry.resolve(model_id).name
        for model_id in (
            "reranker_bge_v2_m3",
            "reranker_qwen3_0_6b",
            "reranker_mxbai_base_v2",
            "reranker_jina_v2_multilingual",
        )
    } == {
        "BAAI/bge-reranker-v2-m3",
        "Qwen/Qwen3-Reranker-0.6B",
        "mixedbread-ai/mxbai-rerank-base-v2",
        "jinaai/jina-reranker-v2-base-multilingual",
    }


@pytest.mark.asyncio
async def test_storage_downloads_once_and_reuses_verified_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete URL installation is reused without a second source request."""
    registry = load_model_registry()
    original = registry.resolve("embedding_001")
    payload = b"test model weights"
    source = original.source
    assert source is not None
    definition = original.model_copy(
        update={
            "source": source.model_copy(
                update={
                    "type": "url",
                    "download_url": "https://models.example.test/model.bin",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "repository": None,
                }
            ),
            "storage": original.storage.model_copy(
                update={"relative_path": "embedding/test-model"}
            ),
        }
    )
    calls = 0

    def download(_url: str, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        Path(destination).write_bytes(payload)

    monkeypatch.setattr("urllib.request.urlretrieve", download)
    storage = ModelStorageManager(tmp_path)

    first = await storage.ensure_available(definition)
    second = await storage.ensure_available(definition)

    assert first == second
    assert calls == 1
    assert (first / ".model_complete").is_file()


class _FakeStorage:
    """Avoid filesystem work when testing lease sharing."""

    async def ensure_available(self, _definition: object) -> Path:
        """Return a stable test path."""
        return Path("/tmp/model")

    def is_available(self, _definition: object) -> bool:
        """Report test weights as available."""
        return True


@pytest.mark.asyncio
async def test_manager_prepare_downloads_weights_without_starting_runtime(
    tmp_path: Path,
) -> None:
    """Explicit preparation makes weights available without creating a runtime."""
    registry = load_model_registry()
    storage = ModelStorageManager(tmp_path)
    manager = LocalModelManager(registry, storage)
    calls = 0

    async def fake_ensure(model_definition: ModelDefinition) -> Path:
        nonlocal calls
        calls += 1
        assert model_definition.id == "gliner_small"
        return tmp_path / "gliner_small"

    storage.ensure_available = fake_ensure  # type: ignore[method-assign]

    assert await manager.prepare("gliner_small") == tmp_path / "gliner_small"
    assert calls == 1
    await manager.close()


class _FakeRuntime:
    """Record lifecycle calls made by LocalModelManager."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    async def start_and_wait(
        self, definition: ModelDefinition, _weights: Path
    ) -> ModelEndpoint:
        """Return a deterministic ready endpoint."""
        self.starts += 1
        return ModelEndpoint(definition.id, "http://127.0.0.1:8000", "test")

    async def status(
        self, _definition: object
    ) -> tuple[bool, ModelEndpoint | None, str | None]:
        """Return an idle fake runtime state."""
        return False, None, "stopped"

    async def stop(self, _definition: object) -> None:
        """Record a stop without external side effects."""
        self.stops += 1

    async def restart(self, _definition: object) -> None:
        """Provide the manager lifecycle contract for the test."""

    async def remove(self, _definition: object) -> None:
        """Provide the manager lifecycle contract for the test."""


@pytest.mark.asyncio
async def test_manager_shares_one_runtime_for_concurrent_leases() -> None:
    """Concurrent callers share one started model and stop only after release."""
    definition = load_model_registry().resolve("embedding_001")
    runtime_config = definition.runtime
    assert runtime_config is not None
    definition = definition.model_copy(
        update={
            "runtime": runtime_config.model_copy(update={"idle_timeout_seconds": 0})
        }
    )
    registry = ModelRegistry({definition.id: definition})
    runtime = _FakeRuntime()
    manager = LocalModelManager(
        registry,
        cast(ModelStorageManager, _FakeStorage()),
        cast(object, runtime),
    )

    async with manager.acquire(definition.id):
        async with manager.acquire(definition.id):
            assert runtime.starts == 1
            assert (await manager.status(definition.id)).active_leases == 2
    await asyncio.sleep(0)
    assert runtime.stops == 1
