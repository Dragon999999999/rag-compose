"""Shared offline fixtures; no automated test may contact an external model."""

import socket
from pathlib import Path

import pymupdf
import pytest
from pydantic import BaseModel, Field
from pydantic_ai import models

from llm_grounding.infrastructure.config.loader import (
    load_ingestion_config,
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.models import ArchitectureConfig


class ProviderPayload(BaseModel):
    """Stable flat schema used only for provider transport tests."""

    keywords: list[str] = Field(min_length=5, max_length=8)
    summary: str = Field(min_length=20, max_length=500)


@pytest.fixture(autouse=True)
def disable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block network connects and real PydanticAI requests, and clear credentials."""
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.delenv("AQUEDUCT_API_KEY", raising=False)
    monkeypatch.delenv("AQUEDUCT_API_KEY_FILE", raising=False)

    def reject_connect(*args: object, **kwargs: object) -> None:
        """Fail immediately if a test accidentally tries to connect to a service."""
        raise AssertionError("Automated tests must not connect to the network")

    monkeypatch.setattr(socket.socket, "connect", reject_connect)
    monkeypatch.setattr(socket, "getaddrinfo", reject_connect)


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    """Create two PDF pages with block insertion order differing from reading order."""
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 200), "Lower block discusses the contribution.")
        page.insert_text((72, 72), "Upper block introduces the research.")
        page = document.new_page()
        page.insert_text((72, 72), "Second page concludes the paper.")
        document.save(path)
    return path


@pytest.fixture
def config() -> ArchitectureConfig:
    """Load the effective query configuration used by the actual application."""
    query = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
    return query.resolve_query(load_model_catalog(), load_ingestion_config())[0]


@pytest.fixture
def payload() -> ProviderPayload:
    """Supply a valid deterministic structured payload."""
    return ProviderPayload(
        keywords=["grounding", "evidence", "research", "validation", "retrieval"],
        summary="The paper describes structured grounding in indexed evidence.",
    )


class RecordingModelClient:
    """Record prompts and validate fixture output through the requested schema."""

    def __init__(self, payload: ProviderPayload) -> None:
        """Store fixed output and a fresh prompt history."""
        self.payload = payload
        self.prompts: list[str] = []

    async def generate[OutputT: BaseModel](
        self, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        """Return a validated instance of the caller's requested output model."""
        self.prompts.append(prompt)
        return output_type.model_validate(self.payload.model_dump())


@pytest.fixture
def model_client(payload: ProviderPayload) -> RecordingModelClient:
    """Create a fresh deterministic generation dependency for each test."""
    return RecordingModelClient(payload)
