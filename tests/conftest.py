"""Shared offline fixtures; no automated test may contact an external model."""

import socket
from pathlib import Path

import pymupdf
import pytest
from pydantic import BaseModel
from pydantic_ai import models

from llm_grounding.infrastructure.config.loader import load_architecture_config
from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.reasoning.basic_summary import BasicSummaryPayload


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
    """Load the checked-in architecture used by the actual application."""
    return load_architecture_config(Path("config/architectures/baseline_v0.yaml"))


@pytest.fixture
def payload() -> BasicSummaryPayload:
    """Supply a valid deterministic extraction without claiming model quality."""
    return BasicSummaryPayload(
        keywords=["grounding", "evidence", "research", "extraction", "validation"],
        summary="The paper describes structured extraction. It establishes a baseline.",
    )


class RecordingModelClient:
    """Record prompts and validate fixture output through the requested schema."""

    def __init__(self, payload: BasicSummaryPayload) -> None:
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
def model_client(payload: BasicSummaryPayload) -> RecordingModelClient:
    """Create a fresh deterministic generation dependency for each test."""
    return RecordingModelClient(payload)
