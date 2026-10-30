"""Verify the real adapter's schema and error behavior without external requests."""

from unittest.mock import AsyncMock

import pytest
from conftest import ProviderPayload
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.infrastructure.llm import aqueduct
from llm_grounding.infrastructure.llm.aqueduct import (
    AqueductModelClient,
    ModelRequestError,
)


async def test_adapter_returns_validated_pydantic_output(
    config: ArchitectureConfig,
    payload: ProviderPayload,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter drives a PydanticAI agent and returns the requested schema."""
    monkeypatch.setenv("AQUEDUCT_API_KEY", "test-key")
    monkeypatch.setattr(
        aqueduct,
        "OpenAIChatModel",
        lambda *args, **kwargs: TestModel(custom_output_args=payload.model_dump()),
    )
    result = await AqueductModelClient(config.reasoner.model).generate(
        "paper", ProviderPayload
    )
    assert result == payload


async def test_invalid_structured_output_exhausts_bounded_retries(
    config: ArchitectureConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema-invalid model output becomes an actionable sanitized adapter failure."""
    monkeypatch.setenv("AQUEDUCT_API_KEY", "test-key")
    monkeypatch.setattr(
        aqueduct,
        "OpenAIChatModel",
        lambda *args, **kwargs: TestModel(
            custom_output_args={"keywords": [], "summary": "short"}
        ),
    )
    with pytest.raises(ModelRequestError, match="valid structured output") as caught:
        await AqueductModelClient(config.reasoner.model).generate(
            "paper", ProviderPayload
        )
    assert isinstance(caught.value.__cause__, UnexpectedModelBehavior)


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("sensitive response"), UnexpectedModelBehavior("sensitive response")],
)
async def test_adapter_sanitizes_failures_and_preserves_causes(
    config: ArchitectureConfig, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Timeout and model errors retain diagnostic causes without echoing raw bodies."""
    monkeypatch.setenv("AQUEDUCT_API_KEY", "test-key")
    monkeypatch.setattr(aqueduct.Agent, "run", AsyncMock(side_effect=failure))
    with pytest.raises(ModelRequestError) as caught:
        await AqueductModelClient(config.reasoner.model).generate(
            "paper", ProviderPayload
        )
    assert caught.value.__cause__ is failure
    assert "sensitive" not in str(caught.value)
    assert "test-key" not in str(caught.value)
