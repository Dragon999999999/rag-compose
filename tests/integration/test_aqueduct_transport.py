"""Exercise the production provider stack against an in-memory HTTP transport."""

import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import models

from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.infrastructure.llm import aqueduct
from llm_grounding.infrastructure.llm.aqueduct import (
    AqueductModelClient,
    ModelRequestError,
)
from llm_grounding.reasoning.basic_summary import BasicSummaryPayload


async def test_real_provider_serializes_schema_and_closes_client(
    config: ArchitectureConfig,
    payload: BasicSummaryPayload,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the exact locked SDK/provider stack without a network connection."""
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Return a simulated OpenAI-compatible structured output tool response."""
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        tool_name = body["tools"][0]["function"]["name"]
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 1,
                "model": config.reasoner.model.name,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "test-tool",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": payload.model_dump_json(),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond), trust_env=False
    )
    settings: dict[str, object] = {}

    def create_client(**kwargs: object) -> AsyncOpenAI:
        """Capture adapter settings and inject the local transport into the real SDK."""
        settings.update(kwargs)
        return AsyncOpenAI(**kwargs, http_client=http_client)

    monkeypatch.setenv("AQUEDUCT_API_KEY", "test-key")
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    monkeypatch.setattr(aqueduct, "AsyncOpenAI", create_client)
    result = await AqueductModelClient(
        config.reasoner.model.model_copy(update={"timeout_seconds": 2, "retries": 0})
    ).generate("test paper", BasicSummaryPayload)
    assert result == payload
    assert len(requests) == 1
    assert requests[0]["model"] == config.reasoner.model.name
    schema = requests[0]["tools"][0]["function"]["parameters"]
    assert schema["properties"]["keywords"]["minItems"] == 5
    assert schema["properties"]["summary"]["maxLength"] == 500
    assert settings["max_retries"] == 0
    assert settings["timeout"] == 2
    assert http_client.is_closed


async def test_permanent_http_error_is_not_retried_or_exposed(
    config: ArchitectureConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authentication failure surfaces safely and closes the client after one call."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Return a permanent failure with a body that must remain out of messages."""
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "sensitive-body"}})

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond), trust_env=False
    )
    monkeypatch.setenv("AQUEDUCT_API_KEY", "test-key")
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    monkeypatch.setattr(
        aqueduct,
        "AsyncOpenAI",
        lambda **kwargs: AsyncOpenAI(**kwargs, http_client=http_client),
    )
    with pytest.raises(ModelRequestError, match="check credentials") as caught:
        await AqueductModelClient(
            config.reasoner.model.model_copy(
                update={"timeout_seconds": 2, "retries": 0}
            )
        ).generate("paper", BasicSummaryPayload)
    assert len(requests) == 1
    assert "sensitive-body" not in str(caught.value)
    assert http_client.is_closed
