"""Aqueduct's OpenAI-compatible structured-generation adapter using PydanticAI.

Each invocation resolves credentials, owns and closes an async client, and bounds
HTTP retries and schema-repair retries independently. An overall deadline bounds
the complete generation, including retries. Error messages exclude response bodies
and credentials while preserving causes for programmatic debugging. Model aliases
may change remotely, so identical inputs and settings do not guarantee replay.
"""

import asyncio

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError, ModelAPIError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from llm_grounding.infrastructure.config.models import ModelConfig
from llm_grounding.infrastructure.llm.credentials import resolve_api_key
from llm_grounding.infrastructure.llm.protocol import GenerationResult
from llm_grounding.observability.models import LLMUsage, UsageOrigin


class ModelRequestError(RuntimeError):
    """Report a failed generation without disclosing vendor response content."""


class AqueductModelClient:
    """Construct provider objects only when structured generation is requested."""

    def __init__(self, config: ModelConfig) -> None:
        """Retain credential-free model configuration without opening connections."""
        self._config = config

    async def generate[OutputT: BaseModel](
        self,
        prompt: str,
        output_type: type[OutputT],
        *,
        model_name: str | None = None,
    ) -> GenerationResult[OutputT]:
        """Call Aqueduct and return validated output, closing the client on every exit.

        Raises:
            RuntimeError: If Aqueduct credentials are unavailable.
            ModelRequestError: If the deadline, network call, or validation fails.
        """
        api_key = resolve_api_key(
            self._config.api_key_env,
            self._config.api_key_file_env,
        )
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with AsyncOpenAI(
                    api_key=api_key,
                    base_url=str(self._config.base_url),
                    timeout=self._config.timeout_seconds,
                    max_retries=self._config.retries,
                ) as client:
                    model = OpenAIChatModel(
                        model_name or self._config.name,
                        provider=OpenAIProvider(openai_client=client),
                    )
                    agent = Agent(
                        model, output_type=output_type, retries=self._config.retries
                    )
                    result = await agent.run(prompt)
                    usage = result.usage
                    request_tokens = getattr(usage, "input_tokens", None)
                    response_tokens = getattr(usage, "output_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)
                    cost_method = getattr(result, "cost", None)
                    cost = cost_method() if callable(cost_method) else None
                    return GenerationResult(
                        output=output_type.model_validate(result.output),
                        usage=LLMUsage(
                            model_id=model_name or self._config.name,
                            input_tokens=request_tokens,
                            output_tokens=response_tokens,
                            total_tokens=total_tokens,
                            requests=1,
                            usage_origin=UsageOrigin.PROVIDER_REPORTED,
                            metadata={
                                "details": getattr(usage, "details", {}) or {},
                                "cost": str(cost) if cost is not None else None,
                            },
                        ),
                        model_id=model_name or self._config.name,
                    )
        except TimeoutError as exc:
            raise ModelRequestError(
                "Aqueduct request exceeded its configured deadline."
            ) from exc
        except (APIError, ModelAPIError) as exc:
            raise ModelRequestError(
                "Aqueduct request failed; check credentials, endpoint, "
                "and model availability."
            ) from exc
        except (AgentRunError, ValidationError) as exc:
            raise ModelRequestError(
                "Aqueduct could not produce valid structured output "
                "within the retry limit."
            ) from exc
