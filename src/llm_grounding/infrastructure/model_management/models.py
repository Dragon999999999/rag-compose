"""Value objects exchanged by local model lifecycle services."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from llm_grounding.embedding.models import SparseEmbedding, TokenEmbedding
from llm_grounding.infrastructure.config.declarative import ModelDefinition


@dataclass(frozen=True)
class ModelEndpoint:
    """Address of a ready local inference server."""

    model_id: str
    base_url: str
    container_name: str


@dataclass(frozen=True)
class ModelStatus:
    """Operational state of one registered local model."""

    model_id: str
    location: str
    weights_available: bool
    runtime_type: str | None
    running: bool
    active_leases: int = 0
    endpoint: ModelEndpoint | None = None
    detail: str | None = None


@dataclass(frozen=True)
class LocalModelHandle:
    """Lease-scoped handle for a ready local OpenAI-compatible inference server."""

    definition: ModelDefinition
    endpoint: ModelEndpoint
    client: "LocalInferenceClient"

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        """Create embeddings through the model's local inference endpoint."""
        response = await self.client.request(
            "/v1/embeddings", {"model": self.definition.name, "input": texts}
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Local embedding server returned no data list")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return tuple(
            tuple(float(value) for value in item["embedding"]) for item in ordered
        )

    async def rerank(
        self, query: str, documents: list[str]
    ) -> tuple[tuple[int, float], ...]:
        """Rerank documents through a compatible local ``/rerank`` endpoint."""
        response = await self.client.request(
            "/rerank",
            {"model": self.definition.name, "query": query, "texts": documents},
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Local reranking server returned no results list")
        return tuple(
            (int(item["index"]), float(item["relevance_score"])) for item in results
        )

    async def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        """Generate text through the local chat-completions-compatible endpoint."""
        response = await self.client.request(
            "/v1/chat/completions",
            {
                "model": self.definition.name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
            },
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Local generation server returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Local generation server returned no text content")
        return message["content"]

    async def sparse_embed(self, texts: list[str]) -> tuple[SparseEmbedding, ...]:
        """Create learned sparse vectors through the configured model endpoint.

        The endpoint is deliberately configurable because TEI-compatible model
        servers expose sparse outputs under different paths.  Responses may use
        either ``indices``/``values`` or an ``embedding`` mapping.
        """
        path = str(
            self.definition.parameter("sparse_endpoint", "/v1/sparse_embeddings")
        )
        response = await self.client.request(
            path,
            {"model": self.definition.name, "input": texts},
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Local sparse model returned no data list")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[SparseEmbedding] = []
        for item in ordered:
            payload = item.get("embedding", item)
            if not isinstance(payload, dict):
                raise RuntimeError("Local sparse model returned an invalid vector")
            indices = payload.get("indices")
            weights = payload.get("values", payload.get("weights"))
            if not isinstance(indices, list) or not isinstance(weights, list):
                raise RuntimeError("Local sparse model returned no indices and weights")
            vectors.append(
                SparseEmbedding(
                    tuple(int(index) for index in indices),
                    tuple(float(weight) for weight in weights),
                    self.definition.id,
                )
            )
        return tuple(vectors)

    async def token_embed(self, texts: list[str]) -> tuple[TokenEmbedding, ...]:
        """Create token-level vectors through a configured late-interaction endpoint."""
        path = str(self.definition.parameter("token_endpoint", "/v1/token_embeddings"))
        response = await self.client.request(
            path,
            {"model": self.definition.name, "input": texts},
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Local token model returned no data list")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[TokenEmbedding] = []
        for item in ordered:
            values = item.get("embedding", item.get("token_embeddings"))
            if not isinstance(values, list):
                raise RuntimeError("Local token model returned no token embeddings")
            offsets_value = item.get("token_offsets", [])
            offsets = (
                tuple((int(pair[0]), int(pair[1])) for pair in offsets_value)
                if isinstance(offsets_value, list)
                else ()
            )
            vectors.append(
                TokenEmbedding(
                    tuple(tuple(float(value) for value in token) for token in values),
                    self.definition.id,
                    offsets,
                )
            )
        return tuple(vectors)


class LocalInferenceClient:
    """Small async JSON client shared by lease-scoped local model handles."""

    def __init__(
        self,
        base_url: str,
        handler: Callable[[str, dict[str, object]], dict[str, object]] | None = None,
    ) -> None:
        """Store an HTTP endpoint or an in-process request handler."""
        self._base_url = base_url.rstrip("/")
        self._handler = handler

    async def request(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """POST one JSON request without adding a provider-specific dependency."""
        if self._handler is not None:
            return await asyncio.to_thread(self._handler, path, payload)
        import json
        from urllib.request import Request, urlopen

        def send() -> dict[str, object]:
            request = Request(
                f"{self._base_url}/{path.lstrip('/')}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=120) as response:  # noqa: S310
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError(
                    "Local inference server returned a non-object JSON value"
                )
            return value

        return send()
