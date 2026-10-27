"""Value objects exchanged by local model lifecycle services."""

from dataclasses import dataclass

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


class LocalInferenceClient:
    """Small async JSON client shared by lease-scoped local model handles."""

    def __init__(self, base_url: str) -> None:
        """Store the endpoint and defer all I/O until a model method is called."""
        self._base_url = base_url.rstrip("/")

    async def request(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """POST one JSON request without adding a provider-specific dependency."""
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
