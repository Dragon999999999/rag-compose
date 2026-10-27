"""Docker-backed local inference runtime lifecycle."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.model_management.models import (
    LocalInferenceClient,
    ModelEndpoint,
)

ReadinessProbe = Callable[[str], Awaitable[bool]]


class DockerModelRuntime:
    """Create, reuse, and stop Docker containers for local model servers."""

    def __init__(
        self, client: Any | None = None, probe: ReadinessProbe | None = None
    ) -> None:
        """Inject a Docker client and readiness probe for deterministic tests."""
        self._client = client
        self._probe = probe

    def _docker(self) -> Any:
        """Create the Docker SDK client only when a local model is requested."""
        if self._client is None:
            try:
                import docker  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "Install the Docker SDK to use local model runtimes"
                ) from exc
            self._client = docker.from_env()
        return self._client

    async def ensure_started(
        self, definition: ModelDefinition, weights: Path
    ) -> ModelEndpoint:
        """Start or reuse a matching container and wait for its readiness endpoint."""
        runtime = definition.runtime
        if runtime is None or runtime.type != "docker":
            raise RuntimeError(
                f"Model {definition.id} does not select a Docker runtime"
            )
        return self._ensure_started_sync(definition, weights)

    def _ensure_started_sync(
        self, definition: ModelDefinition, weights: Path
    ) -> ModelEndpoint:
        """Perform Docker SDK operations outside the event loop."""
        runtime = definition.runtime
        assert runtime is not None
        client = self._docker()
        name = self._container_name(definition)
        fingerprint = self._fingerprint(definition)
        container = None
        try:
            container = client.containers.get(name)
            labels = getattr(container, "labels", {}) or getattr(
                container, "attrs", {}
            ).get("Config", {}).get("Labels", {})
            if labels.get("llm_grounding.model_fingerprint") != fingerprint:
                container.remove(force=True)
                container = None
        except Exception:
            container = None
        if container is None:
            kwargs: dict[str, object] = {
                "name": name,
                "image": runtime.image,
                "command": list(runtime.command) or None,
                "detach": True,
                "labels": {
                    "llm_grounding.model_id": definition.id,
                    "llm_grounding.model_fingerprint": fingerprint,
                },
                "volumes": {
                    str(weights): {"bind": runtime.container_path, "mode": "ro"}
                },
                "ports": {
                    f"{runtime.port}/tcp": runtime.host_port,
                },
                "environment": runtime.environment,
            }
            if runtime.cpu_limit is not None:
                kwargs["nano_cpus"] = int(runtime.cpu_limit * 1_000_000_000)
            if runtime.memory_limit is not None:
                kwargs["mem_limit"] = runtime.memory_limit
            if runtime.gpu_count is not None:
                try:
                    from docker.types import (  # type: ignore[import-untyped]
                        DeviceRequest,
                    )
                except ImportError as exc:
                    raise RuntimeError("Docker GPU support is unavailable") from exc
                kwargs["device_requests"] = [
                    DeviceRequest(count=runtime.gpu_count, capabilities=[["gpu"]])
                ]
            container = client.containers.create(**kwargs)
        status = getattr(container, "status", None) or getattr(
            container, "attrs", {}
        ).get("State", {}).get("Status")
        if status != "running":
            container.start()
        reload_method = getattr(container, "reload", None)
        if callable(reload_method):
            reload_method()
        base_url = self._endpoint_url(container, runtime.host_port, runtime.port)
        return ModelEndpoint(definition.id, base_url, name)

    async def wait_ready(
        self, endpoint: ModelEndpoint, definition: ModelDefinition
    ) -> None:
        """Wait for the configured readiness endpoint before returning a lease."""
        runtime = definition.runtime
        if runtime is None:
            return
        deadline = asyncio.get_running_loop().time() + runtime.readiness_timeout_seconds
        while True:
            ready = (
                await self._probe(f"{endpoint.base_url}{runtime.readiness_path}")
                if self._probe is not None
                else await asyncio.to_thread(
                    self._probe_sync, f"{endpoint.base_url}{runtime.readiness_path}"
                )
            )
            if ready:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"Local model runtime did not become ready: {definition.id}"
                )
            await asyncio.sleep(0.5)

    async def start_and_wait(
        self, definition: ModelDefinition, weights: Path
    ) -> ModelEndpoint:
        """Start a container and wait until its HTTP endpoint responds."""
        endpoint = await self.ensure_started(definition, weights)
        await self.wait_ready(endpoint, definition)
        return endpoint

    async def status(
        self, definition: ModelDefinition
    ) -> tuple[bool, ModelEndpoint | None, str | None]:
        """Return container state without starting a stopped model."""
        runtime = definition.runtime
        if runtime is None or runtime.type != "docker":
            return False, None, "Model does not use Docker runtime"
        return self._status_sync(definition)

    def _status_sync(
        self, definition: ModelDefinition
    ) -> tuple[bool, ModelEndpoint | None, str | None]:
        """Inspect one named container through the Docker SDK."""
        runtime = definition.runtime
        assert runtime is not None
        try:
            container = self._docker().containers.get(self._container_name(definition))
        except Exception:
            return False, None, "Container does not exist"
        reload_method = getattr(container, "reload", None)
        if callable(reload_method):
            reload_method()
        attrs = getattr(container, "attrs", {})
        state = attrs.get("State", {})
        running = bool(
            state.get("Running", getattr(container, "status", None) == "running")
        )
        endpoint = (
            ModelEndpoint(
                definition.id,
                self._endpoint_url(container, runtime.host_port, runtime.port),
                self._container_name(definition),
            )
            if running
            else None
        )
        return (
            running,
            endpoint,
            None if running else str(state.get("Status", "stopped")),
        )

    async def stop(self, definition: ModelDefinition) -> None:
        """Stop a model container while retaining its downloaded weights."""
        self._container_action(definition, "stop")

    async def restart(self, definition: ModelDefinition) -> None:
        """Restart an existing model container without deleting its weights."""
        self._container_action(definition, "restart")

    async def remove(self, definition: ModelDefinition) -> None:
        """Remove a model container while retaining its model-storage directory."""
        self._container_action(definition, "remove")

    def _container_action(self, definition: ModelDefinition, action: str) -> None:
        """Apply one lifecycle action if the named container exists."""
        try:
            container = self._docker().containers.get(self._container_name(definition))
        except Exception:
            return
        getattr(container, action)(force=True) if action == "remove" else getattr(
            container, action
        )()

    @staticmethod
    def _container_name(definition: ModelDefinition) -> str:
        """Return a stable, Docker-safe name for one model ID."""
        safe = "".join(
            char if char.isalnum() or char in "-_" else "-" for char in definition.id
        )
        return f"llm-grounding-model-{safe}"

    @staticmethod
    def _fingerprint(definition: ModelDefinition) -> str:
        """Hash runtime configuration so stale containers are never reused."""
        payload = definition.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _endpoint_url(
        container: Any, host_port: int | None, container_port: int
    ) -> str:
        """Resolve the host endpoint from explicit or Docker-assigned port mappings."""
        if host_port is not None:
            return f"http://127.0.0.1:{host_port}"
        ports = (
            getattr(container, "attrs", {}).get("NetworkSettings", {}).get("Ports", {})
        )
        bindings = ports.get(f"{container_port}/tcp") or []
        if bindings:
            host = bindings[0].get("HostIp") or "127.0.0.1"
            return f"http://{host}:{bindings[0]['HostPort']}"
        return f"http://127.0.0.1:{container_port}"

    @staticmethod
    def _probe_sync(url: str) -> bool:
        """Perform a bounded readiness GET without exposing response content."""
        try:
            with urlopen(url, timeout=2):  # noqa: S310
                return True
        except OSError:
            return False


class InProcessModelRuntime:
    """Lazy Transformers runtime for sparse and token-level local models.

    Model weights are still prepared by ``ModelStorageManager``.  This runtime
    only owns the in-memory tokenizer/model object and releases it on idle stop;
    no model files are deleted.  The optional Transformers and PyTorch packages
    are imported only when a configured in-process model is first used.
    """

    def __init__(self) -> None:
        """Start with no loaded weights and no background worker."""
        self._models: dict[str, _TransformersModel] = {}

    async def ensure_started(
        self, definition: ModelDefinition, weights: Path
    ) -> ModelEndpoint:
        """Load one model once and return a lease endpoint marker."""
        if definition.runtime is None or definition.runtime.type != "in_process":
            raise RuntimeError(
                f"Model {definition.id} does not select in-process runtime"
            )
        if definition.id not in self._models:
            self._models[definition.id] = await asyncio.to_thread(
                _TransformersModel.load, definition, weights
            )
        return ModelEndpoint(definition.id, f"inprocess://{definition.id}", "")

    async def start_and_wait(
        self, definition: ModelDefinition, weights: Path
    ) -> ModelEndpoint:
        """Load the local model; loading itself is the readiness check."""
        return await self.ensure_started(definition, weights)

    def client_for(self, definition: ModelDefinition) -> LocalInferenceClient:
        """Return a request client backed by the loaded model."""
        try:
            model = self._models[definition.id]
        except KeyError as exc:
            raise RuntimeError(
                f"In-process model is not loaded: {definition.id}"
            ) from exc
        return LocalInferenceClient("inprocess://" + definition.id, model.request)

    async def status(
        self, definition: ModelDefinition
    ) -> tuple[bool, ModelEndpoint | None, str | None]:
        """Report whether the model object is resident without loading it."""
        running = definition.id in self._models
        endpoint = (
            ModelEndpoint(definition.id, f"inprocess://{definition.id}", "")
            if running
            else None
        )
        return running, endpoint, None if running else "not loaded"

    async def stop(self, definition: ModelDefinition) -> None:
        """Release model memory while retaining persistent weights."""
        model = self._models.pop(definition.id, None)
        if model is not None:
            model.close()

    async def restart(self, definition: ModelDefinition) -> None:
        """Release the loaded object; the next lease reloads it."""
        await self.stop(definition)

    async def remove(self, definition: ModelDefinition) -> None:
        """Remove only the in-memory model object."""
        await self.stop(definition)


class _TransformersModel:
    """Synchronous inference adapter used behind the async in-process runtime."""

    def __init__(self, definition: ModelDefinition, tokenizer: Any, model: Any) -> None:
        """Store loaded provider objects and model-specific configuration."""
        self._definition = definition
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def load(cls, definition: ModelDefinition, weights: Path) -> "_TransformersModel":
        """Load Transformers weights from the persistent model directory."""
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModel,
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install llm-grounding[embeddings] with Transformers and PyTorch "
                "to use in-process sparse or token models"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(weights),
            local_files_only=True,
            trust_remote_code=bool(getattr(definition, "trust_remote_code", False)),
        )
        model_loader = (
            AutoModelForMaskedLM
            if "splade" in definition.name.casefold()
            else AutoModel
        )
        model = model_loader.from_pretrained(
            str(weights),
            local_files_only=True,
            trust_remote_code=bool(getattr(definition, "trust_remote_code", False)),
        )
        model.eval()
        return cls(definition, tokenizer, model)

    def request(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """Execute the supported dense, sparse, or token representation request."""
        inputs = payload.get("input")
        if not isinstance(inputs, list) or not all(
            isinstance(item, str) for item in inputs
        ):
            raise ValueError("In-process model input must be a list of strings")
        if path.endswith("/sparse_embeddings"):
            return self._sparse(inputs)
        if path.endswith("/token_embeddings"):
            return self._tokens(inputs)
        if path.endswith("/embeddings"):
            return self._dense(inputs)
        raise RuntimeError(f"Unsupported in-process model endpoint: {path}")

    def _forward(self, texts: list[str]) -> tuple[Any, Any]:
        """Tokenize and run a bounded forward pass without retaining activations."""
        import torch  # type: ignore[import-not-found]

        tokenizer = self._tokenizer
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(cast(int, self._definition.parameter("max_length", 256))),
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self._model(**encoded)
        return output, encoded["attention_mask"]

    def _dense(self, texts: list[str]) -> dict[str, object]:
        """Mean-pool token states into the compatibility dense endpoint shape."""
        import torch

        output, mask = self._forward(texts)
        values = output.last_hidden_state
        weights = mask.unsqueeze(-1).to(values.dtype)
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        if bool(self._definition.parameter("normalize", True)):
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return {
            "data": [
                {"index": i, "embedding": row.tolist()} for i, row in enumerate(pooled)
            ]
        }

    def _tokens(self, texts: list[str]) -> dict[str, object]:
        """Return masked token embeddings for MaxSim retrieval."""
        import torch

        output, mask = self._forward(texts)
        values = output.last_hidden_state
        if bool(self._definition.parameter("normalize", True)):
            values = torch.nn.functional.normalize(values, p=2, dim=-1)
        return {
            "data": [
                {
                    "index": i,
                    "token_embeddings": [
                        row.tolist()
                        for row, keep in zip(values[i], mask[i], strict=True)
                        if bool(keep)
                    ],
                }
                for i in range(len(texts))
            ]
        }

    def _sparse(self, texts: list[str]) -> dict[str, object]:
        """Return SPLADE logits when available, with a safe token-weight fallback."""
        import torch

        output, mask = self._forward(texts)
        logits = getattr(output, "logits", None)
        lexical_weights = getattr(output, "lexical_weights", None)
        vectors: list[dict[str, object]] = []
        if lexical_weights is not None:
            for item in lexical_weights:
                if not isinstance(item, dict):
                    raise RuntimeError("Model returned invalid lexical weights")
                vectors.append(
                    {
                        "indices": [int(index) for index in item],
                        "values": [float(value) for value in item.values()],
                    }
                )
        elif logits is not None and len(logits.shape) == 3:
            values = torch.log1p(torch.relu(logits)).max(dim=1).values
            limit = int(cast(int, self._definition.parameter("sparse_top_k", 256)))
            for row in values:
                selected = torch.topk(row, min(limit, row.shape[0]))
                nonzero = selected.values > 0
                vectors.append(
                    {
                        "indices": selected.indices[nonzero].tolist(),
                        "values": selected.values[nonzero].tolist(),
                    }
                )
        else:
            # Models without a vocabulary projection (notably BGE-M3 through
            # plain Transformers) still receive deterministic lexical weights.
            # Provider-specific servers can replace this path with true lexical
            # weights through the same endpoint without changing retrieval code.
            token_ids = self._tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]
            for row, row_mask in zip(token_ids, mask, strict=True):
                selected = [
                    (int(token), 1.0)
                    for token, keep in zip(row, row_mask, strict=True)
                    if bool(keep)
                ]
                vectors.append(
                    {
                        "indices": [item[0] for item in selected],
                        "values": [item[1] for item in selected],
                    }
                )
        return {
            "data": [
                {"index": i, "embedding": vector} for i, vector in enumerate(vectors)
            ]
        }

    def close(self) -> None:
        """Release provider references and opportunistically clear CUDA memory."""
        self._tokenizer = None
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
