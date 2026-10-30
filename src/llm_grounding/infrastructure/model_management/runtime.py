"""Docker-backed local inference runtime lifecycle."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.model_management.models import (
    LocalInferenceClient,
    ModelEndpoint,
)

ReadinessProbe = Callable[[str], Awaitable[bool]]
logger = logging.getLogger(__name__)


class DockerModelRuntime:
    """Create, reuse, and stop Docker containers for local model servers."""

    def __init__(
        self, client: Any | None = None, probe: ReadinessProbe | None = None
    ) -> None:
        """Inject a Docker client and readiness probe for deterministic tests."""
        self._client = client
        self._probe = probe
        self._image_locks: dict[str, asyncio.Lock] = {}

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
        image = runtime.image
        assert image is not None
        image_lock = self._image_locks.setdefault(image, asyncio.Lock())
        async with image_lock:
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
            self._ensure_image(client, runtime.image)
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
            try:
                container.start()
            except Exception as exc:
                diagnostics = self._container_diagnostics(client, name, container)
                logger.error(
                    "Model container %s failed to start: %s; %s",
                    name,
                    exc,
                    diagnostics,
                )
                raise RuntimeError(
                    f"Model container {name} failed to start: {diagnostics}"
                ) from exc
        reload_method = getattr(container, "reload", None)
        if callable(reload_method):
            reload_method()
        base_url = self._endpoint_url(container, runtime.host_port, runtime.port)
        return ModelEndpoint(definition.id, base_url, name)

    @staticmethod
    def _container_diagnostics(
        client: Any, name: str, container: Any | None = None
    ) -> str:
        """Return bounded startup state and logs for a failed model container."""
        try:
            current = container or client.containers.get(name)
            attrs = getattr(current, "attrs", {}) or {}
            state = attrs.get("State", {}) or {}
            status = getattr(current, "status", None) or state.get("Status", "unknown")
            exit_code = state.get("ExitCode")
            logs_method = getattr(current, "logs", None)
            raw_logs = logs_method(tail=100) if callable(logs_method) else b""
            if isinstance(raw_logs, bytes):
                logs = raw_logs.decode("utf-8", errors="replace")
            else:
                logs = str(raw_logs)
            logs = " ".join(logs.split())[-4000:]
            detail = f"status={status}"
            if exit_code is not None:
                detail += f", exit_code={exit_code}"
            if logs:
                detail += f", logs={logs}"
            return detail
        except Exception as exc:
            return f"container diagnostics unavailable: {type(exc).__name__}: {exc}"

    @staticmethod
    def _ensure_image(client: Any, image: str | None) -> None:
        """Ensure one Docker image is local before creating its container.

        Docker image lookup and pull errors are translated here so callers do not
        receive an opaque container-creation failure. The caller serializes this
        operation per image, which prevents concurrent model starts from pulling
        the same image more than once.
        """
        if not image:
            raise RuntimeError("Docker model runtime has no configured image")
        try:
            from docker.errors import ImageNotFound  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Install the Docker SDK to manage local model images"
            ) from exc

        images = getattr(client, "images", None)
        if images is None or not callable(getattr(images, "get", None)):
            raise RuntimeError("Docker client does not provide image management")
        try:
            images.get(image)
            return
        except ImageNotFound:
            pass
        except Exception as exc:
            raise RuntimeError(
                DockerModelRuntime._image_error_message("inspect", image, exc)
            ) from exc

        try:
            images.pull(image)
        except Exception as exc:
            raise RuntimeError(
                DockerModelRuntime._image_error_message("pull", image, exc)
            ) from exc

    @staticmethod
    def _image_error_message(operation: str, image: str, error: Exception) -> str:
        """Return a useful, credential-safe message for a Docker image error."""
        response = getattr(error, "response", None)
        status_code = getattr(error, "status_code", None) or getattr(
            response, "status_code", None
        )
        detail = str(error).strip() or type(error).__name__
        detail = " ".join(detail.split())[:300]
        lowered = detail.lower()
        error_name = type(error).__name__.lower()
        if "timeout" in error_name or "timed out" in lowered:
            category = "timed out"
        elif (
            status_code in {401, 403}
            or "unauthorized" in lowered
            or "authentication" in lowered
            or "access denied" in lowered
            or "forbidden" in lowered
        ):
            category = "was rejected by the registry (authentication or access)"
        elif (
            status_code is not None
            or "registry" in lowered
            or "manifest unknown" in lowered
            or "repository does not exist" in lowered
            or "not found" in lowered
        ):
            category = "was rejected by the registry"
        else:
            category = "failed"
        return f"Docker image {operation} for '{image}' {category}: {detail}"

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
                diagnostics = self._container_diagnostics(
                    self._docker(), endpoint.container_name
                )
                logger.error(
                    "Model container %s did not become ready: %s",
                    endpoint.container_name,
                    diagnostics,
                )
                raise TimeoutError(
                    f"Local model runtime did not become ready: {definition.id}; "
                    f"{diagnostics}"
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

    def __init__(
        self,
        definition: ModelDefinition,
        tokenizer: Any,
        model: Any,
        device: str,
        token_projection: Any | None = None,
    ) -> None:
        """Store loaded provider objects and model-specific configuration."""
        self._definition = definition
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._token_projection = token_projection

    @classmethod
    def load(cls, definition: ModelDefinition, weights: Path) -> "_TransformersModel":
        """Load Transformers weights from the persistent model directory."""
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Install llm-grounding[embeddings] with PyTorch to use local models"
            ) from exc
        if getattr(definition, "backend", "") == "gliner":
            try:
                from gliner import GLiNER  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "GLiNER metadata enrichment requires the optional gliner package"
                ) from exc
            model = GLiNER.from_pretrained(str(weights))
            model.eval()
            configured_device = (
                definition.runtime.device if definition.runtime else "cpu"
            )
            device = configured_device
            if configured_device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            try:
                model.to(device)
            except (RuntimeError, ValueError):
                if device == "cpu":
                    raise
                device = "cpu"
                model.to(device)
            return cls(definition, None, model, device)
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install llm-grounding[embeddings] with Transformers and PyTorch "
                "to use in-process local models"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(weights),
            local_files_only=True,
            trust_remote_code=bool(getattr(definition, "trust_remote_code", False)),
        )
        model_name = definition.name.casefold()
        reranker_kind = str(definition.parameter("reranker_type", ""))
        if (
            reranker_kind == "generative"
            or bool(definition.parameter("generation_model", False))
            or "qwen3-reranker" in model_name
        ):
            model_loader = AutoModelForCausalLM
        elif definition.stage == "reranker":
            model_loader = AutoModelForSequenceClassification
        else:
            model_loader = AutoModel
        model = model_loader.from_pretrained(
            str(weights),
            local_files_only=True,
            trust_remote_code=bool(getattr(definition, "trust_remote_code", False)),
        )
        token_projection = None
        if definition.parameter("representation") == "multivector":
            token_projection = cls._load_token_projection(weights, model)
        model.eval()
        configured_device = definition.runtime.device if definition.runtime else "cpu"
        device = configured_device
        if configured_device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        try:
            model.to(device)
        except (RuntimeError, ValueError):
            if device == "cpu":
                raise
            device = "cpu"
            model.to(device)
        return cls(definition, tokenizer, model, device, token_projection)

    @staticmethod
    def _load_token_projection(weights: Path, model: Any) -> Any:
        """Load the checkpoint projection used by late-interaction encoders.

        ColBERT-v2 stores a 768-to-128 projection alongside the BERT weights.
        ``AutoModel`` loads the BERT base but does not expose that additional
        checkpoint tensor, so the runtime must restore it before indexing or
        querying token vectors.
        """
        import torch

        projection_module = getattr(model, "linear", None)
        projection_weight = getattr(projection_module, "weight", None)
        if projection_weight is not None:
            return projection_weight.detach().float()

        state: Mapping[str, Any] | None = None
        safetensors_path = weights / "model.safetensors"
        if safetensors_path.is_file():
            try:
                from safetensors.torch import load_file

                state = load_file(str(safetensors_path), device="cpu")
            except (ImportError, OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "Unable to load the ColBERT token projection from model.safetensors"
                ) from exc
        else:
            pytorch_path = weights / "pytorch_model.bin"
            if pytorch_path.is_file():
                try:
                    state = torch.load(
                        str(pytorch_path), map_location="cpu", weights_only=True
                    )
                except (OSError, RuntimeError, TypeError) as exc:
                    raise RuntimeError(
                        "Unable to load the ColBERT token projection from "
                        "pytorch_model.bin"
                    ) from exc
        if state is None:
            raise RuntimeError(
                "Multivector model weights do not contain a token projection"
            )
        for key in ("linear.weight", "colbert.linear.weight"):
            projection_weight = state.get(key)
            if projection_weight is not None:
                if (
                    not hasattr(projection_weight, "ndim")
                    or projection_weight.ndim != 2
                ):
                    raise RuntimeError("Multivector token projection must be a matrix")
                return projection_weight.detach().float()
        raise RuntimeError(
            "Multivector model weights do not contain a linear token projection"
        )

    def request(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """Execute the supported dense, sparse, or token representation request."""
        if path.endswith("/chat/completions"):
            return self._generate(payload)
        if path.endswith("/rerank") or path == "/rerank":
            query = payload.get("query")
            texts = payload.get("texts")
            if not isinstance(query, str) or not isinstance(texts, list) or not all(
                isinstance(item, str) for item in texts
            ):
                raise ValueError("In-process rerank input must contain query and texts")
            return self._rerank(query, texts)
        if path.endswith("/entities"):
            inputs = payload.get("input")
            labels = payload.get("labels")
            threshold = payload.get("threshold", 0.35)
            if not isinstance(inputs, list) or not all(
                isinstance(item, str) for item in inputs
            ):
                raise ValueError("Entity input must be a list of strings")
            if not isinstance(labels, list) or not all(
                isinstance(item, str) for item in labels
            ):
                raise ValueError("Entity labels must be a list of strings")
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
                raise ValueError("Entity threshold must be numeric")
            return self._entities(inputs, labels, float(threshold))
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

    def _entities(
        self, texts: list[str], labels: list[str], threshold: float
    ) -> dict[str, object]:
        """Run GLiNER span extraction and normalize its response shape."""
        if getattr(self._definition, "backend", "") != "gliner":
            raise RuntimeError("The configured local model is not a GLiNER model")
        rows: list[dict[str, object]] = []
        for index, text in enumerate(texts):
            raw_entities = self._model.predict_entities(
                text, labels, threshold=threshold
            )
            entities = [
                {
                    "text": str(entity["text"]),
                    "label": str(entity["label"]),
                    "start": int(entity["start"]),
                    "end": int(entity["end"]),
                    "score": float(entity["score"]),
                }
                for entity in raw_entities
            ]
            rows.append({"index": index, "entities": entities})
        return {"data": rows}

    def _generate(self, payload: dict[str, object]) -> dict[str, object]:
        """Generate one deterministic response for a local chat-completions request."""
        import torch

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Local generation input must contain messages")
        prompt = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                encoded = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except (TypeError, ValueError):
                encoded = self._tokenizer(prompt, return_tensors="pt")
        else:
            encoded = self._tokenizer(prompt, return_tensors="pt")
        if not isinstance(encoded, dict):
            encoded = {"input_ids": encoded}
        encoded = self._move_to_device(encoded)
        input_length = encoded["input_ids"].shape[-1]
        raw_max_tokens = payload.get("max_tokens", 256)
        if not isinstance(raw_max_tokens, int) or isinstance(raw_max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        max_new_tokens = raw_max_tokens
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        content = self._tokenizer.decode(
            generated[0][input_length:].detach().cpu(), skip_special_tokens=True
        )
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}]
        }

    def _rerank(self, query: str, texts: list[str]) -> dict[str, object]:
        """Score query-document pairs using a classifier or Qwen relevance prompt."""
        if str(self._definition.parameter("reranker_type", "")) == "generative":
            scores = [self._qwen_score(query, text) for text in texts]
        else:
            scores = self._classifier_scores(query, texts)
        return {
            "results": [
                {"index": index, "relevance_score": score}
                for index, score in enumerate(scores)
            ]
        }

    def _classifier_scores(self, query: str, texts: list[str]) -> list[float]:
        """Run a sequence-classification reranker over bounded query pairs."""
        import torch

        batch_size = int(cast(int, self._definition.parameter("batch_size", 2)))
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=int(
                    cast(int, self._definition.parameter("max_length", 512))
                ),
                return_tensors="pt",
            )
            encoded = self._move_to_device(encoded)
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            if len(logits.shape) != 2:
                raise RuntimeError("Local classifier returned invalid reranker logits")
            if logits.shape[1] == 1:
                values = logits[:, 0]
            else:
                values = torch.sigmoid(logits[:, -1] - logits[:, 0])
            scores.extend(float(value) for value in values.detach().cpu())
        return scores

    def _qwen_score(self, query: str, text: str) -> float:
        """Score one pair with Qwen's yes/no next-token relevance decision."""
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "Judge whether the Document meets the Query. "
                    "Output only yes or no."
                ),
            },
            {"role": "user", "content": f"<Query>: {query}\n<Document>: {text}"},
        ]
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            encoded = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        if isinstance(encoded, Mapping):
            encoded = dict(encoded)
            if "attention_mask" not in encoded:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        else:
            encoded = {
                "input_ids": encoded,
                "attention_mask": torch.ones_like(encoded),
            }
        encoded = self._move_to_device(encoded)
        with torch.inference_mode():
            logits = self._model(**encoded).logits[:, -1, :]
        yes_id = self._token_id("yes", "yes_token_id")
        no_id = self._token_id("no", "no_token_id")
        return float(torch.sigmoid(logits[:, yes_id] - logits[:, no_id])[0].cpu())

    def _token_id(self, token: str, parameter: str) -> int:
        """Resolve an optional configured token ID or the tokenizer's final token."""
        configured = self._definition.parameter(parameter)
        if isinstance(configured, int):
            return configured
        encoded = self._tokenizer(token, add_special_tokens=False)["input_ids"]
        if not encoded:
            raise RuntimeError(f"Tokenizer produced no token ID for {token!r}")
        return int(encoded[-1])

    def _move_to_device(self, encoded: Any) -> Any:
        """Move tensor fields while preserving the tokenizer's encoded shape."""
        if hasattr(encoded, "to"):
            return encoded.to(self._device)
        if isinstance(encoded, dict):
            return {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        raise RuntimeError("Tokenizer returned an unsupported encoded input")

    def _forward(self, texts: list[str]) -> tuple[Any, Any]:
        """Tokenize and run a bounded forward pass without retaining activations."""
        import torch

        tokenizer = self._tokenizer
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(cast(int, self._definition.parameter("max_length", 256))),
            return_tensors="pt",
        )
        encoded = self._move_to_device(encoded)
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
                {"index": i, "embedding": row.detach().cpu().tolist()}
                for i, row in enumerate(pooled)
            ]
        }

    def _tokens(self, texts: list[str]) -> dict[str, object]:
        """Return masked token embeddings for MaxSim retrieval."""
        import torch

        output, mask = self._forward(texts)
        values = output.last_hidden_state
        if self._token_projection is not None:
            projection = self._token_projection.to(values.device, dtype=values.dtype)
            values = torch.matmul(values, projection.transpose(0, 1))
        if bool(self._definition.parameter("normalize", True)):
            values = torch.nn.functional.normalize(values, p=2, dim=-1)
        offsets: list[list[tuple[int, int]]] = []
        try:
            offset_values = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=int(
                    cast(int, self._definition.parameter("max_length", 256))
                ),
                return_offsets_mapping=True,
                return_tensors="pt",
            )["offset_mapping"]
        except (TypeError, ValueError, KeyError):
            offset_values = None
        for index in range(len(texts)):
            if offset_values is None:
                offsets.append([])
            else:
                offsets.append(
                    [
                        (int(pair[0]), int(pair[1]))
                        for pair in offset_values[index]
                    ]
                )
        data: list[dict[str, object]] = []
        for index in range(len(texts)):
            row_offsets = offsets[index] or [(1, 1)] * len(values[index])
            selected_values = [
                row.detach().cpu().tolist()
                for row, pair, keep in zip(
                    values[index], row_offsets, mask[index], strict=True
                )
                if bool(keep) and (not offsets[index] or pair[1] > pair[0])
            ]
            selected_offsets = [
                pair
                for pair, keep in zip(row_offsets, mask[index], strict=True)
                if bool(keep) and (not offsets[index] or pair[1] > pair[0])
            ]
            data.append(
                {
                    "index": index,
                    "token_embeddings": selected_values,
                    "token_offsets": selected_offsets if offsets[index] else [],
                }
            )
        return {"data": data}

    def _sparse(self, texts: list[str]) -> dict[str, object]:
        """Return learned sparse weights with a safe token-weight fallback."""
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
                        "indices": selected.indices[nonzero].detach().cpu().tolist(),
                        "values": selected.values[nonzero].detach().cpu().tolist(),
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
