"""Docker-backed local inference runtime lifecycle."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.model_management.models import ModelEndpoint

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
