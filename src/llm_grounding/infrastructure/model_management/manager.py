"""Lease-based orchestration of model storage and inference runtimes."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.model_management.models import (
    LocalInferenceClient,
    LocalModelHandle,
    ModelEndpoint,
    ModelStatus,
)
from llm_grounding.infrastructure.model_management.registry import ModelRegistry
from llm_grounding.infrastructure.model_management.runtime import (
    DockerModelRuntime,
    InProcessModelRuntime,
)
from llm_grounding.infrastructure.model_management.storage import ModelStorageManager


class LocalModelManager:
    """Resolve, prepare, start, and lease local models on demand."""

    def __init__(
        self,
        registry: ModelRegistry,
        storage: ModelStorageManager | None = None,
        runtime: DockerModelRuntime | None = None,
    ) -> None:
        """Compose registry, persistent storage, and runtime services explicitly."""
        self.registry = registry
        self.storage = storage or ModelStorageManager()
        self.runtime = runtime or DockerModelRuntime()
        self._in_process_runtime = InProcessModelRuntime()
        self._locks: dict[str, asyncio.Lock] = {}
        self._leases: dict[str, int] = {}
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._idle_tasks: dict[str, asyncio.Task[None]] = {}

    @asynccontextmanager
    async def acquire(self, model_id: str) -> AsyncIterator[LocalModelHandle]:
        """Acquire a local model lease and release it after inference completes."""
        definition = self.registry.resolve(model_id)
        if definition.location != "local":
            raise ValueError(
                f"Model {model_id} is remote; use its existing provider adapter"
            )
        lock = self._locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            idle_task = self._idle_tasks.pop(model_id, None)
            if idle_task is not None:
                idle_task.cancel()
            endpoint = self._endpoints.get(model_id)
            if endpoint is None:
                weights = await self.storage.ensure_available(definition)
                endpoint = await self._runtime_for(definition).start_and_wait(
                    definition, weights
                )
                self._endpoints[model_id] = endpoint
            self._leases[model_id] = self._leases.get(model_id, 0) + 1
        runtime = self._runtime_for(definition)
        client_for = getattr(runtime, "client_for", None)
        client = (
            client_for(definition)
            if callable(client_for)
            else LocalInferenceClient(endpoint.base_url)
        )
        handle = LocalModelHandle(definition, endpoint, client)
        try:
            yield handle
        finally:
            await self._release(definition)

    async def _release(self, definition: ModelDefinition) -> None:
        """Release one lease and schedule idle shutdown when no callers remain."""
        model_id = definition.id
        self._leases[model_id] = max(0, self._leases.get(model_id, 1) - 1)
        if self._leases[model_id] != 0 or definition.runtime is None:
            return
        timeout = definition.runtime.idle_timeout_seconds
        runtime = self._runtime_for(definition)
        if timeout == 0:
            await runtime.stop(definition)
            return
        self._idle_tasks[model_id] = asyncio.create_task(
            self._stop_when_idle(definition, timeout)
        )

    async def _stop_when_idle(
        self, definition: ModelDefinition, timeout: float
    ) -> None:
        """Stop a model after its configured idle period unless it is reacquired."""
        try:
            await asyncio.sleep(timeout)
            if self._leases.get(definition.id, 0) == 0:
                await self._runtime_for(definition).stop(definition)
                self._endpoints.pop(definition.id, None)
        except asyncio.CancelledError:
            return
        finally:
            self._idle_tasks.pop(definition.id, None)

    async def status(self, model_id: str) -> ModelStatus:
        """Return weights, runtime, and active-lease state without starting a model."""
        definition = self.registry.resolve(model_id)
        weights_available = self.storage.is_available(definition)
        running, endpoint, detail = await self._runtime_for(definition).status(
            definition
        )
        return ModelStatus(
            model_id=model_id,
            location=definition.location,
            weights_available=weights_available,
            runtime_type=definition.runtime.type if definition.runtime else None,
            running=running,
            active_leases=self._leases.get(model_id, 0),
            endpoint=endpoint,
            detail=detail,
        )

    async def stop(self, model_id: str) -> None:
        """Stop an idle model without deleting its persistent weights."""
        definition = self.registry.resolve(model_id)
        if self._leases.get(model_id, 0):
            raise RuntimeError(f"Model {model_id} has active leases")
        await self._runtime_for(definition).stop(definition)
        self._endpoints.pop(model_id, None)

    async def restart(self, model_id: str) -> None:
        """Restart a model container while retaining its weights."""
        definition = self.registry.resolve(model_id)
        await self._runtime_for(definition).restart(definition)

    async def remove_runtime(self, model_id: str) -> None:
        """Remove a runtime container without removing model weights."""
        definition = self.registry.resolve(model_id)
        if self._leases.get(model_id, 0):
            raise RuntimeError(f"Model {model_id} has active leases")
        await self._runtime_for(definition).remove(definition)
        self._endpoints.pop(model_id, None)

    async def close(self) -> None:
        """Cancel pending idle-shutdown tasks during application shutdown."""
        tasks = tuple(self._idle_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._idle_tasks.clear()
        for model_id, leases in tuple(self._leases.items()):
            if leases == 0:
                await self._runtime_for(self.registry.resolve(model_id)).stop(
                    self.registry.resolve(model_id)
                )
        self._endpoints.clear()

    def _runtime_for(self, definition: ModelDefinition) -> Any:
        """Select Docker or in-process lifecycle without changing manager callers."""
        if definition.runtime is not None and definition.runtime.type == "in_process":
            if isinstance(self.runtime, DockerModelRuntime):
                return self._in_process_runtime
        return self.runtime
