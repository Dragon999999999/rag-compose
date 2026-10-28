"""Verify Docker image management in the local model runtime."""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from docker.errors import ImageNotFound

from llm_grounding.infrastructure.config.loader import load_model_registry
from llm_grounding.infrastructure.model_management.runtime import DockerModelRuntime


class _FakeImages:
    """Provide the Docker image lookup and pull operations used by the runtime."""

    def __init__(
        self, *, available: bool = False, pull_error: Exception | None = None
    ) -> None:
        self.available = available
        self.pull_error = pull_error
        self.get_calls = 0
        self.pull_calls = 0

    def get(self, _image: str) -> object:
        """Return a local image or the SDK's missing-image exception."""
        self.get_calls += 1
        if not self.available:
            raise ImageNotFound("image is not available locally")
        return object()

    def pull(self, _image: str) -> object:
        """Record a pull and make the image available when it succeeds."""
        self.pull_calls += 1
        time.sleep(0.02)
        if self.pull_error is not None:
            raise self.pull_error
        self.available = True
        return object()


class _FakeContainer:
    """Represent the subset of a Docker container used by startup."""

    status = "created"
    labels: dict[str, str]
    attrs: dict[str, Any]

    def __init__(self, labels: dict[str, str]) -> None:
        self.labels = labels
        self.attrs = {"Config": {"Labels": labels}}

    def start(self) -> None:
        """Mark the fake container as running."""
        self.status = "running"

    def reload(self) -> None:
        """Match the Docker SDK reload operation without external I/O."""

    def logs(self, tail: int = 100) -> bytes:
        """Return representative model-server startup output."""
        del tail
        return b"model server startup diagnostics"


class _FakeContainers:
    """Track named container lookup and creation calls."""

    def __init__(self) -> None:
        self.container: _FakeContainer | None = None
        self.create_calls = 0

    def get(self, _name: str) -> _FakeContainer:
        """Return the created container or report that it does not exist."""
        if self.container is None:
            raise KeyError("container not found")
        return self.container

    def create(self, **kwargs: object) -> _FakeContainer:
        """Create one fake container from the supplied Docker labels."""
        self.create_calls += 1
        labels = kwargs["labels"]
        assert isinstance(labels, dict)
        self.container = _FakeContainer(labels)
        return self.container


class _FakeDockerClient:
    """Combine fake image and container managers like a Docker client."""

    def __init__(self, images: _FakeImages) -> None:
        self.images = images
        self.containers = _FakeContainers()


def _docker_definition():
    """Return a registered Docker model definition with a deterministic port."""
    definition = load_model_registry().resolve("embedding_001")
    runtime = definition.runtime
    assert runtime is not None
    return definition.model_copy(
        update={"runtime": runtime.model_copy(update={"host_port": 18000})}
    )


@pytest.mark.asyncio
async def test_existing_docker_image_is_not_pulled() -> None:
    """Container creation reuses an image that is already present locally."""
    images = _FakeImages(available=True)
    client = _FakeDockerClient(images)
    runtime = DockerModelRuntime(client=client)

    await runtime.ensure_started(_docker_definition(), Path("/tmp/model"))

    assert images.get_calls == 1
    assert images.pull_calls == 0
    assert client.containers.create_calls == 1


@pytest.mark.asyncio
async def test_missing_docker_image_is_pulled_before_container_creation() -> None:
    """A missing image is downloaded before the runtime creates its container."""
    images = _FakeImages()
    client = _FakeDockerClient(images)
    runtime = DockerModelRuntime(client=client)

    await runtime.ensure_started(_docker_definition(), Path("/tmp/model"))

    assert images.pull_calls == 1
    assert client.containers.create_calls == 1


@pytest.mark.asyncio
async def test_docker_image_pull_failure_has_actionable_error() -> None:
    """Registry authentication failures identify the image and failure category."""
    images = _FakeImages(pull_error=RuntimeError("unauthorized: access denied"))
    runtime = DockerModelRuntime(client=_FakeDockerClient(images))
    definition = _docker_definition()

    with pytest.raises(RuntimeError, match="authentication or access") as error:
        await runtime.ensure_started(definition, Path("/tmp/model"))

    assert definition.runtime is not None
    assert definition.runtime.image in str(error.value)


@pytest.mark.asyncio
async def test_concurrent_starts_pull_one_shared_image_once() -> None:
    """Concurrent model starts serialize image management for the same image."""
    images = _FakeImages()
    client = _FakeDockerClient(images)
    runtime = DockerModelRuntime(client=client)
    definition = _docker_definition()

    await asyncio.gather(
        runtime.ensure_started(definition, Path("/tmp/model")),
        runtime.ensure_started(definition, Path("/tmp/model")),
    )

    assert images.pull_calls == 1
    assert client.containers.create_calls == 1


@pytest.mark.asyncio
async def test_readiness_timeout_includes_container_startup_logs() -> None:
    """A crashed or unready server exposes bounded container diagnostics."""
    images = _FakeImages(available=True)
    client = _FakeDockerClient(images)
    definition = _docker_definition()
    assert definition.runtime is not None
    definition = definition.model_copy(
        update={
            "runtime": definition.runtime.model_copy(
                update={"readiness_timeout_seconds": 0.01}
            )
        }
    )

    async def never_ready(_url: str) -> bool:
        """Keep the readiness probe unsuccessful for the timeout test."""
        return False

    runtime = DockerModelRuntime(client=client, probe=never_ready)
    with pytest.raises(TimeoutError, match="model server startup diagnostics"):
        await runtime.start_and_wait(definition, Path("/tmp/model"))
