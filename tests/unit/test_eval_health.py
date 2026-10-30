"""Contract tests for the Target Protocol v1 health endpoint."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from llm_grounding.api.eval import create_eval_router
from llm_grounding.application.eval_service import EvalService


class FakeSession:
    """Minimal asynchronous PostgreSQL session substitute."""

    def __init__(self, error: Exception | None = None, delay: float = 0.0) -> None:
        """Configure the result of the SQL probe."""
        self._error = error
        self._delay = delay

    async def __aenter__(self) -> "FakeSession":
        """Enter the fake database session."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Leave the fake database session."""

    async def execute(self, _statement: object) -> None:
        """Execute the health probe or reproduce its configured failure."""
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error


class FakeSessionFactory:
    """Create independent fake sessions for each health probe."""

    def __init__(self, **kwargs: object) -> None:
        """Store session behavior for later context-manager creation."""
        self._kwargs = kwargs

    def __call__(self) -> FakeSession:
        """Return a new fake session."""
        return FakeSession(**self._kwargs)


class FakeVectors:
    """Minimal vector-backend health boundary."""

    def __init__(self, error: Exception | None = None) -> None:
        """Configure an optional backend failure."""
        self._error = error

    async def health_check(self) -> None:
        """Complete or fail the vector service probe."""
        if self._error is not None:
            raise self._error


def _service(
    *,
    repository: FakeSessionFactory,
    vectors: object | None = None,
    timeout: float = 0.05,
) -> EvalService:
    """Build an evaluation service around lightweight dependency doubles."""
    application = SimpleNamespace(
        repository=SimpleNamespace(session_factory=repository),
        _vectors=vectors,
        runner=object(),
    )
    return EvalService(application, health_timeout_seconds=timeout)  # type: ignore[arg-type]


async def _health(service: EvalService) -> httpx.Response:
    """Call the health route through an in-process ASGI client."""
    app = FastAPI()
    app.include_router(create_eval_router(service))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get("/eval/v1/health")


async def test_health_reports_healthy_dependencies() -> None:
    """Healthy PostgreSQL, vectors, and runner return HTTP 200."""
    response = await _health(
        _service(
            repository=FakeSessionFactory(),
            vectors=FakeVectors(),
        )
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"status", "target", "details"}
    assert payload["status"] == "READY"
    assert payload["target"] == {
        "name": "llm-grounding",
        "version": "1.0.0",
        "implementation": "llm-grounding",
    }
    assert all(
        dependency["status"] == "healthy"
        for dependency in payload["details"]["dependencies"].values()
    )


async def test_health_reports_degraded_when_vectors_are_not_configured() -> None:
    """The API remains responsive but reports missing vector configuration."""
    response = await _health(_service(repository=FakeSessionFactory()))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "DEGRADED"
    assert payload["details"]["dependencies"]["vector_db"] == {
        "status": "degraded",
        "error": "vector backend is not configured",
    }


async def test_health_returns_service_unavailable_for_database_failure() -> None:
    """An unavailable essential database makes the health endpoint unhealthy."""
    response = await _health(
        _service(
            repository=FakeSessionFactory(error=ConnectionError("database down")),
            vectors=FakeVectors(),
        )
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "NOT_READY"
    assert payload["details"]["dependencies"]["postgresql"]["status"] == "unhealthy"
    assert "ConnectionError" in payload["details"]["dependencies"]["postgresql"][
        "error"
    ]


@pytest.mark.asyncio
async def test_health_times_out_a_stalled_dependency() -> None:
    """A stalled probe is bounded and produces an unhealthy 503 response."""
    response = await _health(
        _service(
            repository=FakeSessionFactory(delay=1.0),
            vectors=FakeVectors(),
            timeout=0.01,
        )
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "NOT_READY"
    assert payload["details"]["dependencies"]["postgresql"]["status"] == "unhealthy"
    assert "timed out" in payload["details"]["dependencies"]["postgresql"]["error"]
