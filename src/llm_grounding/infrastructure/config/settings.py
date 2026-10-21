"""Parse environment-backed infrastructure settings at one application boundary.

The application supplies these validated values to infrastructure adapters. This
keeps Docker-development defaults out of domain code and makes host deployments
overrideable without changing composition logic.
"""

import os
from dataclasses import dataclass


def _value(name: str, default: str = "") -> str:
    """Read and trim one environment setting without logging its value."""
    return os.environ.get(name, default).strip()


def _boolean(name: str, default: bool) -> bool:
    """Parse a conventional environment boolean or raise a configuration error."""
    value = _value(name, str(default)).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class ApiSettings:
    """Host and port settings for the browser-facing API server."""

    host: str
    port: int

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        """Read and validate the API bind address from the environment."""
        raw_port = _value("API_PORT", "8002")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError(
                "API_PORT must be an integer between 1 and 65535"
            ) from exc
        if not 1 <= port <= 65535:
            raise ValueError("API_PORT must be an integer between 1 and 65535")
        return cls(host=_value("API_HOST", "127.0.0.1"), port=port)


@dataclass(frozen=True)
class MinioSettings:
    """Connection and bucket settings for an S3-compatible artifact store."""

    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool

    @classmethod
    def from_environment(cls) -> "MinioSettings":
        """Read required MinIO settings from the environment.

        Raises:
            RuntimeError: If a required connection or bucket value is missing.
        """
        settings = cls(
            endpoint=_value("MINIO_ENDPOINT"),
            access_key=_value("MINIO_ACCESS_KEY"),
            secret_key=_value("MINIO_SECRET_KEY"),
            bucket=_value("MINIO_BUCKET", "llm-grounding-documents"),
            secure=_boolean("MINIO_SECURE", False),
        )
        if not all((settings.endpoint, settings.access_key, settings.secret_key)):
            raise RuntimeError(
                "Set MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY "
                "to use persisted Basic RAG artifacts."
            )
        return settings


@dataclass(frozen=True)
class QdrantSettings:
    """Connection and versioned collection settings for the derived vector index."""

    url: str
    api_key: str | None
    collection: str

    @classmethod
    def from_environment(cls) -> "QdrantSettings":
        """Read Qdrant settings, accepting a missing API key for local development."""
        url = _value("QDRANT_URL")
        if not url:
            raise RuntimeError("Set QDRANT_URL to use persisted Basic RAG retrieval.")
        api_key = _value("QDRANT_API_KEY") or None
        collection = _value("QDRANT_COLLECTION", "llm_grounding_v1")
        return cls(url=url, api_key=api_key, collection=collection)
