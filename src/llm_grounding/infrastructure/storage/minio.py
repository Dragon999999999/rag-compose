"""MinIO implementation of the artifact-store boundary.

Original PDFs remain canonical binary artifacts in S3-compatible storage. The
adapter materializes a temporary local copy only because the existing PDF parser
accepts paths; callers never receive MinIO SDK objects.
"""

import asyncio
import os
import tempfile
from pathlib import Path, PurePosixPath

from minio import Minio
from minio.error import S3Error

from llm_grounding.infrastructure.config.settings import MinioSettings


class MinioArtifactStore:
    """Store content-addressed PDFs in MinIO and materialize parser inputs on demand."""

    def __init__(self, settings: MinioSettings) -> None:
        """Configure a synchronous MinIO client for thread-offloaded operations."""
        self._bucket = settings.bucket
        self._client = Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
        )

    @staticmethod
    def _validate_key(key: str) -> str:
        """Reject blank, absolute, and traversal-prone object keys."""
        path = PurePosixPath(key)
        if not key or path.is_absolute() or ".." in path.parts:
            raise ValueError("Artifact key must be a relative object key")
        return key

    def _ensure_bucket(self) -> None:
        """Create the configured bucket once when its credentials permit it."""
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    async def put_pdf(self, key: str, source: Path) -> None:
        """Upload a PDF once under a deterministic content-addressed key.

        Raises:
            RuntimeError: If MinIO cannot be reached or the bucket cannot be used.
        """
        object_name = self._validate_key(key)

        def upload() -> None:
            try:
                self._ensure_bucket()
                try:
                    self._client.stat_object(self._bucket, object_name)
                except S3Error as exc:
                    if exc.code not in {"NoSuchKey", "NoSuchObject"}:
                        raise
                    self._client.fput_object(
                        self._bucket,
                        object_name,
                        str(source),
                        content_type="application/pdf",
                    )
            except (OSError, S3Error) as exc:
                raise RuntimeError(
                    "MinIO could not store the document artifact"
                ) from exc

        await asyncio.to_thread(upload)

    async def path_for(self, key: str) -> Path:
        """Download one object to a temporary PDF path for the path-based parser.

        The returned file is intentionally ephemeral; MinIO remains the canonical
        source and re-ingestion can recreate the local parser input at any time.
        """
        object_name = self._validate_key(key)

        def download() -> Path:
            suffix = Path(object_name).suffix or ".pdf"
            descriptor, target = tempfile.mkstemp(
                prefix="llm-grounding-", suffix=suffix
            )
            os.close(descriptor)
            Path(target).unlink(missing_ok=True)
            try:
                self._ensure_bucket()
                self._client.fget_object(self._bucket, object_name, target)
                return Path(target)
            except (OSError, S3Error) as exc:
                Path(target).unlink(missing_ok=True)
                raise FileNotFoundError(
                    f"MinIO artifact not available: {object_name}"
                ) from exc

        return await asyncio.to_thread(download)
