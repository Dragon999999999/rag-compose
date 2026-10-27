"""Persistent, concurrent-safe storage for local model weights."""

import asyncio
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import cast

from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.config.settings import ModelStorageSettings


class ModelStorageError(RuntimeError):
    """Report a model source, integrity, or storage failure."""


class ModelStorageManager:
    """Download local model weights once into a persistent application directory."""

    def __init__(self, root: Path | None = None) -> None:
        """Use ``MODEL_STORAGE_ROOT`` or a user-cache default when omitted."""
        configured = ModelStorageSettings.from_environment().root
        self.root = (root or Path(configured)).expanduser()
        self._locks: dict[str, asyncio.Lock] = {}

    def path_for(self, definition: ModelDefinition) -> Path:
        """Return the validated persistent path for one local model."""
        if definition.storage is None:
            raise ModelStorageError(
                f"Model {definition.id} has no storage configuration"
            )
        candidate = (self.root / definition.storage.relative_path).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ModelStorageError(
                "Model storage relative_path escapes the storage root"
            ) from exc
        return candidate

    def is_available(self, definition: ModelDefinition) -> bool:
        """Return whether a complete local installation is currently available."""
        if definition.location != "local":
            return False
        if (
            definition.source is not None
            and definition.source.type == "local_directory"
        ):
            return self._directory_has_files(
                Path(cast(str, definition.source.path)).expanduser()
            )
        target = self.path_for(definition)
        return (target / ".model_complete").is_file() and self._directory_has_files(
            target
        )

    async def ensure_available(self, definition: ModelDefinition) -> Path:
        """Return local weights, downloading atomically when necessary."""
        if definition.location != "local":
            raise ModelStorageError(
                f"Model {definition.id} is remote and has no local weights"
            )
        lock = self._locks.setdefault(definition.id, asyncio.Lock())
        async with lock:
            return self._ensure_sync(definition)

    def _ensure_sync(self, definition: ModelDefinition) -> Path:
        """Perform one serialized installation without blocking the event loop."""
        source = definition.source
        if source is None:
            raise ModelStorageError(f"Model {definition.id} has no local source")
        if source.type == "local_directory":
            path = Path(cast(str, source.path)).expanduser().resolve()
            if not self._directory_has_files(path):
                raise ModelStorageError(
                    f"Local model directory is missing or empty: {path}"
                )
            return path

        target = self.path_for(definition)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.with_name(f".{target.name}.lock")
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock_file:
            self._lock_file(lock_file)
            try:
                if self.is_available(definition):
                    return target
                if target.exists():
                    shutil.rmtree(target)
                staging = target.with_name(f".{target.name}.partial-{os.getpid()}")
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                if source.type == "huggingface":
                    self._download_huggingface(definition, staging)
                else:
                    self._download_url(source.download_url, source.sha256, staging)
                if not self._directory_has_files(staging):
                    raise ModelStorageError(
                        f"Model source produced no files: {definition.id}"
                    )
                (staging / ".model_complete").write_text("complete\n", encoding="utf-8")
                staging.replace(target)
                return target
            finally:
                self._unlock_file(lock_file)

    @staticmethod
    def _directory_has_files(path: Path) -> bool:
        """Return true for a directory containing at least one non-marker file."""
        return path.is_dir() and any(
            item.name != ".model_complete" for item in path.iterdir()
        )

    @staticmethod
    def _lock_file(file: object) -> None:
        """Acquire a process-level advisory lock on Unix hosts."""
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]

    @staticmethod
    def _unlock_file(file: object) -> None:
        """Release a process-level advisory lock on Unix hosts."""
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]

    @staticmethod
    def _download_huggingface(definition: ModelDefinition, target: Path) -> None:
        """Download a repository snapshot using the optional Hugging Face client."""
        source = definition.source
        assert source is not None
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelStorageError(
                "Install the local-model dependencies to download Hugging Face models"
            ) from exc
        try:
            snapshot_download(
                repo_id=cast(str, source.repository),
                revision=source.revision,
                local_dir=str(target),
            )
        except Exception as exc:
            raise ModelStorageError(
                f"Unable to download Hugging Face model {definition.id}"
            ) from exc

    @staticmethod
    def _download_url(url: object, expected_sha256: str | None, target: Path) -> None:
        """Download one archive or weights file and verify its SHA-256 digest."""
        if url is None:
            raise ModelStorageError("URL model source is missing download_url")
        destination = target / "model.bin"
        temporary = target / ".download.partial"
        try:
            urllib.request.urlretrieve(str(url), temporary)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if expected_sha256 is None or digest.lower() != expected_sha256.lower():
                raise ModelStorageError(
                    "Downloaded model failed SHA-256 integrity verification"
                )
            temporary.replace(destination)
        except ModelStorageError:
            raise
        except Exception as exc:
            raise ModelStorageError(
                "Unable to download model weights from URL"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
