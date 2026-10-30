"""Development artifact storage rooted at an explicit local directory."""

import asyncio
import shutil
from pathlib import Path


class LocalArtifactStore:
    """Copy PDFs atomically enough for local development; future S3 adapters share this contract."""

    def __init__(self, root: Path) -> None:
        """Configure the directory that owns stored PDF artifacts."""
        self._root = root.resolve()

    def _path(self, key: str) -> Path:
        """Reject traversal and map one slash-separated artifact key under the root."""
        candidate = (self._root / key).resolve()
        if self._root not in candidate.parents and candidate != self._root:
            raise ValueError("Artifact key escapes the configured storage root")
        return candidate

    async def put_pdf(self, key: str, source: Path) -> None:
        """Copy a PDF once; a matching content-addressed key is safely reusable."""
        target = self._path(key)
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source, target)

    async def path_for(self, key: str) -> Path:
        """Return the local artifact path only when it exists as a regular file."""
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(f"Stored artifact not found: {key}")
        return path
