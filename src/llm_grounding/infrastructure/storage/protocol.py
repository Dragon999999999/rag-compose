"""Store original PDFs and artifacts outside relational metadata tables."""

from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    """Persist immutable artifact bytes by caller-chosen content-addressed keys."""

    async def put_pdf(self, key: str, source: Path) -> None:
        """Copy a validated PDF to the artifact key without exposing storage details."""
        ...

    async def path_for(self, key: str) -> Path:
        """Return a readable local path or raise FileNotFoundError for a missing artifact."""
        ...
