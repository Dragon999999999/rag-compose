"""Application facade for CLI and future clients; execution belongs to orchestration."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from llm_grounding.domain.documents import DocumentSource
from llm_grounding.domain.runs import RagRequest, RagRun
from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.infrastructure.llm.protocol import StructuredModelClient
from llm_grounding.orchestration.registry import ComponentRegistry
from llm_grounding.orchestration.runner import ArchitectureRunner


def validate_pdf_path(path: Path) -> Path:
    """Resolve an existing PDF path; parsing later validates the actual content.

    Raises:
        FileNotFoundError: If the path is missing or is not a file.
        ValueError: If the extension does not identify a PDF.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {path}")
    return path


class RagService:
    """Expose configured architecture invocation to application clients."""

    def __init__(
        self,
        config: ArchitectureConfig,
        model_client: StructuredModelClient | None = None,
        registry: ComponentRegistry | None = None,
    ) -> None:
        """Resolve configuration and optionally inject an offline model client."""
        self._config = config
        components = registry if registry is not None else ComponentRegistry()
        self._runner = ArchitectureRunner(components.build(config, model_client))

    async def run(self, request: RagRequest) -> RagRun:
        """Invoke orchestration with effective configuration and software version."""
        return await self._runner.run(
            request,
            {
                "architecture_config": self._config.model_dump(mode="json"),
                "package_version": version("llm-grounding"),
            },
        )

    async def process_pdf(self, path: Path) -> RagRun:
        """Validate local input and execute the default extraction task.

        Document IDs are deterministic hashes of the resolved source URI. The
        parser separately records the content digest to identify input changes.
        Invalid application input fails before a run starts; component failures
        expose their partial run through RunFailedError.
        """
        path = validate_pdf_path(path)
        document_id = sha256(path.as_uri().encode("utf-8")).hexdigest()
        return await self.run(RagRequest(DocumentSource(document_id, path)))
