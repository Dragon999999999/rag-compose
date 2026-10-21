"""Read safe YAML into typed composition settings without constructing components.

Pipeline loading also validates closed component identifiers and model-catalog
references before a configuration can reach persistence or a provider adapter.
"""

from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import ValidationError

from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
    ModelCatalog,
    PipelineConfig,
)
from llm_grounding.ingestion.models import IngestionConfig


class ConfigurationError(ValueError):
    """Report malformed settings without echoing potentially sensitive input values."""


class _PipelineValidator(Protocol):
    """Structural type for the closed component registry boundary."""

    def validate_pipeline(self, config: PipelineConfig, catalog: ModelCatalog) -> None:
        """Validate pipeline component and model selections."""


def load_architecture_config(path: Path | None = None) -> ArchitectureConfig:
    """Read explicit YAML or the packaged default, validating its shape.

    The checked-in YAML is bundled in wheels; editable checkouts read that same
    file. Component identifiers are validated separately by the registry.

    Raises:
        ConfigurationError: If YAML cannot be read or fails typed validation.
    """
    try:
        if path is not None:
            content = path.read_text(encoding="utf-8")
        else:
            resource = files("llm_grounding.infrastructure.config") / "baseline_v0.yaml"
            if resource.is_file():
                content = resource.read_text(encoding="utf-8")
            else:
                checkout = Path(__file__).resolve().parents[4]
                content = (
                    checkout / "configs/architectures/baseline_v0.yaml"
                ).read_text(encoding="utf-8")
        return ArchitectureConfig.model_validate(yaml.safe_load(content))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "Unable to read architecture YAML; check its path and syntax."
        ) from exc
    except ValidationError as exc:
        raise ConfigurationError(_validation_locations("architecture", exc)) from exc


def load_ingestion_config(path: Path | None = None) -> IngestionConfig:
    """Read explicit or packaged ingestion settings with the same safe YAML boundary."""
    try:
        if path is not None:
            content = path.read_text(encoding="utf-8")
        else:
            resource = (
                files("llm_grounding.infrastructure.config") / "ingestion_default.yaml"
            )
            content = (
                resource.read_text(encoding="utf-8")
                if resource.is_file()
                else (
                    Path(__file__).resolve().parents[4]
                    / "configs/ingestion/default.yaml"
                ).read_text(encoding="utf-8")
            )
        return IngestionConfig.model_validate(yaml.safe_load(content))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "Unable to read ingestion YAML; check its path and syntax."
        ) from exc
    except ValidationError as exc:
        raise ConfigurationError(_validation_locations("ingestion", exc)) from exc


def load_model_catalog(path: Path | None = None) -> ModelCatalog:
    """Read the shared credential-free model catalog from YAML."""
    try:
        return ModelCatalog.model_validate(_yaml_content(path, "models.yaml"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "Unable to read model catalog YAML; check its path and syntax."
        ) from exc
    except ValidationError as exc:
        raise ConfigurationError(_validation_locations("model catalog", exc)) from exc


def load_pipeline_config(
    path: Path | None = None,
    *,
    catalog: ModelCatalog | None = None,
    registry: _PipelineValidator | None = None,
) -> PipelineConfig:
    """Read, type-check, and closed-registry validate one pipeline YAML file.

    Args:
        path: YAML file containing one complete pipeline definition. When omitted,
            the checked-in or packaged ``basic_rag_v1`` pipeline is used.
        catalog: Shared model catalog; the packaged catalog is used by default.
        registry: Optional component registry, primarily useful for tests.

    Raises:
        ConfigurationError: If YAML, schema, component, or model validation fails.
    """
    try:
        path = path or _default_pipeline_path()
        config = PipelineConfig.model_validate(_yaml_content(path, ""))
        model_catalog = catalog or load_model_catalog()
        if registry is None:
            from llm_grounding.orchestration.registry import ComponentRegistry

            registry = ComponentRegistry()
        registry.validate_pipeline(config, model_catalog)
        return config
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "Unable to read pipeline YAML; check its path and syntax."
        ) from exc
    except ValidationError as exc:
        raise ConfigurationError(_validation_locations("pipeline", exc)) from exc
    except ValueError as exc:
        raise ConfigurationError(
            "Invalid pipeline component or model reference"
        ) from exc


def discover_pipeline_paths(directory: Path | None = None) -> tuple[Path, ...]:
    """Return deterministic YAML pipeline files from the configured directory."""
    root = directory or _checkout_root() / "configs/pipelines"
    if directory is None and not root.is_dir():
        packaged = files("llm_grounding.infrastructure.config") / "pipelines"
        if packaged.is_dir():
            root = Path(str(packaged))
    if not root.is_dir():
        return ()
    return tuple(sorted((*root.glob("*.yaml"), *root.glob("*.yml"))))


def pipeline_source_fingerprint(path: Path) -> str:
    """Hash the exact UTF-8 source bytes used for import change detection."""
    return sha256(path.read_bytes()).hexdigest()


def load_pipeline_configs(
    directory: Path | None = None,
    *,
    catalog: ModelCatalog | None = None,
) -> tuple[tuple[Path, PipelineConfig], ...]:
    """Load all discovered pipeline files in deterministic path order."""
    return tuple(
        (path, load_pipeline_config(path, catalog=catalog))
        for path in discover_pipeline_paths(directory)
    )


def _yaml_content(path: Path | None, packaged_name: str) -> object:
    """Load YAML from an explicit path or a packaged/check-out fallback."""
    if path is not None:
        content = path.read_text(encoding="utf-8")
    elif packaged_name:
        resource = files("llm_grounding.infrastructure.config") / packaged_name
        content = (
            resource.read_text(encoding="utf-8")
            if resource.is_file()
            else (_checkout_root() / "configs/models.yaml").read_text(encoding="utf-8")
        )
    else:
        raise ValueError("An explicit pipeline path is required")
    value = yaml.safe_load(content)
    if value is None:
        raise ValueError("Configuration YAML must contain a mapping")
    return value


def _checkout_root() -> Path:
    """Return the repository root when running from an editable checkout."""
    return Path(__file__).resolve().parents[4]


def _default_pipeline_path() -> Path:
    """Return the packaged or checkout default pipeline YAML path."""
    paths = discover_pipeline_paths()
    for path in paths:
        if path.stem == "basic_rag_v1":
            return path
    raise ConfigurationError("No default basic_rag_v1 pipeline configuration exists")


def _validation_locations(kind: str, error: ValidationError) -> str:
    """Format only schema locations, never invalid input values."""
    locations = ", ".join(
        ".".join(str(part) for part in issue["loc"]) or "root"
        for issue in error.errors(include_input=False)
    )
    return f"Invalid {kind} configuration at: {locations}"
