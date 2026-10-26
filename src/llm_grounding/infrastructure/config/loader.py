"""Read safe YAML into typed composition settings without constructing components.

Pipeline loading also validates closed component identifiers and model-catalog
references before a configuration can reach persistence or a provider adapter.
"""

from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Protocol, cast

import yaml
from pydantic import ValidationError

from llm_grounding.infrastructure.config.declarative import (
    DeclarativePipelineConfig,
)
from llm_grounding.infrastructure.config.models import (
    ModelCatalog,
    PipelineConfig,
)
from llm_grounding.infrastructure.config.registry import ConfigurationRegistry
from llm_grounding.ingestion.models import IngestionConfig


class ConfigurationError(ValueError):
    """Report malformed settings without echoing potentially sensitive input values."""


class _PipelineValidator(Protocol):
    """Structural type for the closed component registry boundary."""

    def validate_pipeline(self, config: PipelineConfig, catalog: ModelCatalog) -> None:
        """Validate pipeline component and model selections."""


def load_ingestion_config(path: Path | None = None) -> IngestionConfig:
    """Read explicit or canonical ingestion settings with the safe YAML boundary.

    The ingestion pipeline is resolved into the execution model used by the
    ingestion service.
    """
    try:
        ingestion_path = path or _default_ingestion_pipeline_path()
        pipeline = load_pipeline_config(ingestion_path)
        if pipeline.kind != "ingestion":
            raise ConfigurationError(
                "The ingestion configuration must select an ingestion pipeline"
            )
        return pipeline.resolve_ingestion(load_model_catalog())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "Unable to read ingestion YAML; check its path and syntax."
        ) from exc
    except ValidationError as exc:
        raise ConfigurationError(_validation_locations("ingestion", exc)) from exc


def load_model_catalog(path: Path | None = None) -> ModelCatalog:
    """Read an explicit catalog or convert standalone model files into one."""
    try:
        if path is None and _standalone_models_directory().is_dir():
            return load_configuration_registry().model_catalog()
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
            the checked-in or packaged ``query_basic_v1`` pipeline is used.
        catalog: Shared model catalog; the packaged catalog is used by default.
        registry: Optional component registry, primarily useful for tests.

    Raises:
        ConfigurationError: If YAML, schema, component, or model validation fails.
    """
    try:
        path = path or _default_pipeline_path()
        value = _yaml_content(path, "")
        standalone = load_configuration_registry()
        config = standalone.pipeline(
            cast(DeclarativePipelineConfig, _model_validate(value, "pipeline"))
        )
        model_catalog = catalog or standalone.model_catalog()
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
    root = directory or _checkout_root() / "config/pipelines"
    if directory is None and not root.is_dir():
        packaged_standalone = (
            files("llm_grounding.infrastructure.config") / "standalone/pipelines"
        )
        if packaged_standalone.is_dir():
            root = Path(str(packaged_standalone))
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
        if not resource.is_file():
            raise FileNotFoundError(
                f"Packaged configuration is missing: {packaged_name}"
            )
        content = resource.read_text(encoding="utf-8")
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
        if path.stem == "query_basic_v1":
            return path
    raise ConfigurationError("No default query pipeline configuration exists")


def _default_ingestion_pipeline_path() -> Path:
    """Return the canonical ingestion pipeline used by the execution service."""
    paths = discover_pipeline_paths()
    for path in paths:
        if path.stem == "ingestion_default_v1":
            return path
    raise ConfigurationError("No default ingestion pipeline configuration exists")


def load_configuration_registry(
    models_directory: Path | None = None,
    components_directory: Path | None = None,
) -> ConfigurationRegistry:
    """Load the canonical standalone model and component registries."""
    try:
        return ConfigurationRegistry.from_directories(
            models_directory or _standalone_models_directory(),
            components_directory or _standalone_components_directory(),
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _standalone_models_directory() -> Path:
    """Return the checkout or packaged standalone model directory."""
    checkout = _checkout_root() / "config/models"
    if checkout.is_dir():
        return checkout
    packaged = files("llm_grounding.infrastructure.config") / "standalone/models"
    return Path(str(packaged))


def _standalone_components_directory() -> Path:
    """Return the checkout or packaged standalone component directory."""
    checkout = _checkout_root() / "config/components"
    if checkout.is_dir():
        return checkout
    packaged = files("llm_grounding.infrastructure.config") / "standalone/components"
    return Path(str(packaged))


def _model_validate(
    value: object, kind: str
) -> DeclarativePipelineConfig:
    """Validate one canonical source model with loader-level error formatting."""
    try:
        return DeclarativePipelineConfig.model_validate(value)
    except ValidationError as exc:
        raise ConfigurationError(_validation_locations(kind, exc)) from exc


def _validation_locations(kind: str, error: ValidationError) -> str:
    """Format only schema locations, never invalid input values."""
    locations = ", ".join(
        ".".join(str(part) for part in issue["loc"]) or "root"
        for issue in error.errors(include_input=False)
    )
    return f"Invalid {kind} configuration at: {locations}"
