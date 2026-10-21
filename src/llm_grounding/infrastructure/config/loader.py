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
    DeclarativeArchitectureConfig,
    DeclarativePipelineConfig,
)
from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
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


def load_architecture_config(path: Path | None = None) -> ArchitectureConfig:
    """Read explicit YAML or the packaged default, validating its shape.

    The checked-in YAML is bundled in wheels; editable checkouts read that same
    file. Component identifiers are validated separately by the registry.

    Raises:
        ConfigurationError: If YAML cannot be read or fails typed validation.
    """
    try:
        value = yaml.safe_load(_architecture_content(path))
        if isinstance(value, dict) and value.get("schema_version") == 2:
            return load_configuration_registry().architecture(
                cast(
                    DeclarativeArchitectureConfig,
                    _model_validate(value, "architecture"),
                )
            )
        return ArchitectureConfig.model_validate(value)
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
    """Read the legacy catalog or convert standalone model files into one."""
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
            the checked-in or packaged ``basic_rag_v1`` pipeline is used.
        catalog: Shared model catalog; the packaged catalog is used by default.
        registry: Optional component registry, primarily useful for tests.

    Raises:
        ConfigurationError: If YAML, schema, component, or model validation fails.
    """
    try:
        path = path or _default_pipeline_path()
        value = _yaml_content(path, "")
        if isinstance(value, dict) and value.get("schema_version") == 2:
            standalone = load_configuration_registry()
            config = standalone.pipeline(
                cast(DeclarativePipelineConfig, _model_validate(value, "pipeline"))
            )
            model_catalog = catalog or standalone.model_catalog()
        else:
            config = PipelineConfig.model_validate(value)
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
    root = directory or _checkout_root() / "config/pipelines"
    if directory is None and not root.is_dir():
        root = _checkout_root() / "configs/pipelines"
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


def _architecture_content(path: Path | None) -> str:
    """Load canonical architecture YAML, falling back to the legacy resource."""
    if path is not None:
        return path.read_text(encoding="utf-8")
    canonical = _checkout_root() / "config/architectures/baseline_v0.yaml"
    if canonical.is_file():
        return canonical.read_text(encoding="utf-8")
    packaged_canonical = (
        files("llm_grounding.infrastructure.config")
        / "standalone/architectures/baseline_v0.yaml"
    )
    if packaged_canonical.is_file():
        return packaged_canonical.read_text(encoding="utf-8")
    resource = files("llm_grounding.infrastructure.config") / "baseline_v0.yaml"
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    return (
        _checkout_root() / "configs/architectures/baseline_v0.yaml"
    ).read_text(encoding="utf-8")


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
) -> DeclarativeArchitectureConfig | DeclarativePipelineConfig:
    """Validate one canonical source model with loader-level error formatting."""
    model = (
        DeclarativeArchitectureConfig
        if kind == "architecture"
        else DeclarativePipelineConfig
    )
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        raise ConfigurationError(_validation_locations(kind, exc)) from exc


def _validation_locations(kind: str, error: ValidationError) -> str:
    """Format only schema locations, never invalid input values."""
    locations = ", ".join(
        ".".join(str(part) for part in issue["loc"]) or "root"
        for issue in error.errors(include_input=False)
    )
    return f"Invalid {kind} configuration at: {locations}"
