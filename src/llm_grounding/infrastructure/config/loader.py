"""Read safe YAML into typed composition settings, without constructing components."""

from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.ingestion.models import IngestionConfig


class ConfigurationError(ValueError):
    """Report malformed settings without echoing potentially sensitive input values."""


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
        locations = ", ".join(
            ".".join(str(part) for part in error["loc"]) or "root"
            for error in exc.errors(include_input=False)
        )
        raise ConfigurationError(
            f"Invalid architecture configuration at: {locations}"
        ) from exc


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
        locations = ", ".join(
            ".".join(str(part) for part in error["loc"]) or "root"
            for error in exc.errors(include_input=False)
        )
        raise ConfigurationError(
            f"Invalid ingestion configuration at: {locations}"
        ) from exc
