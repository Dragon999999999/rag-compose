"""Validate standalone model YAML against the shared Draft 2020-12 schema."""

from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource


def validate_model_file(path: Path, models_directory: Path) -> dict[str, Any]:
    """Parse and validate one model definition with a path-aware error."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema_path = models_directory / "schema.yaml"
        if not schema_path.is_file():
            schema_path = (
                Path(__file__).resolve().parents[4] / "config/models/schema.yaml"
            )
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("model YAML must contain a mapping")
        if not isinstance(schema, dict):
            raise ValueError("model schema YAML must contain a mapping")
        value = _normalize_legacy_value(value)
        schema_uri = schema_path.as_uri()
        schema["$id"] = schema_uri
        validator = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
            registry=Registry().with_resources(
                [(schema_uri, Resource.from_contents(schema))]
            ),
        )
        errors = sorted(validator.iter_errors(value), key=lambda error: error.path)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ValueError(
            f"Unable to validate model configuration file: {path}"
        ) from exc
    if errors:
        locations = ", ".join(
            ".".join(str(part) for part in error.path) or "root" for error in errors
        )
        messages = "; ".join(error.message for error in errors[:3])
        raise ValueError(
            f"Invalid model configuration {path.name} at {locations}: {messages}"
        )
    return value


def _normalize_legacy_value(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize pre-standard model mappings before schema validation."""
    if "location" in value:
        return value
    normalized = dict(value)
    backend = normalized.get("backend")
    normalized["location"] = "remote" if backend == "openai_compatible" else "local"
    parameter_names = {
        "dimension",
        "dimensions",
        "normalize",
        "normalization",
        "batch_size",
        "embedding_space_id",
        "representation",
        "similarity_metric",
    }
    parameters = dict(normalized.get("parameters", {}))
    for name in parameter_names:
        if name in normalized:
            parameters[name] = normalized.pop(name)
    normalized["parameters"] = parameters
    if "base_url" in normalized:
        normalized["connection"] = {
            "endpoint": normalized.pop("base_url"),
            "auth": normalized.pop("auth", {}),
            "timeout_seconds": normalized.pop("timeout_seconds", 120),
            "retries": normalized.pop("retries", 2),
        }
    if normalized["location"] == "local":
        normalized.setdefault(
            "source", {"type": "huggingface", "repository": normalized["name"]}
        )
        normalized.setdefault(
            "storage", {"relative_path": f"{normalized['stage']}/{normalized['id']}"}
        )
        normalized.setdefault(
            "runtime", {"type": "in_process", "idle_timeout_seconds": 0}
        )
    return normalized
