"""Validated model discovery for local lifecycle management."""

from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.config.model_schema import validate_model_file

_MODEL_ADAPTER: TypeAdapter[ModelDefinition] = TypeAdapter(ModelDefinition)


class ModelRegistry:
    """Resolve model IDs from the same YAML source used by pipeline configuration."""

    def __init__(self, models: dict[str, ModelDefinition]) -> None:
        """Create an immutable-by-convention registry after duplicate checks."""
        self._models = dict(models)

    @classmethod
    def from_directory(cls, directory: Path) -> "ModelRegistry":
        """Discover and schema-validate model YAML files in deterministic order."""
        if not directory.is_dir():
            raise ValueError(
                f"Model configuration directory does not exist: {directory}"
            )
        models: dict[str, ModelDefinition] = {}
        for path in sorted(
            path
            for path in (*directory.glob("*.yaml"), *directory.glob("*.yml"))
            if path.name not in {"schema.yaml", "schema.yml"}
        ):
            try:
                value = validate_model_file(path, directory)
                definition = _MODEL_ADAPTER.validate_python(value)
            except (
                OSError,
                UnicodeError,
                yaml.YAMLError,
                ValidationError,
                ValueError,
            ) as exc:
                raise ValueError(f"Invalid model configuration file: {path}") from exc
            if definition.id in models:
                raise ValueError(f"Duplicate model configuration ID: {definition.id}")
            models[definition.id] = definition
        return cls(models)

    def resolve(self, model_id: str) -> ModelDefinition:
        """Return one model definition or raise an actionable reference error."""
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model configuration ID: {model_id}") from exc

    def list(self) -> tuple[ModelDefinition, ...]:
        """Return all definitions in stable ID order."""
        return tuple(self._models[key] for key in sorted(self._models))
