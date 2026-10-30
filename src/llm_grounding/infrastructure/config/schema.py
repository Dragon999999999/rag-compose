"""Validate standalone component YAML against local Draft 2020-12 schemas.

JSON Schema provides the static source contract, including override metadata.
Pydantic remains responsible for the typed runtime configuration produced after
component references and model definitions have been resolved.
"""

from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource


def validate_component_file(
    path: Path,
    family: str,
    components_directory: Path,
) -> dict[str, Any]:
    """Validate one component definition against its family schema.

    Args:
        path: YAML file containing the component instance.
        family: Component-family directory name.
        components_directory: Root containing the parent schema and families.

    Returns:
        The parsed component mapping for subsequent Pydantic validation.

    Raises:
        ValueError: If the YAML, schema, or component instance is invalid.
    """
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema_path = components_directory / family / "schema.yaml"
        parent_schema_path = components_directory / "schema.yaml"
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        parent_schema = yaml.safe_load(parent_schema_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("component YAML must contain a mapping")
        if not isinstance(schema, dict) or not isinstance(parent_schema, dict):
            raise ValueError("component schema YAML must contain a mapping")
        schema_uri = schema_path.as_uri()
        parent_uri = parent_schema_path.as_uri()
        schema["$id"] = schema_uri
        parent_schema["$id"] = parent_uri
        validator = Draft202012Validator(
            schema,
            registry=Registry().with_resources(
                [
                    (schema_uri, Resource.from_contents(schema)),
                    (parent_uri, Resource.from_contents(parent_schema)),
                ]
            ),
        )
        errors = sorted(validator.iter_errors(value), key=lambda error: error.path)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ValueError(
            f"Unable to validate component configuration file: {path}"
        ) from exc
    if errors:
        locations = ", ".join(
            ".".join(str(part) for part in error.path) or "root"
            for error in errors
        )
        messages = "; ".join(error.message for error in errors[:3])
        raise ValueError(
            f"Invalid {family} component {path.name} at {locations}: {messages}"
        )
    return value
