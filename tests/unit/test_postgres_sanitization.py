"""Verify PostgreSQL-bound strings and metadata are null-byte safe."""

from datetime import datetime

from llm_grounding.infrastructure.database.models import (
    DocumentRow,
    PipelineRow,
)
from llm_grounding.infrastructure.database.sanitization import (
    sanitize_postgres_value,
)


def test_sanitize_postgres_value_removes_nested_null_bytes() -> None:
    """Nested metadata containers retain shape while losing null bytes."""
    value = {
        "key\x00": ["value\x00", ("tuple\x00", None)],
        "unchanged": 42,
    }

    assert sanitize_postgres_value(value) == {
        "key": ["value", ("tuple", None)],
        "unchanged": 42,
    }


def test_string_and_text_columns_sanitize_values_at_bind_time() -> None:
    """Ordinary input fields are sanitized independently of JSON metadata."""
    name_type = DocumentRow.__table__.c.name.type
    description_type = PipelineRow.__table__.c.description.type

    assert name_type.process_bind_param("document\x00name", None) == "documentname"
    assert description_type.process_bind_param("description\x00", None) == "description"


def test_jsonb_columns_sanitize_metadata_at_bind_time() -> None:
    """Pipeline metadata is sanitized before PostgreSQL JSONB serialization."""
    config_type = PipelineRow.__table__.c.config_json.type
    value = {"prompt": ["safe\x00text", {"label\x00": "value\x00"}]}

    assert config_type.process_bind_param(value, None) == {
        "prompt": ["safetext", {"label": "value"}]
    }


def test_non_string_values_are_preserved() -> None:
    """Sanitization does not coerce dates, numbers, booleans, or nulls."""
    value = (None, True, 3, 1.5, datetime(2026, 1, 1))

    assert sanitize_postgres_value(value) == value
