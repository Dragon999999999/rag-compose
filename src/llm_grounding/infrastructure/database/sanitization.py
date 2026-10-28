"""PostgreSQL bind types that remove unsupported null bytes from persisted data."""

from typing import Any

from sqlalchemy import JSON as SQLAlchemyJSON
from sqlalchemy import String as SQLAlchemyString
from sqlalchemy import Text as SQLAlchemyText
from sqlalchemy.dialects.postgresql import JSONB as SQLAlchemyJSONB
from sqlalchemy.types import TypeDecorator


def sanitize_postgres_value(value: Any) -> Any:
    """Remove PostgreSQL-incompatible null bytes recursively from a value."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {
            sanitize_postgres_value(key): sanitize_postgres_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_postgres_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_postgres_value(item) for item in value)
    return value


class String(TypeDecorator):
    """Sanitize null bytes in bounded string columns before binding."""

    impl = SQLAlchemyString
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        """Return a PostgreSQL-safe string value."""
        return sanitize_postgres_value(value)


class Text(TypeDecorator):
    """Sanitize null bytes in text columns before binding."""

    impl = SQLAlchemyText
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        """Return a PostgreSQL-safe text value."""
        return sanitize_postgres_value(value)


class JSON(TypeDecorator):
    """Sanitize nested JSON values before the SQLAlchemy JSON serializer runs."""

    impl = SQLAlchemyJSON
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        """Return recursively sanitized JSON-compatible data."""
        return sanitize_postgres_value(value)


class JSONB(TypeDecorator):
    """Sanitize nested JSONB values before PostgreSQL serialization."""

    impl = SQLAlchemyJSONB
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        """Return recursively sanitized JSONB-compatible data."""
        return sanitize_postgres_value(value)
