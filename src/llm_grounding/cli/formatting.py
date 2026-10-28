"""Small, secret-safe output helpers for CLI command modules."""

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

_SENSITIVE_KEYS = {
    "api_key",
    "access_key",
    "secret",
    "secret_key",
    "password",
    "token",
    "credential",
}


def safe_value(value: object, *, key: str | None = None) -> object:
    """Convert a value to JSON-compatible data without exposing credentials."""
    if key is not None:
        normalized_key = key.casefold()
        if normalized_key in _SENSITIVE_KEYS or normalized_key.endswith("_token"):
            return "<redacted>"
        if normalized_key == "environment":
            return "<configured environment>"
    if isinstance(value, BaseModel):
        return safe_value(value.model_dump(mode="json"))
    if is_dataclass(value):
        return safe_value(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {
            str(child_key): safe_value(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [safe_value(child) for child in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def print_json(value: object) -> None:
    """Print one indented, secret-safe JSON document."""
    print(json.dumps(safe_value(value), default=str, indent=2, sort_keys=True))


def print_table(headers: Sequence[object], rows: Iterable[Sequence[object]]) -> None:
    """Print rows with each column aligned to its widest cell.

    Headers are always printed, including when there are no data rows. Values
    are rendered as single-line text so a long description cannot shift the
    following columns or break the table layout.

    Args:
        headers: Column labels in display order.
        rows: Data rows containing one value per header.

    Raises:
        ValueError: If a row does not contain exactly one value per header.
    """
    normalized_headers = [_table_value(value) for value in headers]
    normalized_rows = [
        [_table_value(value) for value in row]
        for row in rows
    ]
    column_count = len(normalized_headers)
    for row in normalized_rows:
        if len(row) != column_count:
            raise ValueError(
                "CLI table rows must contain exactly one value per header"
            )

    if not normalized_headers:
        return

    widths = [len(header) for header in normalized_headers]
    for row in normalized_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        ).rstrip()

    print(render(normalized_headers))
    for row in normalized_rows:
        print(render(row))


def _table_value(value: object) -> str:
    """Convert a table value to safe, single-line display text."""
    if value is None:
        return "-"
    return str(value).replace("\r", " ").replace("\n", " ")


def parse_json_value(raw: str) -> Any:
    """Parse a JSON CLI value, including strings and JSON null."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Override values must be valid JSON") from exc
