"""Verify credential precedence, UTF-8 reading, and clear failures offline."""

from pathlib import Path

import pytest

from llm_grounding.infrastructure.llm.credentials import (
    resolve_api_key,
    resolve_embedding_key,
)


def test_environment_key_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonempty environment key bypasses even an invalid key-file path."""
    monkeypatch.setenv("AQUEDUCT_API_KEY", "  test-environment-key \n")
    monkeypatch.setenv("AQUEDUCT_API_KEY_FILE", "/missing")
    assert resolve_api_key() == "test-environment-key"


def test_embedding_resolution_uses_the_shared_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility embedding resolver does not introduce another key."""
    monkeypatch.setenv("AQUEDUCT_API_KEY", "shared-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "separate-key")

    assert resolve_embedding_key() == "shared-key"


def test_key_file_fallback_is_stripped_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only environment values permit the file fallback."""
    path = tmp_path / "key.txt"
    path.write_text("  test-file-key \n", encoding="utf-8")
    monkeypatch.setenv("AQUEDUCT_API_KEY", " \n")
    monkeypatch.setenv("AQUEDUCT_API_KEY_FILE", str(path))
    assert resolve_api_key() == "test-file-key"


@pytest.mark.parametrize(
    "kind", ["unset", "missing", "directory", "empty", "invalid_utf8"]
)
def test_missing_or_unreadable_credentials_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Credential errors never fall through to an anonymous provider request."""
    path = tmp_path / "key.txt"
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.write_text(" \n", encoding="utf-8")
    elif kind == "invalid_utf8":
        path.write_bytes(b"\xff")
    if kind != "unset":
        monkeypatch.setenv("AQUEDUCT_API_KEY_FILE", str(path))
    with pytest.raises(RuntimeError, match="AQUEDUCT_API_KEY"):
        resolve_api_key()
