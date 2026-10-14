"""Resolve Aqueduct credentials at call time without retaining them in run records."""

import os
from pathlib import Path


def resolve_api_key() -> str:
    """Prefer AQUEDUCT_API_KEY, falling back to a UTF-8 AQUEDUCT_API_KEY_FILE.

    Whitespace-only values are missing. File content is stripped. No dotenv file
    is implicitly loaded, and credentials are never logged.

    Raises:
        RuntimeError: If neither source supplies a nonempty, readable key.
    """
    api_key = os.environ.get("AQUEDUCT_API_KEY", "").strip()
    if api_key:
        return api_key
    key_path = os.environ.get("AQUEDUCT_API_KEY_FILE", "").strip()
    if key_path:
        try:
            path = Path(key_path).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    "AQUEDUCT_API_KEY_FILE must refer to an existing file."
                )
            api_key = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                "Unable to read AQUEDUCT_API_KEY_FILE as UTF-8."
            ) from exc
        if api_key:
            return api_key
        raise RuntimeError("AQUEDUCT_API_KEY_FILE contains an empty key.")
    raise RuntimeError(
        "Set AQUEDUCT_API_KEY or AQUEDUCT_API_KEY_FILE to call Aqueduct."
    )


def resolve_embedding_key() -> str:
    """Resolve embedding credentials with an embedding-specific fallback.

    ``EMBEDDING_API_KEY`` and ``EMBEDDING_API_KEY_FILE`` take precedence. The
    Aqueduct names remain supported so existing deployments can use the same
    credential for reasoning and embedding requests.

    Raises:
        RuntimeError: If neither embedding-specific nor Aqueduct credentials
            provide a nonempty, readable key.
    """
    api_key = os.environ.get("EMBEDDING_API_KEY", "").strip()
    if api_key:
        return api_key
    key_path = os.environ.get("EMBEDDING_API_KEY_FILE", "").strip()
    if key_path:
        try:
            path = Path(key_path).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    "EMBEDDING_API_KEY_FILE must refer to an existing file."
                )
            api_key = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                "Unable to read EMBEDDING_API_KEY_FILE as UTF-8."
            ) from exc
        if api_key:
            return api_key
        raise RuntimeError("EMBEDDING_API_KEY_FILE contains an empty key.")
    return resolve_api_key()
