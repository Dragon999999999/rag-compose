"""Resolve Aqueduct credentials at call time without retaining them in run records."""

import os
from pathlib import Path


def resolve_api_key(
    env_var: str = "AQUEDUCT_API_KEY",
    file_env_var: str = "AQUEDUCT_API_KEY_FILE",
) -> str:
    """Resolve an environment-backed provider credential.

    Whitespace-only values are missing. File content is stripped. No dotenv file
    is implicitly loaded, and credentials are never logged.

    Args:
        env_var: Environment variable containing the key directly.
        file_env_var: Environment variable containing a UTF-8 key-file path.

    Raises:
        RuntimeError: If neither source supplies a nonempty, readable key.
    """
    api_key = os.environ.get(env_var, "").strip()
    if api_key:
        return api_key
    key_path = os.environ.get(file_env_var, "").strip()
    if key_path:
        try:
            path = Path(key_path).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    f"{file_env_var} must refer to an existing file."
                )
            api_key = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                f"Unable to read {file_env_var} as UTF-8."
            ) from exc
        if api_key:
            return api_key
        raise RuntimeError(f"{file_env_var} contains an empty key.")
    raise RuntimeError(
        f"Set {env_var} or {file_env_var} to call the configured provider."
    )


def resolve_embedding_key() -> str:
    """Resolve the shared Aqueduct credential for backward-compatible callers.

    Embedding providers deliberately use the same ``AQUEDUCT_API_KEY`` or
    ``AQUEDUCT_API_KEY_FILE`` as reasoning providers. This compatibility name
    remains available for existing imports but does not define separate
    embedding credentials.

    Raises:
        RuntimeError: If the shared credential is unavailable.
    """
    return resolve_api_key()
