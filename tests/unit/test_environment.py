"""Verify centralized project environment loading semantics."""

from pathlib import Path

from llm_grounding.infrastructure.config import env


def test_load_environment_uses_repository_dotenv_without_overriding(
    monkeypatch,
) -> None:
    """The loader targets the repository `.env` and preserves existing values."""
    calls: list[dict[str, object]] = []

    def record_load(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(env, "load_dotenv", record_load)

    env.load_environment()

    assert calls == [
        {
            "dotenv_path": Path(__file__).resolve().parents[2] / ".env",
            "override": False,
        }
    ]
