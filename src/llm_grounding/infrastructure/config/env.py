"""Centralized loading of project environment configuration."""

from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    """Load the repository `.env` without overriding existing environment values."""
    project_root = Path(__file__).resolve().parents[4]
    load_dotenv(dotenv_path=project_root / ".env", override=False)
