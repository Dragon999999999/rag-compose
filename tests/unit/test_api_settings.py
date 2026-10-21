"""Verify environment-backed API server settings."""

import pytest

from llm_grounding.infrastructure.config.settings import ApiSettings


def test_api_settings_use_configured_host_and_port(monkeypatch) -> None:
    """API settings use values supplied by the environment."""
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "8002")

    assert ApiSettings.from_environment() == ApiSettings("0.0.0.0", 8002)


def test_api_settings_default_to_local_port(monkeypatch) -> None:
    """API settings default to the local development bind address and port."""
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    assert ApiSettings.from_environment() == ApiSettings("127.0.0.1", 8002)


@pytest.mark.parametrize("value", ["0", "65536", "not-a-port"])
def test_api_settings_reject_invalid_ports(monkeypatch, value: str) -> None:
    """Invalid API ports fail before Uvicorn starts."""
    monkeypatch.setenv("API_PORT", value)

    with pytest.raises(ValueError, match="API_PORT"):
        ApiSettings.from_environment()
