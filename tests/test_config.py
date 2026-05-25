"""Config loading: env parsing, defaults, unknown providers."""

from __future__ import annotations

import pytest

from coding_agent.config import load_settings


def test_defaults_when_no_env(monkeypatch, tmp_path):
    # Wipe every variable load_settings might look at.
    for key in [
        "CODING_AGENT_PROVIDER",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "CODING_AGENT_MAX_ITERATIONS",
        "CODING_AGENT_TOOL_OUTPUT_LIMIT",
        "CODING_AGENT_TRANSCRIPT_COMPRESS_AT",
        "CODING_AGENT_TRACE_STDOUT",
    ]:
        monkeypatch.delenv(key, raising=False)
    # Point dotenv at a non-existent file so it can't pick up a real .env.
    s = load_settings(dotenv_path=tmp_path / "nope.env")
    assert s.provider == "fake"
    assert s.max_iterations == 12
    assert s.google_api_key is None


def test_unknown_provider_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_AGENT_PROVIDER", "claude")
    with pytest.raises(ValueError, match="Unknown CODING_AGENT_PROVIDER"):
        load_settings(dotenv_path=tmp_path / "nope.env")


def test_openai_provider_recognised(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = load_settings(dotenv_path=tmp_path / "nope.env")
    assert s.provider == "openai"
    assert s.openai_api_key == "sk-test"
