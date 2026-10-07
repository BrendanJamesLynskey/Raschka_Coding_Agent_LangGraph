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


def test_ollama_provider_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_AGENT_PROVIDER", "ollama")
    for key in ["OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_NUM_CTX"]:
        monkeypatch.delenv(key, raising=False)
    s = load_settings(dotenv_path=tmp_path / "nope.env")
    assert s.provider == "ollama"
    assert s.ollama_model == "qwen3.5:9b"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_num_ctx == 16384


def test_ollama_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_AGENT_PROVIDER", "OLLAMA")  # case-insensitive
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    s = load_settings(dotenv_path=tmp_path / "nope.env")
    assert s.provider == "ollama"
    assert s.ollama_model == "qwen2.5:0.5b"
    assert s.ollama_base_url == "http://gpu-box:11434"
    assert s.ollama_num_ctx == 32768


def test_unknown_provider_message_lists_ollama(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_AGENT_PROVIDER", "llamacpp")
    with pytest.raises(ValueError, match="ollama"):
        load_settings(dotenv_path=tmp_path / "nope.env")
