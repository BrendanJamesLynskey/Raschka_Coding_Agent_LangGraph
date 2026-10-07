"""LLM factory: the Ollama branch.

Nothing here talks to a real Ollama server. The "unreachable" test points
at a loopback port that is guaranteed closed, so the connection is refused
instantly without leaving the machine; the construction test stubs out
langchain-ollama's model check.
"""

from __future__ import annotations

import dataclasses
import socket
import sys

import pytest

from coding_agent.llm import build_chat_model


def _ollama_settings(settings, **overrides):
    return dataclasses.replace(settings, provider="ollama", **overrides)


def _closed_loopback_port() -> int:
    # Bind to an ephemeral port, then release it: nothing is listening.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_missing_dependency_gives_install_hint(settings, monkeypatch):
    # A `None` entry in sys.modules makes `import langchain_ollama` raise
    # ImportError, exactly as if the [ollama] extra weren't installed.
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[ollama\]'"):
        build_chat_model(_ollama_settings(settings))


def test_factory_builds_chat_ollama_with_settings(settings, monkeypatch):
    langchain_ollama = pytest.importorskip("langchain_ollama")
    # Skip the "is the server up / model pulled?" round trip.
    monkeypatch.setattr(
        langchain_ollama.chat_models, "validate_model", lambda client, name: None
    )
    model = build_chat_model(
        _ollama_settings(
            settings,
            ollama_model="qwen2.5:0.5b",
            ollama_base_url="http://example.invalid:11434",
            ollama_num_ctx=8192,
        )
    )
    assert isinstance(model, langchain_ollama.ChatOllama)
    assert model.model == "qwen2.5:0.5b"
    assert model.base_url == "http://example.invalid:11434"
    assert model.num_ctx == 8192
    assert model.temperature == 0.0
    # And it must accept tools, since `choose` always binds them.
    assert model.bind_tools([]) is not None


def test_unreachable_server_fails_fast_with_hint(settings):
    pytest.importorskip("langchain_ollama")
    url = f"http://127.0.0.1:{_closed_loopback_port()}"
    with pytest.raises(RuntimeError, match="ollama serve") as exc:
        build_chat_model(_ollama_settings(settings, ollama_base_url=url))
    assert url in str(exc.value)
