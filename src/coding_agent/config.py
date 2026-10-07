"""Configuration loading.

All runtime knobs come from environment variables (typically loaded from a
`.env` file with `python-dotenv`). Keeping configuration in one typed
dataclass means the rest of the code never has to reach into `os.environ`,
which makes tests easy: a test just constructs a `Settings(...)` directly.

Why a dataclass and not pydantic-settings?
    The agent code only needs to *read* settings, not validate them at HTTP
    boundaries. A plain dataclass keeps the dependency surface small and
    keeps the file readable as a learning resource.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Provider = Literal["gemini", "deepseek", "openai", "ollama", "fake"]

# Kept next to the Literal so the two can't drift apart unnoticed.
VALID_PROVIDERS: frozenset[str] = frozenset(
    {"gemini", "deepseek", "openai", "ollama", "fake"}
)


@dataclass(frozen=True)
class Settings:
    """All configuration the agent needs to run.

    `frozen=True` so accidentally mutating settings mid-run is a TypeError —
    a small but real safety net when you start passing settings into nodes.
    """

    # --- Provider selection -------------------------------------------------
    provider: Provider = "fake"

    # --- Gemini -------------------------------------------------------------
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # --- DeepSeek -----------------------------------------------------------
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # --- OpenAI -------------------------------------------------------------
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    # Empty string means "use the SDK default".
    openai_base_url: str = ""

    # --- Ollama (local open-weights models) --------------------------------
    # No API key: Ollama is a local server. See docs/providers.md for why
    # this particular Qwen tag is the default.
    ollama_model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    # Ollama's own default context window is small (a few thousand tokens)
    # and it *silently truncates* anything longer from the front — which
    # would eat our stable system prefix first. We always pass num_ctx.
    ollama_num_ctx: int = 16384

    # --- Agent loop ---------------------------------------------------------
    max_iterations: int = 12
    tool_output_limit: int = 4000
    transcript_compress_at: int = 20

    # --- Subagents (Component 6) -------------------------------------------
    # Each spawned subagent gets its own, much smaller iteration budget...
    subagent_max_iterations: int = 4
    # ...and a parent run may only spawn this many in total.
    max_subagents: int = 3

    # --- Sandbox ------------------------------------------------------------
    # Every file / shell tool is rooted at this directory; paths that try to
    # escape it are rejected. Default is the `examples/workspace/` dir.
    workspace_root: Path = field(
        default_factory=lambda: Path("examples/workspace").resolve()
    )

    # --- Tracing ------------------------------------------------------------
    trace_dir: Path = field(default_factory=lambda: Path("traces").resolve())
    trace_stdout: bool = True


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", ""}


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def load_settings(dotenv_path: str | Path | None = None) -> Settings:
    """Load `.env` (if present) and assemble a `Settings`.

    Pass `dotenv_path=None` to use the default search behaviour of
    `python-dotenv`, or a specific path to load a particular file (handy
    in tests).
    """

    # `override=False` so values already set in the real environment win.
    load_dotenv(dotenv_path=dotenv_path, override=False)

    provider_raw = os.environ.get("CODING_AGENT_PROVIDER", "fake").strip().lower()
    if provider_raw not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown CODING_AGENT_PROVIDER={provider_raw!r}. "
            "Expected one of: gemini, deepseek, openai, ollama, fake."
        )

    workspace_root = Path(
        os.environ.get("CODING_AGENT_WORKSPACE", "examples/workspace")
    ).resolve()
    trace_dir = Path(os.environ.get("CODING_AGENT_TRACE_DIR", "traces")).resolve()

    return Settings(
        provider=provider_raw,  # type: ignore[arg-type]
        google_api_key=os.environ.get("GOOGLE_API_KEY") or None,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY") or None,
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ),
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", ""),
        ollama_model=os.environ.get("OLLAMA_MODEL") or "qwen3.5:9b",
        ollama_base_url=(
            os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        ),
        ollama_num_ctx=_get_int("OLLAMA_NUM_CTX", 16384),
        max_iterations=_get_int("CODING_AGENT_MAX_ITERATIONS", 12),
        tool_output_limit=_get_int("CODING_AGENT_TOOL_OUTPUT_LIMIT", 4000),
        transcript_compress_at=_get_int("CODING_AGENT_TRANSCRIPT_COMPRESS_AT", 20),
        subagent_max_iterations=_get_int("CODING_AGENT_SUBAGENT_MAX_ITERATIONS", 4),
        max_subagents=_get_int("CODING_AGENT_MAX_SUBAGENTS", 3),
        workspace_root=workspace_root,
        trace_dir=trace_dir,
        trace_stdout=_get_bool("CODING_AGENT_TRACE_STDOUT", True),
    )
