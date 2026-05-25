"""LLM call tracing — pretty stdout panels + structured JSONL on disk.

Why a custom callback handler?
    LangChain already exposes a ``BaseCallbackHandler`` lifecycle for every
    LLM call (start, end, error, token, tool). The framework's built-in
    handlers either don't capture full prompts/responses or render them
    awkwardly. For a *learning* repo the entire point is to see the prompt
    that went over the wire and the raw response, so we roll our own.

Each LLM call produces one ``TraceRecord``. Records are:

* appended to a JSONL file under ``settings.trace_dir`` — one file per run
* and (optionally) rendered to stdout as a ``rich`` panel with colour-coded
  sections for system / user / assistant / tool messages.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import LLMResult
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------


@dataclass
class _MessageView:
    """A flat, JSON-friendly view of a chat message."""

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    name: str | None = None  # used by tool messages


@dataclass
class TraceRecord:
    """One LLM round-trip, captured in full."""

    run_id: str
    parent_run_id: str | None
    provider_class: str
    model_name: str | None
    started_at: float
    duration_s: float
    prompt: list[_MessageView]
    response: _MessageView | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------


def _msg_to_view(msg: BaseMessage) -> _MessageView:
    """Reduce a LangChain message to a serialisable view.

    LangChain messages can carry rich content (lists of parts, tool calls,
    function calls, response metadata). We pull out the bits a human
    reviewing a trace actually cares about.
    """
    role = msg.type  # 'system', 'human', 'ai', 'tool'
    content = msg.content if isinstance(msg.content, str) else str(msg.content)

    tool_calls: list[dict[str, Any]] = []
    if isinstance(msg, AIMessage):
        # `tool_calls` is the canonical, provider-agnostic shape exposed by
        # langchain-core. Each entry has name/args/id.
        for tc in getattr(msg, "tool_calls", []) or []:
            tool_calls.append(
                {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id"),
                }
            )

    name = None
    if isinstance(msg, ToolMessage):
        name = msg.name

    return _MessageView(role=role, content=content, tool_calls=tool_calls, name=name)


# ---------------------------------------------------------------------------
# The callback handler itself
# ---------------------------------------------------------------------------


class TracingCallbackHandler(BaseCallbackHandler):
    """Capture every chat model call to disk + (optionally) stdout.

    Lifecycle:
        on_chat_model_start --> remember prompt + start time
        on_llm_end          --> compute duration, record response, flush
        on_llm_error        --> record error, flush

    The handler is safe to share across many concurrent calls because we
    key all per-call state on ``run_id`` (a UUID supplied by LangChain).
    """

    # We always set this so the framework knows to call us for chat models
    # in addition to legacy LLMs.
    raise_error = False
    run_inline = True

    def __init__(
        self,
        trace_dir: Path,
        *,
        stdout: bool = True,
        run_label: str | None = None,
        console: Console | None = None,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.run_label = run_label or time.strftime("%Y%m%d-%H%M%S")
        self.file_path = self.trace_dir / f"trace-{self.run_label}-{uuid.uuid4().hex[:6]}.jsonl"
        self._console = console or Console(highlight=False, soft_wrap=False)
        self._stdout = stdout

        # run_id -> partial state captured at start
        self._pending: dict[UUID, dict[str, Any]] = {}

        # Records collected this session (useful in tests).
        self.records: list[TraceRecord] = []

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # `messages` is a list-of-lists because LangChain supports batching
        # multiple prompts in one call. We virtually never batch in this
        # agent, so we just take the first prompt.
        prompt = messages[0] if messages else []
        provider_class = (serialized.get("id") or ["unknown"])[-1]
        model_name = (
            (serialized.get("kwargs") or {}).get("model")
            or (serialized.get("kwargs") or {}).get("model_name")
        )
        self._pending[run_id] = {
            "parent_run_id": parent_run_id,
            "started_at": time.time(),
            "provider_class": provider_class,
            "model_name": model_name,
            "prompt": [_msg_to_view(m) for m in prompt],
        }

    # ------------------------------------------------------------------
    # End
    # ------------------------------------------------------------------
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._pending.pop(run_id, None)
        if start is None:
            return  # Not a chat-model call we tracked.

        duration = time.time() - start["started_at"]

        # Pull the assistant message out of the first generation.
        ai_view: _MessageView | None = None
        gens = response.generations[0] if response.generations else []
        if gens:
            gen0 = gens[0]
            # ChatGeneration carries the message under `.message`.
            ai_msg = getattr(gen0, "message", None)
            if ai_msg is not None:
                ai_view = _msg_to_view(ai_msg)
            else:
                ai_view = _MessageView(role="ai", content=str(gen0.text))

        usage = (response.llm_output or {}).get("token_usage") or {}
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
        )
        total_tokens = usage.get("total_tokens")

        record = TraceRecord(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            provider_class=start["provider_class"],
            model_name=start["model_name"],
            started_at=start["started_at"],
            duration_s=duration,
            prompt=start["prompt"],
            response=ai_view,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        self._emit(record)

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._pending.pop(run_id, None)
        if start is None:
            return

        record = TraceRecord(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            provider_class=start["provider_class"],
            model_name=start["model_name"],
            started_at=start["started_at"],
            duration_s=time.time() - start["started_at"],
            prompt=start["prompt"],
            response=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            error=f"{type(error).__name__}: {error}",
        )
        self._emit(record)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def _emit(self, record: TraceRecord) -> None:
        self.records.append(record)
        # JSONL — one record per line.
        with self.file_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), default=str) + "\n")
        if self._stdout:
            self._render(record)

    def _render(self, record: TraceRecord) -> None:
        header = Text.assemble(
            ("LLM call ", "bold"),
            (f"{record.provider_class}", "cyan"),
            (" · ", "dim"),
            (f"{record.model_name or '?'}", "yellow"),
            (" · ", "dim"),
            (f"{record.duration_s * 1000:.0f} ms", "magenta"),
        )
        if record.input_tokens is not None or record.output_tokens is not None:
            header.append(" · ", style="dim")
            header.append(
                f"tokens in={record.input_tokens} out={record.output_tokens}",
                style="green",
            )

        bits: list[Any] = [header, Rule(style="dim")]

        # Prompt — one panel per message, colour by role.
        role_styles = {
            "system": ("magenta", "system"),
            "human": ("blue", "human"),
            "ai": ("green", "assistant"),
            "tool": ("yellow", "tool"),
        }
        for m in record.prompt:
            colour, label = role_styles.get(m.role, ("white", m.role))
            title = f"[bold {colour}]{label}[/bold {colour}]"
            if m.name:
                title += f" [dim]({escape(m.name)})[/dim]"
            content = escape(m.content) if m.content else "[dim](empty)[/dim]"
            bits.append(Panel(content, title=title, border_style=colour, expand=True))

        bits.append(Rule(" response ", style="dim"))

        if record.error:
            bits.append(Panel(escape(record.error), title="[bold red]error[/]", border_style="red"))
        elif record.response is not None:
            resp_content = escape(record.response.content) if record.response.content else "[dim](no text content)[/]"
            bits.append(
                Panel(
                    resp_content,
                    title="[bold green]assistant[/]",
                    border_style="green",
                )
            )
            for tc in record.response.tool_calls:
                args_repr = json.dumps(tc.get("args"), indent=2, default=str)
                bits.append(
                    Panel(
                        f"[bold]{escape(tc.get('name', ''))}[/]\n{escape(args_repr)}",
                        title="[bold yellow]tool_call[/]",
                        border_style="yellow",
                    )
                )

        self._console.print(Panel(Group(*bits), border_style="bright_black"))


def render_records(records: Sequence[TraceRecord], console: Console | None = None) -> None:
    """Convenience: re-render previously captured records (e.g. from JSONL).

    Useful in tests and in a future viewer script that reads a JSONL file
    back from disk.
    """
    handler = TracingCallbackHandler.__new__(TracingCallbackHandler)  # bare instance
    handler._console = console or Console(highlight=False, soft_wrap=False)
    handler._stdout = True
    for r in records:
        handler._render(r)
