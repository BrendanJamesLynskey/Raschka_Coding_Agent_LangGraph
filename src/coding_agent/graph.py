"""The LangGraph wiring — observe -> inspect -> choose -> act -> record loop.

This file is the one-page picture of the whole agent. If you only read one
module to understand the implementation, read this one.

Node responsibilities (mapped to the article)
--------------------------------------------
* ``observe``  — Component 1: collect workspace facts on the first turn.
* ``choose``   — The "model" layer: builds the prompt (Component 2), calls
                 the LLM, captures any tool calls it wants to make.
* ``act``      — Component 3: validates and executes tool calls, clips
                 their output (Component 4), records tool_call /
                 tool_result pairs.
* ``compress`` — Component 4: shrinks the transcript when it grows past
                 ``settings.transcript_compress_at``.
* ``finalize`` — Lifts the model's final answer out of the transcript.

Component 6 (bounded subagents) adds no node: it is a *tool*,
``spawn_subagent``, that runs this same graph again at ``depth=1`` with a
read-only toolset. See ``subagents.py``.

Edges
-----
    START ──> observe ──> choose
    choose ──(tool calls?)──> act ──> compress ──> choose
    choose ──(no tool calls)──> finalize ──> END

Why no separate ``inspect`` node?
    The article distinguishes "observe" (gather environment) from "inspect"
    (look at memory/transcript). In this implementation that distinction
    collapses into ``choose``: building the prompt and inspecting state are
    the same step, and splitting them would just add a no-op node. The
    article's *names* still live in the code as docstring landmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from coding_agent.compression import compress_transcript
from coding_agent.config import Settings
from coding_agent.context import collect_workspace_facts
from coding_agent.llm import build_chat_model
from coding_agent.prompts import build_prompt
from coding_agent.state import (
    AgentState,
    PendingToolCall,
    TranscriptEntry,
    initial_state,
)
from coding_agent.subagents import (
    SUBAGENT_MAX_DEPTH,
    build_spawn_subagent_tool,
    subagent_toolset,
)
from coding_agent.tools import PermissionGate, always_allow, build_tools
from coding_agent.trace import TracingCallbackHandler


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _make_observe(settings: Settings):
    def observe(state: AgentState) -> dict:
        # Only collect facts once — they're "stable" by definition.
        if state["workspace"].root:
            return {}
        facts = collect_workspace_facts(settings.workspace_root)
        return {"workspace": facts}

    return observe


def _make_choose(model, tools: list[BaseTool]):
    """The model call. Returns either tool calls (queued in state) or the
    final answer (placed in ``final_answer``).
    """
    # `bind_tools` is provider-agnostic: it asks the model to return tool
    # calls in the canonical `AIMessage.tool_calls` format.
    bound_model = model.bind_tools(tools)

    def choose(state: AgentState) -> dict:
        # Iteration guard — refuse to loop forever, even if the model
        # never decides to stop.
        if state["iterations"] >= state["max_iterations"]:
            return {
                "final_answer": (
                    "Stopped: reached max_iterations "
                    f"({state['max_iterations']}). "
                    "Increase CODING_AGENT_MAX_ITERATIONS or simplify the task."
                ),
                # Machine-readable reason, so callers (e.g. the subagent
                # tool) needn't parse the human-readable message above.
                "error": "max_iterations",
            }

        prompt = build_prompt(state)
        ai_message: AIMessage = bound_model.invoke(prompt)

        # Start from the current transcript and extend it — we replace
        # the field wholesale (no add-reducer in use).
        new_transcript: list[TranscriptEntry] = list(state["transcript"])

        # On the first turn, log the user request so it survives compression.
        if not new_transcript:
            new_transcript.append(
                TranscriptEntry(kind="user", content=state["user_request"])
            )

        tool_calls = list(getattr(ai_message, "tool_calls", []) or [])

        if tool_calls:
            content = ai_message.content if isinstance(ai_message.content, str) else ""
            if content.strip():
                new_transcript.append(
                    TranscriptEntry(kind="assistant_thought", content=content)
                )
            pending = [
                PendingToolCall(
                    id=tc.get("id") or f"call-{i}",
                    name=tc["name"],
                    args=tc.get("args") or {},
                )
                for i, tc in enumerate(tool_calls)
            ]
            return {
                "transcript": new_transcript,
                "pending_tool_calls": pending,
                "iterations": state["iterations"] + 1,
            }

        # No tool calls -> the model is signalling completion.
        content = (
            ai_message.content
            if isinstance(ai_message.content, str)
            else str(ai_message.content)
        )
        new_transcript.append(
            TranscriptEntry(kind="assistant_thought", content=content)
        )
        return {
            "transcript": new_transcript,
            "pending_tool_calls": [],
            "iterations": state["iterations"] + 1,
            "final_answer": content,
        }

    return choose


def _make_act(tools: list[BaseTool]):
    by_name = {t.name: t for t in tools}

    def act(state: AgentState) -> dict:
        new_transcript: list[TranscriptEntry] = list(state["transcript"])
        touched: list[str] = list(state["memory"].files_touched)

        for call in state["pending_tool_calls"]:
            tool = by_name.get(call.name)
            if tool is None:
                result = f"ERROR: unknown tool {call.name!r}."
            else:
                try:
                    result = tool.invoke(call.args)
                except Exception as e:  # pragma: no cover (boundary safety)
                    result = f"ERROR running {call.name}: {type(e).__name__}: {e}"
            if not isinstance(result, str):
                result = str(result)

            new_transcript.append(
                TranscriptEntry(
                    kind="tool_call",
                    content="",
                    metadata={"tool": call.name, "args": call.args, "id": call.id},
                )
            )
            new_transcript.append(
                TranscriptEntry(
                    kind="tool_result",
                    content=result,
                    metadata={"tool": call.name, "id": call.id},
                )
            )

            # Lightweight "files_touched" tracking for working memory.
            if call.name == "write_file":
                p = call.args.get("path")
                if p and p not in touched:
                    touched.append(p)

        updated_memory = state["memory"]
        updated_memory.files_touched = touched

        return {
            "transcript": new_transcript,
            "pending_tool_calls": [],
            "memory": updated_memory,
        }

    return act


def _make_compress(settings: Settings):
    def compress(state: AgentState) -> dict:
        compressed = compress_transcript(
            state["transcript"], threshold=settings.transcript_compress_at
        )
        # Since the transcript field has no reducer, returning it here
        # cleanly replaces the old list — exactly what we want.
        if compressed is state["transcript"]:
            return {}
        return {"transcript": compressed}

    return compress


def _make_finalize():
    def finalize(state: AgentState) -> dict:
        # Nothing to do here for now — `final_answer` is already set by
        # `choose`. This node exists so the graph reads cleanly: every
        # path ends at a node, not at a conditional.
        return {}

    return finalize


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_choose(state: AgentState) -> str:
    if state.get("final_answer"):
        return "finalize"
    if state["pending_tool_calls"]:
        return "act"
    # Safety net: model returned no tool calls AND no final answer.
    return "finalize"


# ---------------------------------------------------------------------------
# Public assembly
# ---------------------------------------------------------------------------


def build_agent_graph(
    settings: Settings,
    *,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
    fake_responses: Sequence[AIMessage] | None = None,
    fake_subagent_responses: Sequence[AIMessage] | None = None,
    permission_gate: PermissionGate = always_allow,
    depth: int = 0,
):
    """Compile the LangGraph state machine for the configured provider.

    ``depth`` is the only difference between the main agent and a
    subagent: ``0`` builds the parent (full toolset + ``spawn_subagent``),
    ``1`` builds a bounded, read-only child. See ``subagents.py``.

    ``fake_subagent_responses`` is the child's script for the ``fake``
    provider. Parent and child need *separate* scripts — a single shared
    list would interleave unpredictably as the child's calls consumed the
    parent's responses. Every child replays the script from the start,
    so a test can reason about each spawn in isolation.
    """
    tools = build_tools(settings, permission_gate=permission_gate)
    if depth > 0:
        tools = subagent_toolset(tools)
    if depth < SUBAGENT_MAX_DEPTH:

        def run_child(task: str, child_settings: Settings, label: str) -> AgentState:
            # Re-enter the public entry point one level deeper: same graph,
            # its own tracer (run_label), its own fake script.
            child_final, _ = run_agent(
                task,
                child_settings,
                fake_responses=fake_subagent_responses,
                permission_gate=permission_gate,
                run_label=label,
                depth=depth + 1,
            )
            return child_final

        tools.append(
            build_spawn_subagent_tool(
                settings, run_child=run_child, permission_gate=permission_gate
            )
        )
    model = build_chat_model(
        settings, callbacks=callbacks, fake_responses=fake_responses
    )

    graph = StateGraph(AgentState)
    graph.add_node("observe", _make_observe(settings))
    graph.add_node("choose", _make_choose(model, tools))
    graph.add_node("act", _make_act(tools))
    graph.add_node("compress", _make_compress(settings))
    graph.add_node("finalize", _make_finalize())

    graph.add_edge(START, "observe")
    graph.add_edge("observe", "choose")
    graph.add_conditional_edges(
        "choose",
        _route_after_choose,
        {"act": "act", "finalize": "finalize"},
    )
    graph.add_edge("act", "compress")
    graph.add_edge("compress", "choose")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(
    user_request: str,
    settings: Settings,
    *,
    fake_responses: Sequence[AIMessage] | None = None,
    fake_subagent_responses: Sequence[AIMessage] | None = None,
    extra_callbacks: Sequence[BaseCallbackHandler] | None = None,
    run_label: str | None = None,
    permission_gate: PermissionGate = always_allow,
    depth: int = 0,
) -> tuple[AgentState, TracingCallbackHandler]:
    """One-call entry point: run the agent and return the final state + tracer.

    Returns both so callers can inspect the full transcript, the trace
    records, or both. ``depth`` is set by ``spawn_subagent``; leave it at
    ``0`` when calling this yourself.
    """
    tracer = TracingCallbackHandler(
        trace_dir=settings.trace_dir,
        stdout=settings.trace_stdout,
        run_label=run_label,
    )
    callbacks = [tracer, *(extra_callbacks or [])]
    compiled = build_agent_graph(
        settings,
        callbacks=callbacks,
        fake_responses=fake_responses,
        fake_subagent_responses=fake_subagent_responses,
        permission_gate=permission_gate,
        depth=depth,
    )
    initial = initial_state(user_request, max_iterations=settings.max_iterations)
    # LangGraph's per-call recursion_limit needs to comfortably exceed the
    # number of node visits (each loop is ~3 nodes).
    final_state = compiled.invoke(
        initial,
        config={
            "callbacks": callbacks,
            "recursion_limit": settings.max_iterations * 6 + 20,
        },
    )
    return final_state, tracer
