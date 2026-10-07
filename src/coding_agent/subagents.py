"""Bounded subagents — Raschka's "Delegation with Bounded Subagents".

The article's sixth component: let the main agent hand a self-contained
side-question ("which files mention X?", "summarise this module") to a
*subagent*, and get back only the answer — not the subagent's whole
transcript. Delegation keeps the parent's context small, because the
dozens of tool results the child read never enter the parent's prompt.

The catch is the word *bounded*. An unbounded subagent is just a second
agent with the same powers and its own budget, which multiplies every
risk the parent already has. So every bound here is enforced in code,
and each one has a test in ``tests/test_subagents.py``:

==========================  ==============================================
Bound                       Where it is enforced
==========================  ==============================================
Read-only toolset           ``subagent_toolset`` — an *allowlist* of
                            ``list_dir`` + ``read_file``. ``write_file``,
                            ``run_shell``, MCP tools and ``spawn_subagent``
                            are never bound to the child's model, and the
                            child's ``act`` node rejects them as unknown.
Depth limit 1               ``SUBAGENT_MAX_DEPTH``: ``build_agent_graph``
                            only adds ``spawn_subagent`` when
                            ``depth < SUBAGENT_MAX_DEPTH``. Children are
                            built at ``depth=1``, so they can't spawn.
Own iteration cap           ``subagent_settings`` swaps in
                            ``settings.subagent_max_iterations``.
Per-run spawn cap           A counter in the ``spawn_subagent`` closure;
                            past ``settings.max_subagents`` the tool
                            returns an ERROR string (never raises).
Same sandbox                The child inherits ``workspace_root`` — no
                            setting is changed except the iteration cap.
Clipped result              ``_clip(..., settings.tool_output_limit)``
                            before the answer goes back to the parent.
==========================  ==============================================

How the child is built
----------------------
There is no second graph definition. The child is the *same*
``build_agent_graph`` / ``run_agent`` pipeline, called with ``depth=1``.
The depth is the only "role" switch, and it decides the toolset.

This module doesn't import ``graph.py`` (that would be a circular
import: the graph imports this module to get the tool). Instead the graph
hands us a ``run_child`` callable — plain dependency injection, which
also lets the unit tests swap in a stub runner when they only care about
the bookkeeping.

Tracing
-------
Each child runs with its own ``TracingCallbackHandler`` labelled
``subagent-1``, ``subagent-2``, … so the parent and every child land in
separate JSONL files under ``traces/``.

One subtlety. The child is invoked from *inside* the parent's ``act``
node, and LangChain propagates the active run config — callbacks
included — to nested runnables through a ``contextvars`` variable. Left
alone, every child LLM call would also be reported to the *parent's*
tracer, and the parent's JSONL would interleave both runs. Passing the
child's callbacks explicitly isn't enough: LangChain *merges* them with
the inherited ones. So the child runs inside a brand-new, empty
``contextvars.Context`` — nothing is inherited, and the two traces stay
separate. ``tests/test_subagents.py`` checks the parent's record count.
"""

from __future__ import annotations

import contextvars
import dataclasses
from typing import Callable, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from coding_agent.config import Settings
from coding_agent.state import AgentState
from coding_agent.tools import PermissionGate, _clip, always_allow


SUBAGENT_MAX_DEPTH = 1
"""The parent is depth 0; its children are depth 1 and may not spawn."""

READ_ONLY_TOOL_NAMES: tuple[str, ...] = ("list_dir", "read_file")
"""The full set of tools a subagent may use. An allowlist on purpose:
anything added to the parent later (new built-ins, MCP tools) is excluded
from children by default instead of leaking in."""

ChildRunner = Callable[[str, Settings, str], AgentState]
"""``run_child(task, child_settings, run_label) -> final child state``."""


_SUBAGENT_BRIEF = (
    "[You are a read-only subagent working for another agent. You can only "
    "use list_dir and read_file. Investigate the task below, then reply "
    "with a short, factual answer — it is all the other agent will see.]\n"
)


# ---------------------------------------------------------------------------
# The bounds, as small pure functions
# ---------------------------------------------------------------------------


def subagent_toolset(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Filter a toolset down to the read-only allowlist."""
    return [t for t in tools if t.name in READ_ONLY_TOOL_NAMES]


def subagent_settings(settings: Settings) -> Settings:
    """The child's settings: identical, except for the iteration cap.

    ``Settings`` is frozen, so ``dataclasses.replace`` is the way to derive
    a variant. Keeping everything else identical is what guarantees the
    child shares the parent's ``workspace_root`` sandbox, provider and
    output limits.
    """
    return dataclasses.replace(
        settings, max_iterations=settings.subagent_max_iterations
    )


def _child_result_text(final: AgentState, child_settings: Settings) -> str:
    """Turn a child's final state into the text the parent will see.

    Normally that's just ``final_answer``. If the child ran out of
    iterations, its "answer" is only the stop notice — so we append the
    last tool result it saw, which is usually the most useful thing it
    found. Either way the caller clips the result.
    """
    answer = final.get("final_answer") or "(subagent produced no answer)"
    if final.get("error") != "max_iterations":
        return answer

    last_results = [e for e in final.get("transcript", []) if e.kind == "tool_result"]
    note = (
        f"Subagent hit its iteration cap ({child_settings.max_iterations}) "
        "before finishing."
    )
    if not last_results:
        return note
    last = last_results[-1]
    return (
        f"{note} Last tool result ({last.metadata.get('tool', '?')}):\n"
        f"{last.content}"
    )


# ---------------------------------------------------------------------------
# The tool the parent sees
# ---------------------------------------------------------------------------


class SpawnSubagentInput(BaseModel):
    task: str = Field(
        description=(
            "A self-contained question or investigation for a read-only "
            "subagent (it can list and read files, nothing else)."
        )
    )


def build_spawn_subagent_tool(
    settings: Settings,
    *,
    run_child: ChildRunner,
    permission_gate: PermissionGate = always_allow,
) -> BaseTool:
    """Create the parent's ``spawn_subagent(task)`` tool.

    Build one per parent run: the spawn counter lives in this closure, so
    "per run" means "per tool instance". ``run_agent`` compiles a fresh
    graph (and therefore a fresh tool) every time it is called.
    """
    child_settings = subagent_settings(settings)
    limit = settings.tool_output_limit
    spawned = 0

    def spawn_subagent(task: str) -> str:
        nonlocal spawned
        # Same pipeline as every other tool: gate first...
        gate = permission_gate("spawn_subagent", {"task": task})
        if not gate.allowed:
            return f"DENIED by permission gate: {gate.reason}"
        # ...then the spawn cap. An error *string*, not an exception, so
        # the model reads it as a tool result and carries on without it.
        if spawned >= settings.max_subagents:
            return (
                f"ERROR: subagent limit reached ({settings.max_subagents} "
                "per run). Answer with what you already know, or do the "
                "work yourself."
            )
        spawned += 1
        label = f"subagent-{spawned}"
        try:
            # Fresh, empty context: see "Tracing" in the module docstring.
            final = contextvars.Context().run(
                run_child, _SUBAGENT_BRIEF + task, child_settings, label
            )
        except Exception as e:  # boundary safety: a child crash ≠ parent crash
            return f"ERROR: {label} failed: {type(e).__name__}: {e}"
        return _clip(f"[{label}] {_child_result_text(final, child_settings)}", limit)

    return StructuredTool.from_function(
        func=spawn_subagent,
        name="spawn_subagent",
        description=(
            "Delegate a self-contained, read-only investigation to a "
            "subagent and get back its short answer. The subagent can only "
            "list and read files in the workspace, has a small iteration "
            f"budget ({settings.subagent_max_iterations}), and at most "
            f"{settings.max_subagents} may be spawned per run."
        ),
        args_schema=SpawnSubagentInput,
    )
