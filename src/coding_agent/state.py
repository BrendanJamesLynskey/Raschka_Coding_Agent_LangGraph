"""The single source of truth for the agent's running state.

In LangGraph, every node is a pure function ``State -> partial State``. So
the shape of the state object is the most important design decision in the
whole system: it determines what each node can see, what it can affect, and
how Raschka's "two-tier memory" maps onto a concrete data structure.

The state below is deliberately verbose — it names every piece of context
from the article so the graph code stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Sub-records that live inside the state
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceFacts:
    """Stable facts collected once at session start.

    Article calls this the "live repo context" — things like the root path,
    a tree summary, language hints, presence of a README, etc. The point
    is that these are *cheap to look up* and *unlikely to change* during a
    short coding session, so they belong in the cacheable stable prefix.
    """

    root: str
    tree_summary: str
    files_seen: int
    notes: list[str] = field(default_factory=list)


@dataclass
class WorkingMemory:
    """Distilled state for task continuity.

    Compare with ``transcript`` (below): the transcript is the long, raw
    record of everything that happened. Working memory is the short,
    living summary of *what matters right now* — current plan, known
    constraints, open questions, files modified.

    Keeping these separate means we can compress the transcript
    aggressively without losing the thread of the task.
    """

    plan: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    scratchpad: str = ""


@dataclass
class TranscriptEntry:
    """One step in the conversation history.

    We don't store raw LangChain messages here because the transcript needs
    to outlive the run (and roundtrip through JSON cleanly). Instead we
    capture only the four kinds of events the article describes.
    """

    kind: str          # "user" | "assistant_thought" | "tool_call" | "tool_result" | "summary"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pending tool calls (queued by `choose`, drained by `act`)
# ---------------------------------------------------------------------------


@dataclass
class PendingToolCall:
    id: str
    name: str
    args: dict[str, Any]


# ---------------------------------------------------------------------------
# The state itself
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """The shared dict every LangGraph node reads and writes.

    No reducers are used here — every node that wants to extend the
    transcript reads ``state["transcript"]`` and returns the full new list.
    That is slightly more verbose at call sites, but it means the
    ``compress`` node can replace the transcript wholesale without
    fighting an ``operator.add`` reducer. For a learning repo, "explicit
    list every time" is friendlier than two ways of updating the same
    field.
    """

    # --- Request ----------------------------------------------------------
    user_request: str

    # --- Context (Raschka components 1 & 5) ------------------------------
    workspace: WorkspaceFacts
    memory: WorkingMemory
    transcript: list[TranscriptEntry]

    # --- Per-turn scratch space ------------------------------------------
    # Tool calls the model just emitted; the `act` node drains them.
    pending_tool_calls: list[PendingToolCall]

    # --- Bookkeeping ------------------------------------------------------
    iterations: int
    max_iterations: int

    # --- Outputs ----------------------------------------------------------
    final_answer: str | None
    error: str | None


def initial_state(user_request: str, *, max_iterations: int) -> AgentState:
    """Construct a fresh state dict for a new run."""
    return AgentState(
        user_request=user_request,
        workspace=WorkspaceFacts(root="", tree_summary="", files_seen=0),
        memory=WorkingMemory(),
        transcript=[],
        pending_tool_calls=[],
        iterations=0,
        max_iterations=max_iterations,
        final_answer=None,
        error=None,
    )
