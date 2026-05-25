"""Prompt construction — Raschka's "Prompt Shape and Cache Reuse".

The article's most operational insight is: structure each prompt as a
**stable prefix** followed by a **dynamic suffix**. The prefix never
changes during the session, so providers that support prompt caching
(OpenAI, Anthropic, Gemini) can re-use it for free across every turn.

Stable prefix
-------------
* The system policy (role, rules, how to behave).
* Tool catalogue (names + descriptions — already injected by
  ``model.bind_tools``, so we don't repeat it in the system message).
* Workspace facts collected once at session start.

Dynamic suffix
--------------
* The compressed transcript of what's happened so far.
* The current working-memory snapshot.
* The user request and any tool results pending the model's attention.
"""

from __future__ import annotations

from textwrap import dedent

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from coding_agent.state import AgentState, TranscriptEntry, WorkingMemory, WorkspaceFacts


# ---------------------------------------------------------------------------
# Stable prefix
# ---------------------------------------------------------------------------


_SYSTEM_POLICY = dedent(
    """\
    You are a careful coding assistant operating inside a sandboxed workspace.

    Rules:
      1. Prefer reading before writing — inspect the relevant files first.
      2. Use the smallest tool that does the job. Never run a shell command
         when a single read_file or write_file would suffice.
      3. All file paths are RELATIVE to the workspace root shown below.
         Never use absolute paths or `..`.
      4. When you are confident the task is complete, reply with a plain
         text final answer and DO NOT request more tools.
      5. Keep your replies short. The user wants the change made, not
         a lecture.
    """
)


def build_system_message(workspace: WorkspaceFacts) -> SystemMessage:
    """The system message — this is the stable, cacheable prefix."""
    body = dedent(
        f"""\
        {_SYSTEM_POLICY}
        ---
        Workspace summary (stable for this session):

        root: {workspace.root}
        files seen at startup: {workspace.files_seen}
        notes: {", ".join(workspace.notes) or "(none)"}

        Tree:
        {workspace.tree_summary}
        """
    )
    return SystemMessage(content=body)


# ---------------------------------------------------------------------------
# Dynamic suffix — transcript replay
# ---------------------------------------------------------------------------


def _memory_snapshot(memory: WorkingMemory) -> str:
    lines = ["Working memory:"]
    lines.append(f"  plan: {memory.plan or '(empty)'}")
    lines.append(f"  open_questions: {memory.open_questions or '(none)'}")
    lines.append(f"  files_touched: {memory.files_touched or '(none)'}")
    if memory.scratchpad:
        lines.append(f"  scratchpad: {memory.scratchpad}")
    return "\n".join(lines)


def _transcript_to_messages(transcript: list[TranscriptEntry]) -> list[BaseMessage]:
    """Replay the transcript as a sequence of LangChain messages.

    We use lightweight assistant/human turns rather than re-instantiating
    every original AIMessage/tool_call pair, because:

      * Tool calls have already been *executed* — the model just needs to
        see the call+result pair as context for what's happened.
      * This keeps the prompt assembly logic dead simple and provider-
        agnostic. Real tool-use schemas vary per provider; a plain textual
        replay always works.
    """
    msgs: list[BaseMessage] = []
    for entry in transcript:
        if entry.kind == "user":
            msgs.append(HumanMessage(content=entry.content))
        elif entry.kind == "assistant_thought":
            msgs.append(AIMessage(content=entry.content))
        elif entry.kind == "tool_call":
            tool = entry.metadata.get("tool", "?")
            args = entry.metadata.get("args", {})
            msgs.append(
                AIMessage(content=f"[tool_call] {tool}({args})")
            )
        elif entry.kind == "tool_result":
            tool = entry.metadata.get("tool", "?")
            msgs.append(
                HumanMessage(content=f"[tool_result from {tool}]\n{entry.content}")
            )
        elif entry.kind == "summary":
            # A compressed-transcript block, produced by compression.py.
            msgs.append(
                HumanMessage(content=f"[summary of earlier turns]\n{entry.content}")
            )
    return msgs


def build_prompt(state: AgentState) -> list[BaseMessage]:
    """Assemble the full message list for the next LLM call.

    Order matters:
        [ system (stable prefix) ]
        [ replayed transcript    ]   <- dynamic
        [ working-memory note    ]   <- dynamic
        [ current user request   ]   <- dynamic
    """
    workspace: WorkspaceFacts = state["workspace"]
    memory: WorkingMemory = state["memory"]
    transcript: list[TranscriptEntry] = state["transcript"]

    msgs: list[BaseMessage] = [build_system_message(workspace)]
    msgs.extend(_transcript_to_messages(transcript))
    msgs.append(HumanMessage(content=_memory_snapshot(memory)))

    # Always re-state the user request at the end so the model can't lose
    # the goal when the transcript gets long.
    msgs.append(HumanMessage(content=f"User request: {state['user_request']}"))
    return msgs
