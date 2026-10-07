"""Bounded subagents (Component 6): one test per bound, plus the happy path.

All runs use the fake provider. The parent and the child get separate
scripts (``fake_responses`` / ``fake_subagent_responses``); every spawned
child replays its script from the start.
"""

from __future__ import annotations

import dataclasses

from langchain_core.messages import AIMessage

from coding_agent.graph import build_agent_graph, run_agent
from coding_agent.subagents import (
    READ_ONLY_TOOL_NAMES,
    build_spawn_subagent_tool,
    subagent_settings,
    subagent_toolset,
)
from coding_agent.tools import ToolApproval, build_tools


def _call(name: str, args: dict, call_id: str = "c1", content: str = "") -> AIMessage:
    return AIMessage(
        content=content, tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _spawn(task: str = "What does notes.txt say?", call_id: str = "p1") -> AIMessage:
    return _call("spawn_subagent", {"task": task}, call_id)


def _tool_results(final, tool: str) -> list[str]:
    return [
        e.content
        for e in final["transcript"]
        if e.kind == "tool_result" and e.metadata.get("tool") == tool
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parent_spawns_child_which_reads_file(settings):
    (settings.workspace_root / "notes.txt").write_text("The answer is 42.\n")

    child_script = [
        _call("read_file", {"path": "notes.txt"}),
        AIMessage(content="notes.txt says: The answer is 42."),
    ]
    parent_script = [
        _spawn(),
        AIMessage(content="According to the subagent, the answer is 42."),
    ]
    final, tracer = run_agent(
        "find the answer",
        settings,
        fake_responses=parent_script,
        fake_subagent_responses=child_script,
        run_label="parent",
    )

    # The child's answer came back to the parent as a tool result...
    [result] = _tool_results(final, "spawn_subagent")
    assert result == "[subagent-1] notes.txt says: The answer is 42."
    # ...and was in the prompt the parent's final answer was produced from.
    second_prompt = "\n".join(m.content for m in tracer.records[1].prompt)
    assert "The answer is 42." in second_prompt
    assert final["final_answer"] == "According to the subagent, the answer is 42."

    # The child's file read never entered the parent's transcript.
    assert not _tool_results(final, "read_file")

    # Parent and child traced separately: the parent's tracer saw only its
    # own two calls, and the child wrote its own JSONL file.
    assert len(tracer.records) == 2
    [child_trace] = list(settings.trace_dir.glob("trace-subagent-1-*.jsonl"))
    assert len(child_trace.read_text().strip().splitlines()) == 2


# ---------------------------------------------------------------------------
# Bound: read-only toolset
# ---------------------------------------------------------------------------


def test_subagent_toolset_is_read_only_allowlist(settings):
    names = {t.name for t in subagent_toolset(build_tools(settings))}
    assert names == set(READ_ONLY_TOOL_NAMES) == {"list_dir", "read_file"}


def test_child_write_is_refused(settings):
    child_script = [
        _call("write_file", {"path": "pwned.txt", "content": "x"}),
        AIMessage(content="tried to write"),
    ]
    final, _ = run_agent(
        "delegate",
        settings,
        fake_responses=[_spawn("write a file"), AIMessage(content="done")],
        fake_subagent_responses=child_script,
    )
    assert not (settings.workspace_root / "pwned.txt").exists()
    # The child's act node rejected it as a tool it doesn't have.
    child_state = _run_child_directly(settings, child_script)
    [refusal] = _tool_results(child_state, "write_file")
    assert refusal == "ERROR: unknown tool 'write_file'."
    assert final["final_answer"] == "done"


def _run_child_directly(settings, child_script):
    """Run a depth-1 graph on its own so a test can inspect its transcript."""
    final, _ = run_agent(
        "child task",
        subagent_settings(settings),
        fake_responses=child_script,
        run_label="child-direct",
        depth=1,
    )
    return final


# ---------------------------------------------------------------------------
# Bound: depth limit 1
# ---------------------------------------------------------------------------


def test_child_cannot_spawn_grandchild(settings):
    child_script = [
        _spawn("grandchild, please", call_id="g1"),
        AIMessage(content="could not delegate"),
    ]
    final, _ = run_agent(
        "delegate",
        settings,
        fake_responses=[_spawn(), AIMessage(content="ok")],
        fake_subagent_responses=child_script,
    )
    # Only subagent-1 ever ran: no trace file for a second-level child.
    assert len(list(settings.trace_dir.glob("trace-subagent-*"))) == 1

    [result] = _tool_results(final, "spawn_subagent")
    assert result == "[subagent-1] could not delegate"

    child_state = _run_child_directly(settings, child_script)
    [refusal] = _tool_results(child_state, "spawn_subagent")
    assert refusal == "ERROR: unknown tool 'spawn_subagent'."


# ---------------------------------------------------------------------------
# Bound: per-run spawn cap
# ---------------------------------------------------------------------------


def test_spawn_cap_returns_error_string(settings):
    settings = dataclasses.replace(settings, max_subagents=2, max_iterations=6)
    parent_script = [
        _spawn(call_id="p1"),
        _spawn(call_id="p2"),
        _spawn(call_id="p3"),
        AIMessage(content="finished"),
    ]
    final, _ = run_agent(
        "delegate thrice",
        settings,
        fake_responses=parent_script,
        fake_subagent_responses=[AIMessage(content="child answer")],
    )
    results = _tool_results(final, "spawn_subagent")
    assert results[:2] == ["[subagent-1] child answer", "[subagent-2] child answer"]
    assert results[2].startswith("ERROR: subagent limit reached (2 per run)")
    assert final["final_answer"] == "finished"
    assert len(list(settings.trace_dir.glob("trace-subagent-*"))) == 2


def test_spawn_cap_counts_per_tool_instance(settings):
    """Unit-level: the counter lives in the closure, one per parent run."""
    settings = dataclasses.replace(settings, max_subagents=1)
    calls: list[str] = []

    def stub_runner(task, child_settings, label):
        calls.append(label)
        return {"final_answer": "ok", "transcript": []}

    tool = build_spawn_subagent_tool(settings, run_child=stub_runner)
    assert tool.invoke({"task": "a"}) == "[subagent-1] ok"
    assert tool.invoke({"task": "b"}).startswith("ERROR: subagent limit reached")
    # A fresh tool (= a fresh parent run) starts counting again.
    fresh = build_spawn_subagent_tool(settings, run_child=stub_runner)
    assert fresh.invoke({"task": "c"}) == "[subagent-1] ok"
    assert calls == ["subagent-1", "subagent-1"]


# ---------------------------------------------------------------------------
# Bounds: own iteration cap, same sandbox, clipped result
# ---------------------------------------------------------------------------


def test_child_iteration_cap_still_returns_clipped_answer(settings):
    settings = dataclasses.replace(
        settings, subagent_max_iterations=2, tool_output_limit=300
    )
    (settings.workspace_root / "big.txt").write_text("x" * 5000)

    # The child reads forever; its own cap (2), not the parent's (6), stops it.
    child_script = [_call("read_file", {"path": "big.txt"})]
    final, _ = run_agent(
        "delegate",
        settings,
        fake_responses=[_spawn("read big.txt"), AIMessage(content="ok")],
        fake_subagent_responses=child_script,
    )
    [result] = _tool_results(final, "spawn_subagent")
    assert result.startswith("[subagent-1] Subagent hit its iteration cap (2)")
    assert "Last tool result (read_file)" in result
    assert "output clipped" in result
    assert len(result) <= settings.tool_output_limit


def test_child_settings_keep_sandbox_and_change_only_iterations(settings):
    child = subagent_settings(settings)
    assert child.workspace_root == settings.workspace_root
    assert child.max_iterations == settings.subagent_max_iterations
    assert dataclasses.replace(child, max_iterations=settings.max_iterations) == settings


def test_long_child_answer_is_clipped(settings):
    settings = dataclasses.replace(settings, tool_output_limit=200)
    tool = build_spawn_subagent_tool(
        settings,
        run_child=lambda t, s, label: {"final_answer": "y" * 1000, "transcript": []},
    )
    out = tool.invoke({"task": "anything"})
    assert len(out) <= 200
    assert "output clipped" in out


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_spawn_goes_through_permission_gate(settings):
    seen: list[str] = []

    def deny_spawn(name, args):
        seen.append(name)
        if name == "spawn_subagent":
            return ToolApproval(allowed=False, reason="no delegation today")
        return ToolApproval(allowed=True)

    tool = build_spawn_subagent_tool(
        settings,
        run_child=lambda *a: (_ for _ in ()).throw(AssertionError("must not run")),
        permission_gate=deny_spawn,
    )
    assert tool.invoke({"task": "x"}) == "DENIED by permission gate: no delegation today"
    assert seen == ["spawn_subagent"]


def test_child_crash_becomes_error_string(settings):
    def boom(task, child_settings, label):
        raise RuntimeError("provider down")

    tool = build_spawn_subagent_tool(settings, run_child=boom)
    assert tool.invoke({"task": "x"}) == "ERROR: subagent-1 failed: RuntimeError: provider down"


def test_parent_graph_binds_spawn_but_child_graph_does_not(settings):
    script = [AIMessage(content="hi")]
    parent = build_agent_graph(settings, fake_responses=script)
    child = build_agent_graph(settings, fake_responses=script, depth=1)
    assert "act" in parent.get_graph().nodes and "act" in child.get_graph().nodes
    # Same graph *shape* for both — the difference is only the toolset.
    assert parent.get_graph().draw_mermaid() == child.get_graph().draw_mermaid()
