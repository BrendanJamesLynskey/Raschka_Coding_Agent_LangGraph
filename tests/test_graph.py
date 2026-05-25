"""End-to-end agent runs using the fake provider.

These tests exercise the *whole graph*: observe -> choose -> act -> compress
-> finalize, including transcript writes, tracing, and routing.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from coding_agent.graph import run_agent


def test_one_shot_no_tools(settings):
    """Model answers immediately, with no tool calls."""
    scripted = [AIMessage(content="Done — no tools needed.")]
    final, tracer = run_agent("say hi", settings, fake_responses=scripted)
    assert final["final_answer"] == "Done — no tools needed."
    assert len(tracer.records) == 1
    # Transcript: user request + assistant_thought.
    kinds = [e.kind for e in final["transcript"]]
    assert kinds == ["user", "assistant_thought"]


def test_full_loop_with_write_then_finish(settings):
    """Model writes a file then signals completion."""
    scripted = [
        AIMessage(
            content="writing",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "out.txt", "content": "hi"},
                    "id": "c1",
                }
            ],
        ),
        AIMessage(content="Wrote out.txt with 'hi'. Done."),
    ]
    final, tracer = run_agent(
        "create out.txt with 'hi'", settings, fake_responses=scripted
    )
    assert (settings.workspace_root / "out.txt").read_text() == "hi"
    assert final["final_answer"].startswith("Wrote out.txt")
    # Working memory should reflect the file we wrote.
    assert "out.txt" in final["memory"].files_touched
    # Two LLM calls.
    assert len(tracer.records) == 2
    # Transcript should include tool_call + tool_result pair.
    kinds = [e.kind for e in final["transcript"]]
    assert "tool_call" in kinds
    assert "tool_result" in kinds


def test_max_iterations_safety_net(settings):
    """If the model never stops asking for tools, the agent bails out."""
    # Cycle the same useless tool call forever; FakeMessagesListChatModel
    # cycles through responses, so this fires every turn.
    looping = AIMessage(
        content="looping",
        tool_calls=[{"name": "list_dir", "args": {"path": "."}, "id": "x"}],
    )
    final, _ = run_agent(
        "loop forever please", settings, fake_responses=[looping]
    )
    assert "max_iterations" in (final.get("final_answer") or "")


def test_compression_kicks_in(settings):
    """When the transcript exceeds the threshold, the head is summarised."""
    # settings has transcript_compress_at=6; max_iterations=6.
    # Each loop adds: assistant_thought + tool_call + tool_result = 3 entries
    # (plus the user entry on turn 1). After 3 loops we have:
    #   1 (user) + 3*3 = 10 entries -> > threshold -> compress.
    list_tool = AIMessage(
        content="listing",
        tool_calls=[{"name": "list_dir", "args": {"path": "."}, "id": "x"}],
    )
    final_msg = AIMessage(content="ok done")
    scripted = [list_tool, list_tool, list_tool, final_msg]
    final, _ = run_agent("explore", settings, fake_responses=scripted)
    kinds = [e.kind for e in final["transcript"]]
    assert "summary" in kinds, f"expected a summary entry; got {kinds}"
