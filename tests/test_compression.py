"""Transcript compression — threshold, idempotency, summary content."""

from __future__ import annotations

from coding_agent.compression import KEEP_RECENT, compress_transcript
from coding_agent.state import TranscriptEntry


def _make(n: int) -> list[TranscriptEntry]:
    return [
        TranscriptEntry(
            kind="tool_call",
            content="",
            metadata={"tool": "read_file", "args": {"path": f"f{i}.py"}},
        )
        for i in range(n)
    ]


def test_below_threshold_is_passthrough():
    t = _make(5)
    out = compress_transcript(t, threshold=10)
    assert out is t  # exact identity — no allocation


def test_above_threshold_compresses_head_keeps_tail():
    t = _make(20)
    out = compress_transcript(t, threshold=10)
    assert out[0].kind == "summary"
    assert len(out) == 1 + KEEP_RECENT
    # The tail should be the *last* KEEP_RECENT entries verbatim.
    assert out[-1].metadata["args"]["path"] == "f19.py"


def test_summary_mentions_files_and_tool():
    t = _make(20)
    out = compress_transcript(t, threshold=10)
    summary = out[0].content
    assert "tool calls" in summary
    assert "read_file" in summary
    assert "f0.py" in summary  # an early file should show up in the digest


def test_idempotent_on_second_pass():
    t = compress_transcript(_make(20), threshold=10)
    again = compress_transcript(t, threshold=10)
    assert again is t  # already small enough
