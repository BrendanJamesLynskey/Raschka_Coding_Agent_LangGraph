"""Transcript compression — Raschka's "Context Reduction".

When the transcript grows past a threshold, we replace older entries with
a single ``summary`` entry. We keep the most recent N entries verbatim
(they are most relevant), and squash the older tail into a short text
summary that names the files touched and the high-level steps taken.

Why not call the LLM to summarise?
    A model-based summariser is the realistic production approach. Here
    we use a deterministic rule-based summariser so:

      * Tests don't need a real LLM.
      * Learners can read the summariser and immediately see what
        information survives compression.

    Swapping in an LLM summariser is one function call; the rest of the
    graph doesn't change.
"""

from __future__ import annotations

from collections import Counter

from coding_agent.state import TranscriptEntry


KEEP_RECENT = 8
"""How many of the most-recent entries to retain verbatim."""


def compress_transcript(
    transcript: list[TranscriptEntry],
    *,
    threshold: int,
) -> list[TranscriptEntry]:
    """If too long, summarise the head and keep the tail verbatim.

    Idempotent: a summary entry is not itself re-summarised, because the
    next call will see ``len(transcript) <= threshold`` again.
    """
    if len(transcript) <= threshold:
        return transcript

    head = transcript[:-KEEP_RECENT]
    tail = transcript[-KEEP_RECENT:]

    summary_text = _summarise(head)
    summary_entry = TranscriptEntry(
        kind="summary",
        content=summary_text,
        metadata={"compressed_entries": len(head)},
    )
    return [summary_entry, *tail]


def _summarise(entries: list[TranscriptEntry]) -> str:
    """Deterministic, rule-based digest of an entry list.

    Captures:
      * Number of user turns, tool calls, tool results.
      * Which tools were called, and how often.
      * Files referenced (extracted from tool args/results where present).
    """
    kinds = Counter(e.kind for e in entries)
    tools_called = Counter(
        e.metadata.get("tool", "?")
        for e in entries
        if e.kind == "tool_call"
    )
    files_seen: set[str] = set()
    for e in entries:
        path = e.metadata.get("args", {}).get("path") if e.metadata else None
        if path:
            files_seen.add(path)

    lines = [
        f"Compressed {len(entries)} earlier entries.",
        f"  user turns: {kinds.get('user', 0)}",
        f"  assistant thoughts: {kinds.get('assistant_thought', 0)}",
        f"  tool calls: {kinds.get('tool_call', 0)}",
        f"  tool results: {kinds.get('tool_result', 0)}",
    ]
    if tools_called:
        breakdown = ", ".join(f"{k}={v}" for k, v in tools_called.most_common())
        lines.append(f"  tools used: {breakdown}")
    if files_seen:
        # Cap to avoid the summary itself blowing up.
        sample = sorted(files_seen)[:20]
        lines.append(f"  files referenced: {', '.join(sample)}")
    return "\n".join(lines)
