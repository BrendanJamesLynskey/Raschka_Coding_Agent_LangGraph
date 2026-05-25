"""Smoke test using the FAKE provider — no API key, no network.

Walks the agent through:
    1. List the workspace directory.
    2. Read `hello.txt`.
    3. Write a new file `greeting.txt`.
    4. Reply with a final answer.

The point is to demonstrate the entire loop end-to-end while letting the
user inspect the rich trace + JSONL output exactly as a real run would
produce it.

Run with:
    python examples/01_smoke_test_fake.py
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from coding_agent.config import Settings
from coding_agent.graph import run_agent


def main() -> None:
    workspace = Path(__file__).parent / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "hello.txt").write_text("Hello, world!\n", encoding="utf-8")

    # Scripted tool-call sequence. Each AIMessage is one model turn.
    scripted = [
        AIMessage(
            content="I will start by listing the workspace.",
            tool_calls=[
                {"name": "list_dir", "args": {"path": "."}, "id": "c1"}
            ],
        ),
        AIMessage(
            content="Reading hello.txt to see its contents.",
            tool_calls=[
                {"name": "read_file", "args": {"path": "hello.txt"}, "id": "c2"}
            ],
        ),
        AIMessage(
            content="Writing a new greeting based on what I found.",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {
                        "path": "greeting.txt",
                        "content": "Hi from the coding agent! I read hello.txt.\n",
                    },
                    "id": "c3",
                }
            ],
        ),
        AIMessage(
            content=(
                "Done. I read hello.txt and wrote greeting.txt with a short "
                "acknowledgement."
            ),
        ),
    ]

    settings = Settings(
        provider="fake",
        workspace_root=workspace.resolve(),
        trace_dir=(Path(__file__).parent.parent / "traces").resolve(),
        max_iterations=8,
    )

    final, tracer = run_agent(
        "Read hello.txt and produce a greeting.txt summarising what you found.",
        settings,
        fake_responses=scripted,
        run_label="smoke",
    )

    print("\n=== final answer ===\n")
    print(final.get("final_answer"))
    print(f"\nTrace file: {tracer.file_path}")
    print(f"LLM calls captured: {len(tracer.records)}")


if __name__ == "__main__":
    main()
