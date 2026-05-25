"""Run the agent against a real provider (Gemini / DeepSeek / OpenAI).

Reads provider + key + model from `.env`. Make sure you've copied
`.env.example` to `.env` and filled in the right block.

Run with:
    python examples/02_real_run.py "Write a hello world in main.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

from coding_agent.config import load_settings
from coding_agent.graph import run_agent


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python examples/02_real_run.py 'task description'")
        return 1
    request = " ".join(argv[1:])

    settings = load_settings()
    # Steer the workspace to the example dir if the user didn't override it.
    if str(settings.workspace_root).endswith("examples/workspace"):
        ws = (Path(__file__).parent / "workspace").resolve()
        ws.mkdir(parents=True, exist_ok=True)
        # We don't mutate the frozen Settings — load_settings already
        # resolved the path. Just ensure it exists.

    print(f"Provider: {settings.provider}")
    print(f"Workspace: {settings.workspace_root}")
    print(f"Trace dir: {settings.trace_dir}")
    print()

    final, tracer = run_agent(request, settings)

    print("\n=== final answer ===\n")
    print(final.get("final_answer") or "(no answer produced)")
    print(f"\nTrace file: {tracer.file_path}")
    print(f"LLM calls captured: {len(tracer.records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
