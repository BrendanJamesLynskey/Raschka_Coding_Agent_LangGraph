"""Command-line entry point: ``coding-agent "your request here"``.

Tiny wrapper around ``run_agent`` — most of the heavy lifting lives in the
library code. The CLI exists so the README has a one-liner the user can
run.
"""

from __future__ import annotations

import argparse
import sys

from coding_agent.config import load_settings
from coding_agent.graph import run_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description=(
            "Run the Raschka-style coding agent on a free-form request. "
            "Provider, models and limits come from your .env file."
        ),
    )
    parser.add_argument("request", help="The task to give the agent.")
    parser.add_argument(
        "--workspace",
        help="Override CODING_AGENT_WORKSPACE for this run.",
        default=None,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence the rich trace output (JSONL is still written).",
    )
    args = parser.parse_args(argv)

    if args.workspace:
        import os

        os.environ["CODING_AGENT_WORKSPACE"] = args.workspace
    if args.quiet:
        import os

        os.environ["CODING_AGENT_TRACE_STDOUT"] = "0"

    settings = load_settings()
    final, tracer = run_agent(args.request, settings)

    print("\n=== final answer ===\n")
    print(final.get("final_answer") or "(no answer produced)")
    print(f"\nTrace file: {tracer.file_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
