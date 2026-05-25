"""Shared pytest fixtures.

Everything here is designed so the test suite NEVER makes an outbound API
call: settings default to the `fake` provider, and the workspace is a
fresh tmp_path per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A safe, isolated Settings for a single test."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    traces = tmp_path / "traces"
    traces.mkdir()
    return Settings(
        provider="fake",
        workspace_root=workspace,
        trace_dir=traces,
        max_iterations=6,
        transcript_compress_at=6,
        tool_output_limit=2000,
        trace_stdout=False,  # tests shouldn't spam stdout
    )
