"""Sandboxed tools: path validation, clipping, shell argv parsing."""

from __future__ import annotations

import pytest

from coding_agent.tools import (
    PathOutsideSandbox,
    _clip,
    _resolve_inside,
    build_tools,
)


def test_resolve_inside_accepts_relative(settings):
    p = _resolve_inside(settings.workspace_root, "sub/x.txt")
    assert str(p).startswith(str(settings.workspace_root.resolve()))


def test_resolve_inside_rejects_traversal(settings):
    with pytest.raises(PathOutsideSandbox):
        _resolve_inside(settings.workspace_root, "../escape.txt")


def test_resolve_inside_rejects_absolute(settings):
    with pytest.raises(PathOutsideSandbox):
        _resolve_inside(settings.workspace_root, "/etc/passwd")


def test_clip_short_text_unchanged():
    assert _clip("abc", 100) == "abc"


def test_clip_long_text_truncated():
    out = _clip("x" * 500, 100)
    assert "clipped" in out
    assert len(out) <= 100 + 80  # under limit + tail message


def test_list_dir_lists_contents(settings):
    (settings.workspace_root / "a.txt").write_text("a")
    (settings.workspace_root / "sub").mkdir()
    list_dir = next(t for t in build_tools(settings) if t.name == "list_dir")
    out = list_dir.invoke({"path": "."})
    assert "a.txt" in out
    assert "sub" in out


def test_read_write_roundtrip(settings):
    tools = {t.name: t for t in build_tools(settings)}
    tools["write_file"].invoke({"path": "x.py", "content": "print('hi')\n"})
    text = tools["read_file"].invoke({"path": "x.py"})
    assert text == "print('hi')\n"


def test_write_blocks_traversal(settings):
    tools = {t.name: t for t in build_tools(settings)}
    with pytest.raises(PathOutsideSandbox):
        tools["write_file"].invoke({"path": "../bad.txt", "content": "no"})


def test_run_shell_executes_inside_workspace(settings):
    tools = {t.name: t for t in build_tools(settings)}
    (settings.workspace_root / "marker.txt").write_text("ok")
    # `ls` works on Linux/macOS — the test environment is Linux.
    out = tools["run_shell"].invoke({"cmd": "ls", "timeout": 5})
    assert "marker.txt" in out
    assert "[exit 0]" in out


def test_run_shell_no_shell_expansion(settings):
    """`shell=False` is a real security boundary — confirm no expansion."""
    tools = {t.name: t for t in build_tools(settings)}
    # `$HOME` should be passed literally to /bin/echo, not expanded.
    out = tools["run_shell"].invoke({"cmd": "echo $HOME", "timeout": 5})
    assert "$HOME" in out
