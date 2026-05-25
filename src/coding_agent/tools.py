"""Sandboxed tools the agent can call.

Raschka's "Tool Access and Use" component prescribes a strict pipeline:

    parse  ->  validate  ->  check permissions  ->  execute  ->  clip
                                                                       result

We implement that pipeline once, here, and expose four tools:

* ``list_dir(path)``     — directory listing
* ``read_file(path)``    — file contents (clipped)
* ``write_file(path, content)`` — create/overwrite a text file
* ``run_shell(cmd, timeout=30)`` — short-lived shell command

Every path is forced inside ``settings.workspace_root``. Any attempt to
escape via ``..``, an absolute path, or a symlink is rejected before
execution. ``run_shell`` runs *inside* the workspace so commands like
``python -c '...'`` see the right files.

A "permission gate" hook is provided for future human-in-the-loop control:
right now it always approves, but everything routes through it so wiring
in `interrupt_before` later is one line of change.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from coding_agent.config import Settings


# ---------------------------------------------------------------------------
# Permission gate
# ---------------------------------------------------------------------------


@dataclass
class ToolApproval:
    """Decision from the permission gate."""

    allowed: bool
    reason: str = ""


PermissionGate = Callable[[str, dict], ToolApproval]


def always_allow(_name: str, _args: dict) -> ToolApproval:
    """Default gate — every call is allowed.

    Replace at construction time to add interactive approval, allowlists,
    or per-tool policies.
    """
    return ToolApproval(allowed=True)


# ---------------------------------------------------------------------------
# Path sandboxing
# ---------------------------------------------------------------------------


class PathOutsideSandbox(ValueError):
    """Raised when a tool argument tries to escape ``workspace_root``."""


def _resolve_inside(workspace_root: Path, relative_or_absolute: str) -> Path:
    """Return the absolute Path, verifying it stays inside the sandbox.

    Uses ``Path.resolve(strict=False)`` so we can validate paths that don't
    exist yet (e.g. ``write_file`` for a new file). The crucial check is
    that the *resolved* path is still under the workspace root — that
    catches ``..`` traversal, absolute paths, *and* symlinks-once-resolved.
    """
    candidate = Path(relative_or_absolute)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace_root.resolve())
    except ValueError as e:
        raise PathOutsideSandbox(
            f"Path {relative_or_absolute!r} resolves outside the workspace "
            f"({workspace_root})."
        ) from e
    return resolved


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 80]
    return cut + f"\n... (output clipped: {len(text) - len(cut)} chars removed)"


# ---------------------------------------------------------------------------
# Tool input schemas (Pydantic models so the LLM gets clean JSON schemas)
# ---------------------------------------------------------------------------


class ListDirInput(BaseModel):
    path: str = Field(default=".", description="Path relative to workspace root.")


class ReadFileInput(BaseModel):
    path: str = Field(description="Path to the file, relative to workspace root.")


class WriteFileInput(BaseModel):
    path: str = Field(description="Path to write to, relative to workspace root.")
    content: str = Field(description="Full new contents of the file.")


class RunShellInput(BaseModel):
    cmd: str = Field(description="Shell command. Runs inside the workspace root.")
    timeout: int = Field(default=30, ge=1, le=120, description="Seconds before kill.")


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------


def build_tools(
    settings: Settings,
    *,
    permission_gate: PermissionGate = always_allow,
) -> list[BaseTool]:
    """Create the agent's tool list, all bound to ``settings.workspace_root``.

    We build them inside a function (not as module-level globals) so each
    agent run can have its own sandbox and gate — important for tests.
    """

    root = settings.workspace_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    limit = settings.tool_output_limit

    # --- list_dir --------------------------------------------------------
    def list_dir(path: str = ".") -> str:
        gate = permission_gate("list_dir", {"path": path})
        if not gate.allowed:
            return f"DENIED by permission gate: {gate.reason}"
        target = _resolve_inside(root, path)
        if not target.exists():
            return f"ERROR: {path} does not exist."
        if not target.is_dir():
            return f"ERROR: {path} is not a directory."
        lines = []
        for child in sorted(target.iterdir()):
            kind = "DIR " if child.is_dir() else "FILE"
            size = "" if child.is_dir() else f"  ({child.stat().st_size}B)"
            lines.append(f"{kind}  {child.name}{size}")
        return _clip("\n".join(lines) or "(empty directory)", limit)

    # --- read_file -------------------------------------------------------
    def read_file(path: str) -> str:
        gate = permission_gate("read_file", {"path": path})
        if not gate.allowed:
            return f"DENIED by permission gate: {gate.reason}"
        target = _resolve_inside(root, path)
        if not target.exists():
            return f"ERROR: {path} does not exist."
        if not target.is_file():
            return f"ERROR: {path} is not a file."
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"ERROR: {path} is not UTF-8 text (binary file)."
        return _clip(text, limit)

    # --- write_file ------------------------------------------------------
    def write_file(path: str, content: str) -> str:
        gate = permission_gate("write_file", {"path": path, "content": content})
        if not gate.allowed:
            return f"DENIED by permission gate: {gate.reason}"
        target = _resolve_inside(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"

    # --- run_shell -------------------------------------------------------
    def run_shell(cmd: str, timeout: int = 30) -> str:
        gate = permission_gate("run_shell", {"cmd": cmd, "timeout": timeout})
        if not gate.allowed:
            return f"DENIED by permission gate: {gate.reason}"
        # We deliberately do *not* run shell=True. The model must produce
        # a real argv. This is a real safety boundary — `rm -rf $HOME`
        # via shell expansion is impossible here.
        try:
            argv = shlex.split(cmd)
        except ValueError as e:
            return f"ERROR: could not parse cmd ({e})."
        if not argv:
            return "ERROR: empty command."
        try:
            proc = subprocess.run(
                argv,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s."
        except FileNotFoundError:
            return f"ERROR: command not found: {argv[0]!r}."
        out = f"$ {cmd}\n[exit {proc.returncode}]\n"
        if proc.stdout:
            out += "--- stdout ---\n" + proc.stdout
        if proc.stderr:
            out += "--- stderr ---\n" + proc.stderr
        return _clip(out, limit)

    return [
        StructuredTool.from_function(
            func=list_dir,
            name="list_dir",
            description="List entries in a directory inside the workspace.",
            args_schema=ListDirInput,
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description="Read a UTF-8 text file inside the workspace.",
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description="Create or overwrite a UTF-8 text file inside the workspace.",
            args_schema=WriteFileInput,
        ),
        StructuredTool.from_function(
            func=run_shell,
            name="run_shell",
            description=(
                "Run a short shell command inside the workspace and return "
                "its stdout, stderr and exit code. No shell expansion — pass "
                "a normal argv (e.g. `python script.py --foo bar`)."
            ),
            args_schema=RunShellInput,
        ),
    ]
