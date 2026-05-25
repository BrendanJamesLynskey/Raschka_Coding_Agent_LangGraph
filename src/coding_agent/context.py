"""Workspace context collection — Raschka's "Live Repo Context".

The agent calls ``collect_workspace_facts`` once, on its first turn, to
build a short, stable summary of the repo it's working in. That summary
goes into the prompt's *stable prefix* (see ``prompts.py``) so prompt
caching can re-use it across every turn of the session.

We deliberately keep this **summary** small: a directory tree truncated to
a few dozen entries, plus a couple of "is there a README / pyproject?"
hints. The model can always read individual files via tools.
"""

from __future__ import annotations

from pathlib import Path

from coding_agent.state import WorkspaceFacts


# Files we never list — they're noise that crowds out useful entries.
_IGNORED = {
    "__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", "*.egg-info",
}


def _is_ignored(p: Path) -> bool:
    return p.name in _IGNORED or any(p.match(g) for g in _IGNORED if "*" in g)


def collect_workspace_facts(root: Path, *, max_entries: int = 60) -> WorkspaceFacts:
    """Walk ``root`` and produce a compact, deterministic summary.

    Determinism matters: if two runs see the same workspace, they should
    see the same string, otherwise prompt caching is defeated.
    """
    root = root.resolve()
    if not root.exists():
        return WorkspaceFacts(
            root=str(root),
            tree_summary="(workspace directory does not exist yet)",
            files_seen=0,
            notes=["workspace_root missing"],
        )

    entries: list[str] = []
    files_seen = 0
    notes: list[str] = []

    # Sorted walk for determinism.
    all_paths = sorted(root.rglob("*"))
    for p in all_paths:
        if _is_ignored(p):
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if p.is_dir():
            entries.append(f"{rel}/")
        else:
            files_seen += 1
            size = p.stat().st_size
            entries.append(f"{rel}  ({size}B)")
        if len(entries) >= max_entries:
            entries.append(f"... and more (truncated at {max_entries} entries)")
            break

    tree_summary = "\n".join(entries) if entries else "(workspace is empty)"

    if (root / "README.md").exists():
        notes.append("README.md present")
    if (root / "pyproject.toml").exists():
        notes.append("pyproject.toml present (Python project)")
    if (root / "package.json").exists():
        notes.append("package.json present (Node project)")

    return WorkspaceFacts(
        root=str(root),
        tree_summary=tree_summary,
        files_seen=files_seen,
        notes=notes,
    )
