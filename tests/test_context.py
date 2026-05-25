"""Workspace context: determinism, truncation, ignore list."""

from __future__ import annotations

from coding_agent.context import collect_workspace_facts


def test_empty_workspace(tmp_path):
    facts = collect_workspace_facts(tmp_path)
    assert facts.files_seen == 0
    assert "empty" in facts.tree_summary.lower()


def test_listing_is_deterministic(tmp_path):
    for name in ["b.py", "a.py", "c.py"]:
        (tmp_path / name).write_text("x")
    one = collect_workspace_facts(tmp_path).tree_summary
    two = collect_workspace_facts(tmp_path).tree_summary
    assert one == two
    # Sorted listing means 'a.py' must appear before 'b.py'.
    assert one.index("a.py") < one.index("b.py") < one.index("c.py")


def test_truncation(tmp_path):
    for i in range(120):
        (tmp_path / f"f{i:03d}.txt").write_text("x")
    facts = collect_workspace_facts(tmp_path, max_entries=20)
    assert "truncated" in facts.tree_summary


def test_ignores_noise_dirs(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "real.py").write_text("x")
    facts = collect_workspace_facts(tmp_path)
    assert "__pycache__" not in facts.tree_summary
    assert "real.py" in facts.tree_summary


def test_notes_detect_python_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    facts = collect_workspace_facts(tmp_path)
    assert any("pyproject" in n for n in facts.notes)
