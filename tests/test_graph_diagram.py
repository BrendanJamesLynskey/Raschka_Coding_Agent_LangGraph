"""The mermaid diagram in the docs must match the compiled graph.

If this fails you changed the graph's nodes or edges: run
``python scripts/render_graph_diagram.py`` and commit the result.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "render_graph_diagram", REPO / "scripts" / "render_graph_diagram.py"
)
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)

_MERMAID = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)


def _normalise(mermaid: str) -> list[str]:
    # Compare as a sorted list of lines: any added/removed node or edge is
    # caught, but the *order* langgraph happens to emit edges in (which
    # differs between langgraph releases) is not.
    return sorted(line.strip() for line in mermaid.splitlines() if line.strip())


@pytest.mark.parametrize("path", render.DOCS_WITH_DIAGRAM, ids=lambda p: p.name)
def test_committed_diagram_matches_compiled_graph(path):
    block = render.BLOCK.search(path.read_text(encoding="utf-8"))
    assert block, f"{path.name} lost its graph-diagram markers"
    committed = _MERMAID.search(block.group(0))
    assert committed, f"{path.name}: no ```mermaid block between the markers"
    assert _normalise(committed.group(1)) == _normalise(render.current_mermaid()), (
        f"{path.name}: the mermaid diagram is out of date with graph.py. "
        "Run: python scripts/render_graph_diagram.py"
    )
