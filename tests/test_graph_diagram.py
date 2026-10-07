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
    """Reduce mermaid source to the graph itself, as a sorted line list.

    Two things vary between library releases without the graph changing:
    the config header (YAML front matter vs. a ``%%{init}%%`` line) and
    the order edges are emitted in. So we keep only what follows the
    ``graph TD;`` line, and sort it. Any added or removed node or edge
    still changes the result.
    """
    lines = [line.strip() for line in mermaid.splitlines() if line.strip()]
    body = lines[next(i for i, line in enumerate(lines) if line.startswith("graph ")):]
    return sorted(body)


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
