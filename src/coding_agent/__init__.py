"""Raschka_Coding_Agent_LangGraph
================================

A LangGraph implementation of the coding-agent architecture described in
Sebastian Raschka's article *Components of a Coding Agent*
(https://magazine.sebastianraschka.com/p/components-of-a-coding-agent).

The package is organised one component per module so the mapping from the
article to the code is obvious. See `docs/architecture.md` for the full
walkthrough.

Public surface kept deliberately small — most learning happens by reading
the modules directly.
"""

from coding_agent.config import Settings, load_settings
from coding_agent.graph import build_agent_graph, run_agent

__all__ = [
    "Settings",
    "load_settings",
    "build_agent_graph",
    "run_agent",
]
