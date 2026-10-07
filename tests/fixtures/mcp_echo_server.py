"""A tiny stdio MCP server used only by the offline test suite.

Run by ``tests/test_mcp_tools.py`` as a subprocess via
``langchain-mcp-adapters``. It exposes two deliberately trivial tools so
the tests are about the *plumbing* (loading, prefixing, gating, clipping,
the sync bridge), not about what the tools compute.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# WARNING level keeps the per-request INFO lines out of the test output.
mcp = FastMCP("fixture", log_level="WARNING")


@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two integers and return the sum as text."""
    return str(a + b)


@mcp.tool()
def shout(text: str, times: int = 1) -> str:
    """Upper-case `text` and repeat it `times` times (handy for clipping tests)."""
    return text.upper() * times


if __name__ == "__main__":
    mcp.run(transport="stdio")
