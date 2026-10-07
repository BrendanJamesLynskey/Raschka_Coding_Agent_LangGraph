"""MCP tools — plugging external tool servers into Component 3.

The `Model Context Protocol <https://modelcontextprotocol.io>`_ is a
standard way for a tool server (a filesystem, a database, a calculator,
a ticket tracker…) to advertise tools that any agent can call. This module
loads tools from the MCP servers listed in a JSON file and hands them to
the graph alongside the built-in tools — **through the same pipeline**:

    MCP tool call  ->  permission gate  ->  execute (via MCP)  ->  clip

So nothing an MCP server offers can skip the ``PermissionGate`` or flood
the prompt with an unclipped result.

Configuration
-------------
Set ``CODING_AGENT_MCP_CONFIG`` to a JSON file shaped exactly like the
``connections`` argument of ``langchain_mcp_adapters``'
``MultiServerMCPClient`` — one entry per server, keyed by server name::

    {
      "tools": {
        "transport": "stdio",
        "command": "node",
        "args": ["../MCP_Gateway_Playground/servers/tools-server/dist/index.js"]
      },
      "remote": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp"
      }
    }

Unset means "no MCP": the agent behaves exactly as before. See
``examples/mcp_config.example.json`` and ``docs/mcp.md``.

Naming
------
Every MCP tool is renamed ``<server>__<tool>`` (e.g. ``tools__calculate``).
Built-in tool names never contain ``__``, so an MCP server can't shadow
``read_file``; and two servers that both offer ``search`` stay distinct.
(The adapter has its own ``tool_name_prefix`` option, but it joins with a
single underscore, which is ambiguous next to names like ``read_file``.)

The sync/async bridge
---------------------
``langchain-mcp-adapters`` tools are **async-only**: in the installed
version (0.3.x) calling ``tool.invoke(...)`` raises
``NotImplementedError: StructuredTool does not support sync invocation``.
The graph's ``act`` node calls ``tool.invoke`` synchronously.

Two fixes were possible:

1. Make ``act`` an ``async def`` node using ``ainvoke``, and run the graph
   with ``ainvoke``. That ripples outward: ``run_agent`` becomes async (or
   wraps ``asyncio.run``), and ``spawn_subagent`` — which calls
   ``run_agent`` from *inside* ``act`` — would then need an async path
   too, or it would try to start an event loop inside a running one.
2. Keep the graph synchronous and give each MCP tool a small **sync
   wrapper** that runs the async call to completion.

We use (2): it touches no existing node, and the bridge is ten lines in
one place (``_run_sync``). The wrapper also keeps an async ``coroutine``,
so an async caller using ``ainvoke`` gets the native path for free.

The cost: each sync call runs a short-lived event loop, and because we
don't hold an MCP session open across the run, the adapter opens a fresh
session per call (for stdio, that means starting the server process).
That is fine for a learning agent making a handful of calls; a long-lived
session would need its lifetime tied to the graph run — a follow-up.
"""

from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Coroutine, TypeVar

from langchain_core.tools import BaseTool, StructuredTool

from coding_agent.config import Settings
from coding_agent.tools import PermissionGate, _clip, always_allow


SEPARATOR = "__"
"""Joins server name and tool name: ``tools__calculate``."""

SUPPORTED_TRANSPORTS = frozenset({"stdio", "streamable_http"})

_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_mcp_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read and sanity-check an MCP server config file.

    We only check what *this* module relies on (the shape, the server
    names that become prefixes, the transport). Everything else in each
    entry is passed through untouched to ``MultiServerMCPClient``.
    """
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ValueError(f"CODING_AGENT_MCP_CONFIG file not found: {path}") from e
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object of {{server_name: {{...}}}}.")

    for name, entry in config.items():
        if not _SERVER_NAME.match(name) or SEPARATOR in name:
            raise ValueError(
                f"{path}: server name {name!r} must be letters, digits, '-' "
                f"or '_' and must not contain {SEPARATOR!r} (it becomes the "
                "tool-name prefix)."
            )
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry for {name!r} must be an object.")
        transport = entry.get("transport")
        if transport not in SUPPORTED_TRANSPORTS:
            raise ValueError(
                f"{path}: server {name!r} has transport={transport!r}; "
                f"expected one of {sorted(SUPPORTED_TRANSPORTS)}."
            )
    return config


# ---------------------------------------------------------------------------
# The sync bridge
# ---------------------------------------------------------------------------


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    The normal case — the graph was invoked synchronously, so there is no
    event loop in this thread — is just ``asyncio.run``. If a loop *is*
    already running here (a Jupyter cell, an async caller), ``asyncio.run``
    would refuse; we then run the coroutine on a worker thread with its
    own loop and block until it finishes.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _content_to_text(content: Any) -> str:
    """Flatten an MCP tool result into the plain text our transcript stores.

    The adapter returns either a string or a list of content blocks
    (``{"type": "text", "text": ...}``, images, embedded resources…). We
    keep the text and leave a placeholder for anything else.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(f"[{block.get('type', 'non-text')} content omitted]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Wrapping one MCP tool
# ---------------------------------------------------------------------------


def wrap_mcp_tool(
    server: str,
    tool: BaseTool,
    *,
    limit: int,
    permission_gate: PermissionGate = always_allow,
) -> BaseTool:
    """Re-expose an adapter tool under our name, gate and clipping."""
    name = f"{server}{SEPARATOR}{tool.name}"

    def _denied(args: dict[str, Any]) -> str | None:
        gate = permission_gate(name, args)
        return None if gate.allowed else f"DENIED by permission gate: {gate.reason}"

    async def _execute(args: dict[str, Any]) -> str:
        try:
            content = await tool.ainvoke(args)
        except Exception as e:  # server crashed, bad args, transport error…
            return f"ERROR running {name}: {type(e).__name__}: {e}"
        return _clip(_content_to_text(content), limit)

    async def _acall(**kwargs: Any) -> str:
        if (denial := _denied(kwargs)) is not None:
            return denial
        return await _execute(kwargs)

    def _call(**kwargs: Any) -> str:
        # Gate *before* the bridge so a denial costs no subprocess.
        if (denial := _denied(kwargs)) is not None:
            return denial
        return _run_sync(_execute(kwargs))

    return StructuredTool(
        name=name,
        description=tool.description or f"MCP tool {tool.name} from server {server}.",
        # The adapter gives us the server's JSON Schema as a dict, which
        # StructuredTool accepts as-is and bind_tools passes to the model.
        args_schema=tool.args_schema,
        func=_call,
        coroutine=_acall,
        metadata={"mcp_server": server, "mcp_tool": tool.name},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_mcp_tools(
    settings: Settings,
    *,
    permission_gate: PermissionGate = always_allow,
) -> list[BaseTool]:
    """Load, prefix, gate and clip every tool from the configured servers.

    Returns ``[]`` when ``settings.mcp_config`` is unset, so callers can
    use it unconditionally.
    """
    if settings.mcp_config is None:
        return []

    connections = load_mcp_config(settings.mcp_config)
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        raise RuntimeError(
            "CODING_AGENT_MCP_CONFIG is set but langchain-mcp-adapters is not "
            "installed. Run: pip install -e '.[mcp]'"
        ) from e

    client = MultiServerMCPClient(connections)  # type: ignore[arg-type]
    wrapped: list[BaseTool] = []
    # One server at a time so each tool knows which prefix it belongs to.
    for server in connections:
        try:
            server_tools = _run_sync(client.get_tools(server_name=server))
        except Exception as e:
            raise RuntimeError(
                f"Could not list tools from MCP server {server!r}: "
                f"{type(e).__name__}: {e}"
            ) from e
        wrapped.extend(
            wrap_mcp_tool(
                server,
                t,
                limit=settings.tool_output_limit,
                permission_gate=permission_gate,
            )
            for t in server_tools
        )
    return wrapped
