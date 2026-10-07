"""MCP tools: loading, prefixing, gating, clipping, and a full graph run.

Uses a tiny FastMCP server in ``tests/fixtures/mcp_echo_server.py``,
launched over stdio with the current Python interpreter — no Node, no
network. Skipped cleanly unless the ``[mcp]`` extra is installed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_mcp_adapters")
pytest.importorskip("mcp")

from langchain_core.messages import AIMessage  # noqa: E402

from coding_agent.graph import run_agent  # noqa: E402
from coding_agent.mcp_tools import (  # noqa: E402
    _run_sync,
    load_mcp_config,
    load_mcp_tools,
    wrap_mcp_tool,
)
from coding_agent.tools import ToolApproval  # noqa: E402

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
CONNECTIONS = {
    "fixture": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(FIXTURE_SERVER)],
    }
}


@pytest.fixture
def mcp_settings(settings, tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps(CONNECTIONS))
    return dataclasses.replace(settings, mcp_config=config)


@pytest.fixture(scope="module")
def raw_tools():
    """The adapter's own (async-only, unprefixed) tools, listed once.

    Starting the stdio fixture server costs ~0.5 s, so the gate / clip
    tests wrap these directly with ``wrap_mcp_tool`` instead of re-listing
    through ``load_mcp_tools`` every time. (Each *call* still starts a
    fresh server session; see the module docstring of mcp_tools.py.)
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(CONNECTIONS)
    return _by_name(asyncio.run(client.get_tools(server_name="fixture")))


def _wrap(raw_tools, name, **kwargs):
    kwargs.setdefault("limit", 2000)
    return wrap_mcp_tool("fixture", raw_tools[name], **kwargs)


def _by_name(tools):
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# Loading + naming
# ---------------------------------------------------------------------------


def test_no_config_means_no_tools(settings):
    assert settings.mcp_config is None
    assert load_mcp_tools(settings) == []


def test_tools_load_with_server_prefix(mcp_settings):
    tools = _by_name(load_mcp_tools(mcp_settings))
    assert set(tools) == {"fixture__add", "fixture__shout"}
    assert tools["fixture__add"].metadata == {"mcp_server": "fixture", "mcp_tool": "add"}
    # The server's JSON Schema survives the rename (it is what the model sees).
    assert set(tools["fixture__add"].args_schema["properties"]) == {"a", "b"}


def test_adapter_tools_really_are_async_only(raw_tools):
    # The reason the bridge exists. If a future adapter release adds sync
    # support, this test tells us the bridge can go.
    with pytest.raises(NotImplementedError):
        raw_tools["add"].invoke({"a": 1, "b": 2})


def test_sync_invoke_goes_through_bridge(raw_tools):
    assert _wrap(raw_tools, "add").invoke({"a": 2, "b": 40}) == "42"


def test_async_invoke_also_works(raw_tools):
    assert asyncio.run(_wrap(raw_tools, "add").ainvoke({"a": 1, "b": 1})) == "2"


def test_bridge_works_inside_running_loop():
    async def inner() -> int:
        return 7

    async def outer() -> int:
        # A sync caller nested inside a running loop (e.g. Jupyter).
        return _run_sync(inner())

    assert asyncio.run(outer()) == 7


# ---------------------------------------------------------------------------
# Same pipeline as built-ins: gate + clip
# ---------------------------------------------------------------------------


def test_permission_gate_sees_prefixed_name_and_can_deny(raw_tools):
    seen: list[tuple[str, dict]] = []

    def gate(name, args):
        seen.append((name, args))
        if name == "fixture__shout":
            return ToolApproval(allowed=False, reason="too loud")
        return ToolApproval(allowed=True)

    add = _wrap(raw_tools, "add", permission_gate=gate)
    shout = _wrap(raw_tools, "shout", permission_gate=gate)
    assert add.invoke({"a": 1, "b": 2}) == "3"
    assert shout.invoke({"text": "hi"}) == "DENIED by permission gate: too loud"
    # Called exactly once per tool call, with the prefixed name.
    assert seen == [("fixture__add", {"a": 1, "b": 2}), ("fixture__shout", {"text": "hi"})]


def test_output_is_clipped_to_tool_output_limit(raw_tools):
    shout = _wrap(raw_tools, "shout", limit=200)
    out = shout.invoke({"text": "abc", "times": 500})  # 1500 chars
    assert out.startswith("ABCABC")
    assert "output clipped" in out
    assert len(out) <= 200


def test_tool_error_becomes_string_not_exception(raw_tools):
    out = _wrap(raw_tools, "add").invoke({"a": "not-a-number", "b": 1})
    # Newer adapters return the server's error as content; older ones raise
    # ToolException, which our wrapper turns into "ERROR running ...".
    # Either way the model gets a string, and the graph keeps running.
    assert isinstance(out, str)
    assert "Error executing tool add" in out


# ---------------------------------------------------------------------------
# Full graph run
# ---------------------------------------------------------------------------


def test_full_graph_run_calls_mcp_tool(mcp_settings):
    scripted = [
        AIMessage(
            content="adding",
            tool_calls=[{"name": "fixture__add", "args": {"a": 19, "b": 23}, "id": "m1"}],
        ),
        AIMessage(content="19 + 23 = 42"),
    ]
    final, tracer = run_agent("add 19 and 23", mcp_settings, fake_responses=scripted)
    results = [e for e in final["transcript"] if e.kind == "tool_result"]
    assert [(r.metadata["tool"], r.content) for r in results] == [("fixture__add", "42")]
    assert final["final_answer"] == "19 + 23 = 42"
    assert len(tracer.records) == 2


def test_subagent_does_not_get_mcp_tools(mcp_settings):
    child_script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "fixture__add", "args": {"a": 1, "b": 1}, "id": "c1"}],
        ),
        AIMessage(content="no mcp here"),
    ]
    final, _ = run_agent(
        "child task",
        mcp_settings,
        fake_responses=child_script,
        depth=1,
    )
    [result] = [e.content for e in final["transcript"] if e.kind == "tool_result"]
    assert result == "ERROR: unknown tool 'fixture__add'."


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, message",
    [
        ([], "expected a JSON object"),
        ({"bad__name": {"transport": "stdio", "command": "x"}}, "must not contain"),
        ({"ok": {"transport": "sse", "url": "http://x"}}, "transport='sse'"),
        ({"ok": "not-an-object"}, "must be an object"),
    ],
)
def test_bad_configs_are_rejected(tmp_path, config, message):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        load_mcp_config(path)


def test_streamable_http_entry_is_accepted(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"remote": {"transport": "streamable_http", "url": "http://h/mcp"}}))
    assert load_mcp_config(path)["remote"]["url"] == "http://h/mcp"


def test_missing_config_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_mcp_config(tmp_path / "nope.json")
