# MCP tools

The agent can use tools from any [Model Context Protocol](https://modelcontextprotocol.io)
server, alongside its four built-in tools. The loader lives in
[`src/coding_agent/mcp_tools.py`](../src/coding_agent/mcp_tools.py) and is
built on [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters).

```bash
pip install -e '.[mcp]'
export CODING_AGENT_MCP_CONFIG=examples/mcp_config.example.json
coding-agent "Use the calculate tool to compute 1234 * 5678"
```

## Configuration

`CODING_AGENT_MCP_CONFIG` names a JSON file. Its shape is exactly the
`connections` argument of `MultiServerMCPClient`: one entry per server,
keyed by server name. Two transports are supported:

```json
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
```

* Leave the variable unset and there is no MCP, so the agent behaves
  exactly as before.
* Relative paths in `args` resolve against the directory you run the
  agent from. Stdio servers inherit the agent's working directory.
* The loader checks only what it relies on: a JSON object of objects,
  server names that are safe as prefixes (letters, digits, `-` and `_`,
  no `__`), and `transport` set to `stdio` or `streamable_http`. Every
  other field goes to the adapter unchanged.

## What the agent does with them

```
MCP tool call → permission gate → execute (over MCP) → clip → transcript
```

* **Prefixed names.** Each tool is renamed `<server>__<tool>`, e.g.
  `tools__calculate`. Built-in names never contain `__`, so a server
  can't shadow `read_file`. `build_agent_graph` also refuses to start
  if two tools share a name.
* **Same permission gate.** The gate sees the prefixed name and the
  arguments, and runs *before* the server is contacted.
* **Same clipping.** Results are clipped to `CODING_AGENT_TOOL_OUTPUT_LIMIT`.
  Non-text content blocks (images, resources) become placeholders.
* **Errors are strings.** A server-side error or a crash comes back as
  text for the model to read, never as an exception inside the graph.
* **Parent only.** Subagents are read-only, so MCP tools are never given
  to them, and their servers are not even started for a child.

## The sync/async bridge

Adapter tools are async-only. With `langchain-mcp-adapters` 0.3.2,
`tool.invoke(...)` raises
`NotImplementedError: StructuredTool does not support sync invocation`,
and `tests/test_mcp_tools.py` pins that behaviour. The graph's `act` node
is synchronous.

Rather than making `act` async, which would ripple into `run_agent` and
into `spawn_subagent` (it calls `run_agent` from inside `act`), each MCP
tool gets a small sync wrapper. The wrapper runs the coroutine with
`asyncio.run`. If an event loop is already running in that thread, as in
Jupyter, it uses a one-off worker thread instead. The wrapper also keeps
the native `coroutine`, so `ainvoke` works too. The full reasoning is in
the module docstring.

Trade-off: no MCP session is held open across a run, so the adapter
opens a new session for each call. For stdio, that means starting the
server process each time, which is about 0.5 s for the Python test
fixture.

## Live interop check: MCP_Gateway_Playground's tools-server

This is what was actually run in the build sandbox (Node 22.22.0,
`langchain-mcp-adapters` 0.3.2, `mcp` 1.30.0). It needs no API key.
[`BrendanJamesLynskey/MCP_Gateway_Playground`](https://github.com/BrendanJamesLynskey/MCP_Gateway_Playground)
was checked out as a sibling of this repo.

```bash
# 1. Build the TypeScript stdio server.
cd ../MCP_Gateway_Playground/servers/tools-server
npm install && npm run build            # -> dist/index.js

# 2. From this repo's root, list and call its tools through our loader.
cd -
CODING_AGENT_MCP_CONFIG=examples/mcp_config.example.json python - <<'PY'
from coding_agent.config import load_settings
from coding_agent.mcp_tools import load_mcp_tools
tools = {t.name: t for t in load_mcp_tools(load_settings())}
for name, t in tools.items():
    print(f"{name:24s} {list(t.args_schema.get('properties', {}))}")
print(tools["tools__calculate"].invoke({"expression": "2 * (3 + 4) ^ 2"}))
PY
```

Output:

```
tools__echo              ['message']
tools__calculate         ['expression']
tools__current_time      []
tools__generate_uuid     []
98
```

Then a full agent run, with a local model (Ollama, `qwen2.5:1.5b` and
`qwen3.5:0.8b`) choosing the MCP tool by itself:

```bash
CODING_AGENT_MCP_CONFIG=examples/mcp_config.example.json \
CODING_AGENT_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:1.5b \
coding-agent --quiet "Use the calculate tool to compute 1234 * 5678 and tell me the result."
```

Both models made a native `tools__calculate` call with
`{"expression": "1234 * 5678"}` and then answered `7006652`, which is
correct. The JSONL trace for each run shows two LLM calls: the tool call,
then the answer.

## Tests

`tests/test_mcp_tools.py` starts the tiny FastMCP server in
[`tests/fixtures/mcp_echo_server.py`](../tests/fixtures/mcp_echo_server.py)
over stdio with the current Python interpreter, so no Node and no
network are needed. It checks loading, prefixes, the gate (including a
deny), clipping, error strings, the bridge (sync, async, and inside a
running loop), a full fake-provider graph run, and that subagents don't
get MCP tools. The module is skipped cleanly if the `[mcp]` extra isn't
installed.
