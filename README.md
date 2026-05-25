# Raschka_Coding_Agent_LangGraph

A LangGraph implementation of the coding-agent architecture described in
Sebastian Raschka's article
[**Components of a Coding Agent**](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent).

The goal is **learnability**, not production polish: every component from the
article maps onto a single, heavily-commented module so you can read the code
top-to-bottom and watch the ideas turn into Python.

* Supports **Google Gemini**, **DeepSeek**, **OpenAI**, and a built-in
  scripted **fake** provider for tests.
* Every LLM call is captured in full — pretty stdout panels via `rich` plus
  one-record-per-line JSONL on disk.
* Sandboxed file + shell tools rooted at a workspace directory; paths that
  try to escape are rejected before the tool runs.
* 29 offline tests, **zero** of which hit a real provider.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

cp .env.example .env       # then edit .env with your provider + key

# Smoke test (no API key needed — uses the scripted fake provider):
python examples/01_smoke_test_fake.py

# Real run against whichever provider you set in .env:
python examples/02_real_run.py "list the files and tell me what this is"

# Or as an installed command:
coding-agent "create a hello.py that prints 'hi'"
```

The agent operates inside `examples/workspace/` by default — drop your
project files in there (or point `CODING_AGENT_WORKSPACE` at another
directory) before running it on a real task.

---

## What's in the box

### Architecture

The article identifies six components; this implementation maps one module
per component so the mapping is obvious:

| Article component                        | Module                              |
|------------------------------------------|-------------------------------------|
| 1 · Live Repo Context                    | `src/coding_agent/context.py`       |
| 2 · Prompt Shape and Cache Reuse         | `src/coding_agent/prompts.py`       |
| 3 · Tool Access and Use                  | `src/coding_agent/tools.py`         |
| 4 · Context Reduction                    | `src/coding_agent/compression.py`   |
| 5 · Structured Session Memory            | `src/coding_agent/state.py`         |
| 6 · Delegation with Bounded Subagents    | *(documented as a planned extension)* |

The state machine that ties them together lives in
[`src/coding_agent/graph.py`](src/coding_agent/graph.py). It is short
on purpose — the loop is the whole point:

```
START → observe → choose → (tool calls? act → compress → choose
                            | done?       → finalize → END)
```

See [`docs/architecture.md`](docs/architecture.md) for the full walkthrough.

### Providers

Provider selection is `.env`-driven. Set `CODING_AGENT_PROVIDER` to one of
`gemini`, `deepseek`, `openai`, or `fake`, and fill in the corresponding
keys. The agent code never imports a concrete provider — everything goes
through the factory in [`src/coding_agent/llm.py`](src/coding_agent/llm.py).
See [`docs/providers.md`](docs/providers.md).

### Tracing

Every LLM call produces a `TraceRecord` containing the full prompt, the
assistant response, any tool calls, latency, and token counts. Records are
streamed both as **rich panels to stdout** and as **JSONL to disk** under
`traces/`. See [`docs/tracing.md`](docs/tracing.md).

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/architecture.md`](docs/architecture.md) | The six components, the graph, the state, the loop |
| [`docs/providers.md`](docs/providers.md) | How provider selection works; how to add a new one |
| [`docs/tracing.md`](docs/tracing.md) | TraceRecord schema; reading a JSONL trace; rich rendering |
| [`docs/learning_notes.md`](docs/learning_notes.md) | Annotated tour for newcomers to LangGraph or coding agents |

---

## Tests

```bash
.venv/bin/python -m pytest
```

29 tests, all offline, ~0.3s runtime. The graph tests use the scripted
fake provider; the trace tests use the real LangChain callback lifecycle
through that fake provider, so the integration story is covered without
spending a token.

---

## License

Educational use. Code provided as-is.

## References

* Sebastian Raschka, *Components of a Coding Agent* —
  [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent)
* LangGraph — [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
* LangChain callbacks — [python.langchain.com](https://python.langchain.com/docs/concepts/callbacks/)
