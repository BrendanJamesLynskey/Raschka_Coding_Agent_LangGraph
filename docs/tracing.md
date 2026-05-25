# Tracing

Every LLM round-trip the agent makes is captured in full. There is no
sampling, no truncation in the trace itself (only in tool outputs going
back into the prompt), and no opaque "request id" — the actual prompt
and the actual response are right there.

## What's captured

One `TraceRecord` per LLM call:

| Field | Meaning |
|-------|---------|
| `run_id`           | LangChain-supplied UUID for this call |
| `parent_run_id`    | Parent run if this call was nested |
| `provider_class`   | e.g. `ChatGoogleGenerativeAI`, `ChatOpenAI` |
| `model_name`       | e.g. `gemini-2.5-flash`, `deepseek-chat` |
| `started_at`       | Unix timestamp |
| `duration_s`       | Wall-clock latency |
| `prompt`           | Full list of messages sent to the model (`role`, `content`, `tool_calls`, `name`) |
| `response`         | The assistant message: text content + any `tool_calls` |
| `input_tokens`     | From `llm_output.token_usage` if the provider reports it |
| `output_tokens`    | "" |
| `total_tokens`     | "" |
| `error`            | Set instead of `response` if the call raised |

The handler lives in
[`src/coding_agent/trace.py`](../src/coding_agent/trace.py). It is a
plain `BaseCallbackHandler` subclass — no special hooks, no patched
clients. That means it will trace any LangChain chat model, whether
you're using the agent or not.

## Where it goes

* **Stdout (rich panels).** Colour-coded boxes per message —
  magenta = system, blue = human, green = assistant, yellow = tool.
  Tool calls render with their JSON args. Useful while iterating.

* **JSONL (`traces/trace-<label>-<hash>.jsonl`).** One JSON record per
  line. Easy to grep, diff, and load back later for analysis.

Toggle stdout off with `CODING_AGENT_TRACE_STDOUT=0` if you want clean
output but still want the JSONL log on disk.

## Reading a trace file later

```python
import json
from coding_agent.trace import TraceRecord, render_records

# Load records back
with open("traces/trace-...jsonl") as fh:
    records = [TraceRecord(**json.loads(line)) for line in fh]

# Re-render them to the terminal
render_records(records)
```

`_MessageView` is a plain dataclass, so loading nested data needs one
extra step:

```python
from coding_agent.trace import TraceRecord, _MessageView

def _load(path):
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            d["prompt"] = [_MessageView(**m) for m in d["prompt"]]
            if d.get("response"):
                d["response"] = _MessageView(**d["response"])
            yield TraceRecord(**d)
```

## Latency, tokens, and cost

The tracer pulls `input_tokens` / `output_tokens` / `total_tokens` from
`llm_output.token_usage` when the provider reports them. OpenAI and
DeepSeek always do; Gemini reports them via its own path which
`langchain-google-genai` normalises. If a field is `None` in the trace,
the provider didn't include it.

The trace doesn't compute cost in dollars — provider prices change too
often for that to be useful in-repo. The JSONL file gives you everything
you need to compute it externally.

## How it plugs in

`run_agent` constructs a single `TracingCallbackHandler` per run and
attaches it both to the model (so every direct invocation is traced) and
to the LangGraph config (so any internal LangGraph LLM calls are too).

```python
tracer = TracingCallbackHandler(trace_dir=settings.trace_dir, ...)
compiled = build_agent_graph(settings, callbacks=[tracer, ...])
final = compiled.invoke(state, config={"callbacks": [tracer, ...]})
```

That double-attach is intentional. Attaching only to the model misses
LangGraph-managed calls (rare, but possible); attaching only to the
config can miss calls made on a re-bound model. Both is belt-and-braces.

## Adding more events

The handler currently hooks `on_chat_model_start`, `on_llm_end`, and
`on_llm_error`. Easy extensions:

* `on_tool_start` / `on_tool_end` — to time tool execution alongside
  LLM calls.
* `on_llm_new_token` — to capture streaming token-by-token (useful for
  measuring TTFT).

Both are one-method additions to `TracingCallbackHandler`. The
`TraceRecord` dataclass would need a couple of optional fields to
accommodate the new event types.
