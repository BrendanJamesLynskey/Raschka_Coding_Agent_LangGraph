# Learning notes

A guided tour for two audiences:

* You're new to LangGraph but comfortable with Python.
* You're new to coding agents but have built RAG / chatbots before.

Read in order; each section ends with "go look at" pointers into the
code.

---

## 1 · What a coding agent is, really

Coding agents like Cursor, Aider, Claude Code, and Codex CLI look
magical, but the loop they run is short:

```
loop:
    look at the workspace
    look at what's happened so far
    ask the model "what should I do next?"
    if the model says "call this tool" → do it, then record the result
    if the model says "I'm done" → stop, return the answer
```

Everything else is engineering around that loop: how to keep the prompt
small enough to fit in context, how to stop tools from doing damage, how
to remember things across turns. Raschka's article names those concerns
("components") and this repo implements each one as a module.

→ Go look at: [`src/coding_agent/graph.py`](../src/coding_agent/graph.py)

---

## 2 · What LangGraph adds

LangChain's `Runnable` interface lets you compose chains:
`prompt | model | parser`. That is great for linear pipelines.
Agents aren't linear — they loop, branch, and re-enter the same step
many times. LangGraph is a tiny state-machine framework for *that*
shape.

Three things to know:

1. **State.** A `TypedDict` (or `dataclass`/`pydantic`) that every node
   reads and writes. Fields can have *reducers*: functions that combine
   the old value with what the node returned. The default is "replace".

2. **Nodes.** Plain functions: `state -> partial state`. The framework
   merges the partial back into the state using the reducers.

3. **Edges.** Either unconditional (`A → B`) or conditional (a routing
   function picks the next node based on state).

That's it. The whole framework is a Pregel-style loop that fires nodes
until you reach `END`.

→ Go look at: [`src/coding_agent/state.py`](../src/coding_agent/state.py)

---

## 3 · Why the stable / dynamic prompt split matters

LLM providers cache prompts: if the first N tokens of your next request
are byte-identical to a recent request, you get those tokens at ~0 cost
and much lower latency. The article calls this "prompt shape and cache
reuse" and it is the single biggest cost lever in a coding agent that
runs many turns.

The recipe:

* Put everything that doesn't change during the session at the start —
  system policy, tool catalogue, workspace summary. **Don't reorder
  these between turns.**
* Put everything that changes after that — transcript, working memory,
  current request.

In code we keep the policy and workspace summary inside one
`SystemMessage` built at the top of `build_prompt`. The dynamic suffix
is always assembled in the same order so cache hits stay consistent.

→ Go look at: [`src/coding_agent/prompts.py`](../src/coding_agent/prompts.py)

---

## 4 · Why "transcript" and "working memory" are different

It's tempting to keep only one big list. Two reasons not to:

* **Compression destroys information.** When you summarise old turns,
  you lose detail you might need later. Working memory is *also* a
  summary, but it tracks different stuff (current plan, open questions)
  and is updated by the agent itself, not by a summariser.

* **Transcript is for replay; working memory is for now.** If you ever
  want to resume a session, you replay the transcript. Working memory is
  for the model to remember *what it was about to do*.

Read `state.py` for the dataclasses, then `compression.py` for what
happens when the transcript grows past the threshold.

→ Go look at: [`src/coding_agent/compression.py`](../src/coding_agent/compression.py)

---

## 5 · Why the sandbox isn't optional

The shell tool runs whatever the model asks. If the model is jailbroken,
prompt-injected, or just hallucinating, `rm -rf $HOME` is one tool call
away. Two defences in this repo:

1. **Path resolution.** Every file path is `pathlib.Path.resolve()`d and
   then checked against the workspace root. `..`, absolute paths, and
   symlinks-after-resolution all fail.

2. **No shell expansion.** `run_shell` uses `shlex.split` + `subprocess`
   with `shell=False`. The string `echo $HOME` becomes
   `["echo", "$HOME"]` and prints the literal `$HOME`. There is no shell
   to expand variables, run pipes, or chain commands.

These are real boundaries enforced by the OS, not vibes. Test them
with `tests/test_tools.py`.

→ Go look at: [`src/coding_agent/tools.py`](../src/coding_agent/tools.py)

---

## 6 · Why you want a custom tracer

LangChain's built-in handlers either don't capture full prompts or
render them awkwardly. For a *learning* project the whole point is to
see exactly what went over the wire. We subclass `BaseCallbackHandler`
and emit:

* a `TraceRecord` per call (full prompt + response + tokens + latency),
* rendered to stdout as rich panels for the human running it,
* serialised to JSONL for later analysis.

The handler hooks the standard lifecycle (`on_chat_model_start`,
`on_llm_end`, `on_llm_error`) and keys per-call state on `run_id`. It
works for any LangChain chat model, not just the ones in this agent.

→ Go look at: [`src/coding_agent/trace.py`](../src/coding_agent/trace.py)

---

## 7 · Why the test suite hits no APIs

`FakeMessagesListChatModel` (with our `_BindableFakeChatModel` tweak)
returns scripted `AIMessage`s in order. That lets us test:

* the full graph (`tests/test_graph.py`)
* the trace handler (`tests/test_trace.py`)
* the tools' sandbox (`tests/test_tools.py`)
* the workspace context collector (`tests/test_context.py`)
* the compression policy (`tests/test_compression.py`)

…without spending a token, in 0.3 seconds. That's not just a cost
optimisation — it's what lets you iterate on the agent loop with the
confidence of a normal test suite.

→ Go look at: [`tests/test_graph.py`](../tests/test_graph.py)

---

## 8 · What's deliberately missing

Things this repo does not (yet) do, all of which are within scope and
have natural hooks in the existing code:

* **Subagent delegation.** Wire a `spawn_subagent` tool that compiles
  the same graph with a tighter `max_iterations` and a sub-directory
  workspace; return its `final_answer` as the tool result.
* **Human-in-the-loop approval.** Set `interrupt_before=["act"]` when
  compiling the graph; let the operator approve / edit / reject the
  pending tool calls.
* **Persistent sessions.** Add a `MemorySaver` (or `SqliteSaver`) when
  compiling; thread an ID through the config; you can now resume.
* **Model-based summariser.** Replace `_summarise` in `compression.py`
  with a small LLM call. The signature stays the same.

Each is a short follow-up. The graph and state are structured so they
don't require rewrites.
