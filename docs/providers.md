# Providers

The agent supports five chat-model providers — three hosted APIs, one
local open-weights server, and a scripted fake. The choice lives in
`.env` (or any env var; `.env` is just a convenient default):

```
CODING_AGENT_PROVIDER=gemini   # one of: gemini | deepseek | openai | ollama | fake
```

## Configuring each provider

### Gemini

```env
CODING_AGENT_PROVIDER=gemini
GOOGLE_API_KEY=AI...
GEMINI_MODEL=gemini-2.5-flash       # or gemini-2.5-pro, etc.
```

Get a key at <https://aistudio.google.com/apikey>. Uses
`langchain-google-genai`.

### DeepSeek

```env
CODING_AGENT_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat        # or deepseek-reasoner
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Get a key at <https://platform.deepseek.com/>. DeepSeek exposes an
OpenAI-compatible API, so we reach it through `langchain-openai` with a
custom `base_url`.

### OpenAI

```env
CODING_AGENT_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini            # or gpt-4o, o1-mini, etc.
# OPENAI_BASE_URL=                  # only set for Azure or a proxy
```

Get a key at <https://platform.openai.com/api-keys>. Leave
`OPENAI_BASE_URL` blank to use the standard OpenAI endpoint.

### Ollama (local open-weights — Qwen by default)

```env
CODING_AGENT_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=16384
```

No API key, no per-token bill: the model runs on your own machine through
[Ollama](https://ollama.com). Uses `langchain-ollama`'s `ChatOllama`,
which lives in an optional extra so the base install stays small:

```bash
pip install -e '.[ollama]'

# Install Ollama itself: https://ollama.com/download  (Linux: the
# install script needs `zstd`), then:
ollama serve                 # if it isn't already running as a service
ollama pull qwen3.5:9b       # ~6.6 GB download (q4_K_M)
python examples/02_real_run.py "read hello.txt and tell me what it says"
```

The factory constructs `ChatOllama(..., validate_model_on_init=True)`, so
a stopped server or an un-pulled model fails **at startup** with a message
naming both fixes, rather than as an `httpx` error halfway through the
first `choose` node.

#### Why `qwen3.5:9b` is the default

Chosen from the [Ollama library](https://ollama.com/search?c=tools&q=qwen)
in October 2026, looking for the newest Qwen tag that (a) carries the
**tools** capability, so `bind_tools` produces native tool calls rather
than JSON-in-text, and (b) fits a laptop (roughly 7–14B):

| Candidate | Size (q4) | Why not the default |
|-----------|-----------|---------------------|
| **`qwen3.5:9b`** | 6.6 GB | **Chosen.** Newest family in range; tools + thinking; 256K native context. |
| `qwen3:8b` / `qwen3:14b` | 5.2 / 9.3 GB | Previous generation; 40K context. Good fallbacks. |
| `qwen2.5-coder:7b` / `:14b` | 4.7 / 9.0 GB | Coder-tuned but two generations old; tool calls more often leak into plain text. |
| `qwen3-coder` | 30B+ | Too big for most laptops. |

`qwen3.5` is a *thinking* model: by default it reasons before answering,
which costs latency but tends to improve tool choice. Override
`OLLAMA_MODEL` freely — any tag with the "tools" badge works.

> **Not live-tested at 9B.** The cloud sandbox this was built in has no
> GPU, so the live runs below used ≤1.5B models. The 9B default is a
> recommendation from the library listing, not a measured result.

#### The `num_ctx` gotcha

Ollama's default context window is small (a few thousand tokens,
depending on version and available memory) — far below what the model
supports. A prompt longer than that is **silently truncated from the
front**. For this agent the front is the *stable system prefix*: the
policy and the workspace tree. Truncation doesn't error; the model just
quietly forgets its rules and the tool list's context.

So the factory always passes `num_ctx=settings.ollama_num_ctx`
(default 16384). Raise it for big workspaces — memory use grows with it —
or shrink it on small machines. If a local run starts behaving as if it
never saw the system prompt, this is the first thing to check.

#### How reliable is tool calling on small local models?

Measured in the build sandbox (CPU-only, Ollama 0.40.0, `num_ctx=16384`),
task `"Read hello.txt and tell me exactly what it says."`, 3 runs each:

| Model | Native `read_file` call | Answer correct |
|-------|-------------------------|----------------|
| `qwen2.5:0.5b` | 3/3 | 0/3 — misquotes the file as "Hello, World!" |
| `qwen2.5:1.5b` | 3/3 | 3/3 |
| `qwen3.5:0.8b` | 3/3 | 3/3 |

A multi-step task tells a different story: `"Create hi.py containing a
Python program that prints hi, then run it with: python3 hi.py and report
the output."` (`CODING_AGENT_MAX_ITERATIONS=8`, 3 runs each):

| Model | Wrote a correct `hi.py` | Ran it and reported `hi` | What went wrong |
|-------|-------------------------|--------------------------|-----------------|
| `qwen2.5:1.5b` | 3/3 | 0/3 | After the first native call, it *typed* the next call as text (`[tool_call] run_shell({...})`) or a fake result (`OK: created hi.py`), which the loop treats as a final answer. |
| `qwen3.5:0.8b` | 3/3 | 0/3 | Kept calling tools (rewriting, re-reading, re-running `hi.py`) until the iteration guard stopped it, 3/3. It used absolute paths copied from the system prompt; they resolve inside the workspace, so the sandbox allowed them. |

The `qwen2.5:1.5b` failure is worth understanding. `prompts.py` replays
past tool calls as plain text (`[tool_call] name(args)`) to stay
provider-agnostic, and a small model will imitate that format instead of
emitting a native tool call. Larger models don't, in practice, but if you
build on small local models, replaying history as native
`AIMessage(tool_calls=...)` / `ToolMessage` pairs is the fix.

Rules of thumb:

* **Tool *selection* is easy; tool *sequencing* is hard.** Even sub-1B
  models emit well-formed calls for single-step tasks. Multi-step plans
  (read → edit → run → check) are where small models stall, repeat
  themselves, or answer before finishing.
* `CODING_AGENT_MAX_ITERATIONS` is your friend — a small model stuck in a
  loop is stopped by the iteration guard in `choose`, not by your patience.
* Prefer 7B+ for real work; ≤1.5B is fine for proving the wiring.
* Keep `temperature=0.0` (the factory does): sampling noise turns into
  malformed tool arguments quickly at small sizes.

### Fake (no key, no network)

```env
CODING_AGENT_PROVIDER=fake
```

Used by the test suite and `examples/01_smoke_test_fake.py`. You pass in
a list of `AIMessage`s when calling `run_agent` and the model returns
them in order, cycling when exhausted. Perfect for deterministic tests
of the graph, tools, and tracing without spending a token.

## How the factory works

The agent never imports a concrete provider class. It calls
[`build_chat_model(settings)`](../src/coding_agent/llm.py) and gets back
a `BaseChatModel`. That single seam is what makes swapping providers
trivial and what makes the fake-provider test path possible.

```python
# src/coding_agent/llm.py (sketch)

def build_chat_model(settings, *, callbacks=None, fake_responses=None):
    if settings.provider == "fake":
        return _BindableFakeChatModel(responses=list(fake_responses))
    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.gemini_model, ...)
    if settings.provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(base_url=settings.deepseek_base_url, ...)
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(...)
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama        # optional extra
        return ChatOllama(model=..., base_url=..., num_ctx=...)
```

Providers are imported lazily so users on (say) DeepSeek don't have to
install the Google SDK — and `langchain-ollama` isn't installed at all
unless you ask for the `[ollama]` extra.

## Adding a new provider

1. Add the provider name to the `Provider` literal and the validation
   set in `config.py`.
2. Add API key + model fields to `Settings` and parse them in
   `load_settings`.
3. Add a branch in `build_chat_model` that returns a `BaseChatModel`
   for the new provider.
4. Document it here.

That's the whole change — no graph or prompt edits required.

## Tool calling support

The agent always calls `model.bind_tools(tools)` in `choose`. All four
real providers above implement `bind_tools` in their LangChain
integrations. If you add a provider that doesn't (some older
integrations), wrap it like `_BindableFakeChatModel` does in `llm.py`
and either route tool calls through structured prompting or use a
LangChain `ToolNode` adapter.
