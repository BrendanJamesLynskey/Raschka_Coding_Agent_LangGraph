# Providers

The agent supports four chat-model providers. The choice lives in
`.env` (or any env var; `.env` is just a convenient default):

```
CODING_AGENT_PROVIDER=gemini   # one of: gemini | deepseek | openai | fake
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
```

Providers are imported lazily so users on (say) DeepSeek don't have to
install the Google SDK.

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

The agent always calls `model.bind_tools(tools)` in `choose`. All three
real providers above implement `bind_tools` in their LangChain
integrations. If you add a provider that doesn't (some older
integrations), wrap it like `_BindableFakeChatModel` does in `llm.py`
and either route tool calls through structured prompting or use a
LangChain `ToolNode` adapter.
