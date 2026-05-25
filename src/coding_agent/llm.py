"""LLM factory — provider selection and the fake model used for tests.

The agent never imports a concrete provider directly. It calls
``build_chat_model(settings)`` and gets back a ``BaseChatModel`` it can talk
to identically. That single seam is what makes it cheap to swap providers
(or drop in a fake during tests).

Supported providers
-------------------
* ``gemini``   — Google's Gemini via ``langchain-google-genai``.
* ``deepseek`` — DeepSeek's OpenAI-compatible API, reached through
  ``langchain-openai`` with a ``base_url`` override.
* ``openai``   — OpenAI's own models via ``langchain-openai``.
* ``fake``     — A scripted in-memory chat model. Pre-load it with the
  ``AIMessage``s you want returned in order. Used in tests and the smoke
  example. No network, no key.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from coding_agent.config import Settings


class _BindableFakeChatModel(FakeMessagesListChatModel):
    """A FakeMessagesListChatModel that accepts ``bind_tools``.

    The default fake model raises ``NotImplementedError`` on ``bind_tools``,
    which breaks the agent graph that always binds tools. We override the
    method to return ``self`` because the scripted AIMessages we feed in
    already include their own ``tool_calls`` — there is no real model to
    teach about the tool schemas.
    """

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        # Ignore the tools — scripted responses already carry tool_calls.
        return self


def build_chat_model(
    settings: Settings,
    *,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
    fake_responses: Sequence[AIMessage] | None = None,
) -> BaseChatModel:
    """Construct the chat model for the configured provider.

    Parameters
    ----------
    settings:
        The loaded ``Settings`` (provider name + keys + model names).
    callbacks:
        Callbacks to attach to the model — typically the tracing handler.
        We attach them at construction time so *every* call made via this
        model is traced, without each caller having to remember.
    fake_responses:
        Only used when ``settings.provider == "fake"``. The list of
        ``AIMessage``s the fake model will return in order. The fake will
        cycle if more calls are made than messages provided.
    """
    cbs = list(callbacks) if callbacks else None

    if settings.provider == "fake":
        if not fake_responses:
            raise ValueError(
                "provider='fake' requires fake_responses to be supplied "
                "(a list of AIMessages the model should return in order)."
            )
        # Returns messages in order and cycles when exhausted — exactly
        # what we want for deterministic tests.
        model = _BindableFakeChatModel(responses=list(fake_responses))
        if cbs:
            model = model.with_config({"callbacks": cbs})
        return model

    if settings.provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError(
                "CODING_AGENT_PROVIDER=gemini but GOOGLE_API_KEY is not set."
            )
        # Import lazily so users on (say) DeepSeek don't need the Google deps.
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            # `convert_system_message_to_human=False` keeps the system role
            # explicit, which the article's "stable prefix" pattern relies on.
            temperature=0.0,
            callbacks=cbs,
        )

    if settings.provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError(
                "CODING_AGENT_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set."
            )
        # DeepSeek exposes an OpenAI-compatible endpoint, so we use
        # langchain-openai with a custom base_url and api_key.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.0,
            callbacks=cbs,
        )

    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "CODING_AGENT_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        from langchain_openai import ChatOpenAI

        kwargs = dict(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
            callbacks=cbs,
        )
        # Only pass base_url if the user explicitly set one — otherwise the
        # SDK's own default (api.openai.com) is used.
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    # Should be unreachable thanks to validation in config.load_settings.
    raise ValueError(f"Unsupported provider: {settings.provider!r}")
