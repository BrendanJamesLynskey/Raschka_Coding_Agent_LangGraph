"""Tracing handler: it captures calls, writes JSONL, doesn't crash on empty.

These tests use the FakeMessagesListChatModel directly so we exercise the
real LangChain callback lifecycle without touching a network.
"""

from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from coding_agent.trace import TracingCallbackHandler


def test_handler_captures_call(settings):
    tracer = TracingCallbackHandler(trace_dir=settings.trace_dir, stdout=False)
    model = FakeMessagesListChatModel(responses=[AIMessage(content="hello back")])
    model.invoke([HumanMessage(content="hi")], config={"callbacks": [tracer]})

    assert len(tracer.records) == 1
    rec = tracer.records[0]
    assert rec.response.content == "hello back"
    assert rec.prompt[0].content == "hi"
    assert rec.duration_s >= 0


def test_jsonl_written(settings):
    tracer = TracingCallbackHandler(trace_dir=settings.trace_dir, stdout=False)
    model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    model.invoke([HumanMessage(content="ping")], config={"callbacks": [tracer]})

    contents = tracer.file_path.read_text().strip().splitlines()
    assert len(contents) == 1
    parsed = json.loads(contents[0])
    assert parsed["response"]["content"] == "ok"
    assert parsed["prompt"][0]["content"] == "ping"


def test_tool_call_recorded(settings):
    tracer = TracingCallbackHandler(trace_dir=settings.trace_dir, stdout=False)
    ai = AIMessage(
        content="calling",
        tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "c1"}],
    )
    model = FakeMessagesListChatModel(responses=[ai])
    model.invoke([HumanMessage(content="go")], config={"callbacks": [tracer]})

    rec = tracer.records[0]
    assert rec.response is not None
    assert rec.response.tool_calls == [
        {"name": "read_file", "args": {"path": "x"}, "id": "c1"}
    ]
