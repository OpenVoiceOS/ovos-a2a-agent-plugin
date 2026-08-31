"""Tests for A2AChatEngine — solver contract with a mocked A2A server."""

import json
import pytest
import httpx
import respx

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_a2a_solver.engine import A2AChatEngine


BASE_URL = "https://engine.test"

CARD_RESPONSE = {
    "name": "EngineAgent",
    "description": "Mocked engine",
    "url": BASE_URL,
    "version": "1.0",
    "capabilities": {"streaming": False},
    "skills": [],
}

TASK_OK = {
    "jsonrpc": "2.0",
    "id": "x",
    "result": {
        "artifacts": [{"parts": [{"type": "text", "text": "Pong!"}]}]
    },
}


def _make_engine(streaming: bool = False) -> A2AChatEngine:
    return A2AChatEngine(config={"agent_url": BASE_URL, "streaming": streaming})


@respx.mock
def test_continue_chat_returns_assistant_message():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=TASK_OK))

    engine = _make_engine()
    messages = [AgentMessage(role=MessageRole.USER, content="Ping")]
    reply = engine.continue_chat(messages, session_id="sess-1")
    assert reply.role == MessageRole.ASSISTANT
    assert reply.content == "Pong!"


@respx.mock
def test_continue_chat_multi_turn_history():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    captured = {}

    def capture(request: httpx.Request, *_):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=TASK_OK)

    respx.post(BASE_URL).mock(side_effect=capture)

    engine = _make_engine()
    messages = [
        AgentMessage(role=MessageRole.USER, content="Hello"),
        AgentMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        AgentMessage(role=MessageRole.USER, content="How are you?"),
    ]
    engine.continue_chat(messages)
    params = captured["body"]["params"]
    # Last user message goes into "message"
    assert params["message"]["parts"][0]["text"] == "How are you?"
    # Prior turns go into history
    assert len(params["history"]) == 2
    assert params["history"][0]["role"] == "user"
    assert params["history"][1]["role"] == "assistant"


@respx.mock
def test_continue_chat_system_message_prepended_to_history():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    captured = {}

    def capture(request: httpx.Request, *_):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=TASK_OK)

    respx.post(BASE_URL).mock(side_effect=capture)

    engine = _make_engine()
    messages = [
        AgentMessage(role=MessageRole.SYSTEM, content="You are a pirate."),
        AgentMessage(role=MessageRole.USER, content="Where is the treasure?"),
    ]
    engine.continue_chat(messages)
    params = captured["body"]["params"]
    history = params.get("history", [])
    # System turn is injected at index 0 as a user message (A2A parts format)
    assert history[0]["role"] == "user"
    assert "pirate" in history[0]["parts"][0]["text"]


@respx.mock
def test_continue_chat_no_user_message_returns_empty():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    engine = _make_engine()
    messages = [AgentMessage(role=MessageRole.SYSTEM, content="Only system")]
    reply = engine.continue_chat(messages)
    assert reply.role == MessageRole.ASSISTANT
    assert reply.content == ""


@respx.mock
def test_stream_tokens_fallback_when_not_streaming():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=TASK_OK))

    engine = _make_engine(streaming=False)
    messages = [AgentMessage(role=MessageRole.USER, content="Ping")]
    chunks = list(engine.stream_tokens(messages))
    assert chunks == ["Pong!"]


def test_missing_agent_url_raises():
    with pytest.raises(ValueError, match="agent_url"):
        A2AChatEngine(config={})


@respx.mock
def test_card_fetch_failure_does_not_abort_init():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(404)
    )
    # Should not raise; card is optional
    engine = A2AChatEngine(config={"agent_url": BASE_URL})
    assert engine._card is None


@respx.mock
def test_continue_chat_accepts_and_ignores_tools_kwarg():
    """Base ChatEngine.continue_chat contract declares a ``tools`` kwarg
    (ovos_plugin_manager.templates.agents.ChatEngine.continue_chat). Engines
    that do not support native tool calling must still accept it and ignore
    it, so the agentic loop's positional call sites keep working and so any
    future caller that passes ``tools=`` by keyword does not blow up."""
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=TASK_OK))

    engine = _make_engine()
    assert engine.supports_tools is False
    messages = [AgentMessage(role=MessageRole.USER, content="Ping")]
    reply = engine.continue_chat(messages, session_id="sess-1", tools=[{"type": "function"}])
    assert isinstance(reply, AgentMessage)
    assert reply.role == MessageRole.ASSISTANT
    assert reply.content == "Pong!"


def test_split_messages_helper():
    messages = [
        AgentMessage(role=MessageRole.SYSTEM, content="Be helpful."),
        AgentMessage(role=MessageRole.USER, content="Q1"),
        AgentMessage(role=MessageRole.ASSISTANT, content="A1"),
        AgentMessage(role=MessageRole.USER, content="Q2"),
    ]
    user_text, history = A2AChatEngine._split_messages(messages)
    assert user_text == "Q2"
    # history: system (as user), Q1, A1
    assert len(history) == 3
    assert history[0]["content"] == "Be helpful."
    assert history[1] == {"role": "user", "content": "Q1"}
    assert history[2] == {"role": "assistant", "content": "A1"}
