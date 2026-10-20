"""A2AChatEngine against a real a2a-sdk agent (see a2a_agent.py)."""
import re

import pytest
from a2a.client.errors import AgentCardResolutionError

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_a2a_solver.client import A2AClient, AgentCard
from ovos_a2a_solver.engine import A2AChatEngine

USER, ASSISTANT, SYSTEM = MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM


def answer_text(answer: str) -> str:
    """The text part of the test agent's ``echo[<contextId>]: <text>`` answer."""
    match = re.fullmatch(r"echo\[[^\]]+\]: (.*)", answer, re.DOTALL)
    assert match, answer
    return match[1]


def msgs(*turns):
    return [AgentMessage(role=role, content=text) for role, text in turns]


@pytest.fixture
def engine(a2a_agent_url):
    return A2AChatEngine(config={"agent_url": a2a_agent_url})


def test_init_reads_the_agent_card(engine):
    assert engine._card.name == "test-agent"


def test_continue_chat_returns_the_agent_answer(engine):
    reply = engine.continue_chat(msgs((USER, "Ping")), session_id="sess-1")
    assert reply.role == ASSISTANT
    assert answer_text(reply.content) == "Ping"


def test_continue_chat_sends_only_the_pending_user_message(engine):
    history = msgs((SYSTEM, "Be helpful."), (USER, "Q1"), (ASSISTANT, "A1"), (USER, "Q2"))
    assert answer_text(engine.continue_chat(history, session_id="multi").content) == "Q2"


def test_continue_chat_accepts_and_ignores_tools(engine):
    assert engine.supports_tools is False
    reply = engine.continue_chat(msgs((USER, "Ping")), session_id="t", tools=[{"type": "function"}])
    assert answer_text(reply.content) == "Ping"


def test_continue_chat_without_pending_user_message_returns_empty(engine):
    assert engine.continue_chat(msgs((USER, "Q1"), (ASSISTANT, "A1"))).content == ""


def test_continue_chat_raises_when_the_task_fails(engine):
    with pytest.raises(RuntimeError, match="backend exploded"):
        engine.continue_chat(msgs((USER, "fail")))


def test_stream_tokens_streams_chunks(a2a_agent_url):
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    assert engine._streaming is True
    assert list(engine.stream_tokens(msgs((USER, "chunks a b")))) == ["a ", "b "]


def test_stream_tokens_falls_back_to_one_answer_without_streaming(engine):
    (chunk,) = engine.stream_tokens(msgs((USER, "hi")), session_id="fb")
    assert answer_text(chunk) == "hi"


def test_stream_tokens_without_pending_user_message_yields_nothing(a2a_agent_url):
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    assert list(engine.stream_tokens(msgs((SYSTEM, "sys")))) == []


def test_streaming_is_disabled_when_the_card_does_not_offer_it(a2a_agent_url, monkeypatch):
    card = AgentCard(name="NoStream", description="", url=a2a_agent_url, streaming=False)
    monkeypatch.setattr(A2AClient, "fetch_agent_card", lambda self: card)
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    assert engine._streaming is False


def test_unreachable_agent_does_not_abort_init_but_fails_the_chat():
    engine = A2AChatEngine(config={"agent_url": "http://127.0.0.1:1", "timeout": "3"})
    assert engine._card is None
    assert engine._client.timeout == 3.0
    with pytest.raises(AgentCardResolutionError):
        engine.continue_chat(msgs((USER, "hello")))


def test_missing_agent_url_raises():
    with pytest.raises(ValueError, match="agent_url"):
        A2AChatEngine(config={})


@pytest.mark.parametrize("turns, expected", [
    ([(SYSTEM, "s"), (USER, "Q1"), (ASSISTANT, "A1"), (USER, "Q2")], "Q2"),
    ([(USER, "Q1"), (USER, "Q2")], "Q2"),
    ([(USER, "Q1"), (ASSISTANT, "A1")], ""),
    ([(ASSISTANT, "A1")], ""),
    ([], ""),
])
def test_pending_user_text(turns, expected):
    assert A2AChatEngine._pending_user_text(msgs(*turns)) == expected


def test_each_session_has_its_own_remote_conversation(engine):
    def context(session_id):
        reply = engine.continue_chat(msgs((USER, "hi")), session_id=session_id).content
        return re.fullmatch(r"echo\[([^\]]+)\]: hi", reply)[1]

    kitchen = context("kitchen")
    assert context("kitchen") == kitchen
    assert context("bedroom") != kitchen
