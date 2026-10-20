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


def test_stream_sentences_splits_the_streamed_answer_into_sentences(a2a_agent_url):
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    sentences = list(engine.stream_sentences(
        msgs((USER, "chunks Hello there. How are you? Fine, thanks.")), session_id="sent"))
    assert sentences == ["Hello there.", "How are you?", "Fine, thanks."]


def test_stream_sentences_yields_a_sentence_before_the_stream_ends(a2a_agent_url, monkeypatch):
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    consumed = []

    def tokens(*args, **kwargs):
        for chunk in ("Hello there. ", "How are you? ", "Fine, thanks. "):
            consumed.append(chunk)
            yield chunk

    monkeypatch.setattr(engine, "stream_tokens", tokens)
    stream = engine.stream_sentences(msgs((USER, "x")))
    assert next(stream) == "Hello there."
    assert consumed == ["Hello there. ", "How are you? "]
    assert list(stream) == ["How are you?", "Fine, thanks."]


def test_stream_sentences_yields_an_unpunctuated_line_when_the_next_artifact_starts(a2a_agent_url, monkeypatch):
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    consumed = []

    def tokens(*args, **kwargs):
        for chunk in ("First part", "\nSecond part", "\nThird part"):
            consumed.append(chunk)
            yield chunk

    monkeypatch.setattr(engine, "stream_tokens", tokens)
    stream = engine.stream_sentences(msgs((USER, "x")))
    assert next(stream) == "First part"
    assert consumed == ["First part", "\nSecond part"]
    assert list(stream) == ["Second part", "Third part"]
    engine._client.close()


def test_stream_sentences_uses_the_streaming_call_of_the_agent(a2a_agent_url, monkeypatch):
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    calls = []
    for name in ("send_task", "stream_task"):
        original = getattr(engine._client, name)
        monkeypatch.setattr(engine._client, name, lambda *a, _n=name, _o=original, **kw: (calls.append(_n), _o(*a, **kw))[1])
    list(engine.stream_sentences(msgs((USER, "chunks a b"))))
    assert calls == ["stream_task"]


def test_stream_sentences_without_streaming_yields_the_whole_answer(engine):
    (sentence,) = engine.stream_sentences(msgs((USER, "One. Two.")), session_id="whole")
    assert answer_text(sentence) == "One. Two."


@pytest.mark.parametrize("shape, sentences", [
    ("adk", ["The answer is 42.", "Have a nice day."]),
    ("progress", ["The answer is 42.", "Have a nice day."]),
    ("cumulative", ["The answer is 42.", "Have a nice day."]),
    ("artifacts", ["First part.", "Second part."]),
])
def test_stream_sentences_speaks_each_stream_shape_once(a2a_agent_url, shape, sentences):
    engine = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    assert list(engine.stream_sentences(msgs((USER, shape)), session_id=shape)) == sentences


@pytest.mark.parametrize("shape, sentences", [
    ("rewrite", ["The capital of Australia is Sydney.", "The capital of Australia is Canberra."]),
    ("interim", ["Thinking...", "The answer is 42.", "Have a nice day."]),
    ("helloworld", ["Hello, World!", "I am your agent."]),
    ("nopunct", ["First part", "Second part"]),
    ("wsrewrite", ["The answer is 42.", "Have a nice day."]),
    ("tworewrites", ["Alpha old.", "Beta old.", "Alpha new.", "Beta new."]),
    ("poem", ["Roses are red", "Violets are blue", "Sugar is sweet", "And so are you."]),
])
def test_stream_sentences_ends_with_the_final_answer(a2a_agent_url, shape, sentences):
    streaming = A2AChatEngine(config={"agent_url": a2a_agent_url, "streaming": True})
    blocking = A2AChatEngine(config={"agent_url": a2a_agent_url})
    final = list(blocking.stream_sentences(msgs((USER, shape)), session_id=shape))
    spoken = list(streaming.stream_sentences(msgs((USER, shape)), session_id=shape))
    assert spoken == sentences
    assert " ".join(spoken).endswith(" ".join(final))
