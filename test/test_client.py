"""A2AClient against a real a2a-sdk agent (see a2a_agent.py)."""
import gc
import re
import threading
import time

import httpx
import pytest
from a2a.client.errors import A2AClientError, A2AClientTimeoutError, AgentCardResolutionError
from a2a.utils.errors import A2AError, InvalidParamsError, TaskNotFoundError, UnsupportedOperationError

from ovos_a2a_solver import client as client_module
from ovos_a2a_solver.client import A2AClient, AgentCard

ECHO = re.compile(r"echo\[(?P<context>[^\]]+)\]: (?P<text>.*)", re.DOTALL)


def echo(answer: str) -> tuple[str, str]:
    """Split the test agent's ``echo[<contextId>]: <text>`` answer."""
    match = ECHO.fullmatch(answer)
    assert match, answer
    return match["context"], match["text"]


@pytest.fixture
def client(a2a_agent_url):
    with A2AClient(a2a_agent_url, timeout=10) as c:
        yield c


def test_fetch_agent_card_reads_the_well_known_card(client, a2a_agent_url):
    card = client.fetch_agent_card()
    assert isinstance(card, AgentCard)
    assert card.name == "test-agent"
    assert card.url == a2a_agent_url
    assert card.streaming is True
    assert [s.id for s in card.skills] == ["echo"]


def test_send_task_returns_the_agent_answer(client):
    assert echo(client.send_task("hello", session_id="s-1"))[1] == "hello"


def test_the_agent_assigns_the_context_and_not_the_session_id(client):
    context, _ = echo(client.send_task("hello", session_id="kitchen"))
    assert context != "kitchen"


def test_one_session_keeps_its_context_across_turns(client):
    first, _ = echo(client.send_task("one", session_id="conv-7"))
    second, _ = echo(client.send_task("two", session_id="conv-7"))
    streamed, _ = echo("".join(client.stream_task("three", session_id="conv-7")))
    assert first == second == streamed


def test_two_sessions_get_two_contexts(client):
    device_1, _ = echo(client.send_task("hello", session_id="default"))
    device_2, _ = echo(A2AClient(client.base_url, timeout=10).send_task("hello", session_id="default"))
    other, _ = echo(client.send_task("hello", session_id="living-room"))
    assert len({device_1, device_2, other}) == 3


def test_no_session_starts_a_new_context_each_time(client):
    assert echo(client.send_task("a"))[0] != echo(client.send_task("b"))[0]


def test_the_context_map_keeps_the_most_recent_sessions(client, monkeypatch):
    monkeypatch.setattr(client_module, "MAX_SESSIONS", 2)
    contexts = {s: echo(client.send_task("x", session_id=s))[0] for s in ("a", "b")}
    assert echo(client.send_task("x", session_id="a"))[0] == contexts["a"]
    client.send_task("x", session_id="c")
    assert list(client._sessions) == ["a", "c"]
    assert echo(client.send_task("x", session_id="b"))[0] != contexts["b"]


def continued(reply: str, text: str) -> bool:
    """True when the test agent continued the task waiting for input, not a new one."""
    return re.fullmatch(rf"continued\[[^\]]+\]: {text}", reply) is not None


def test_the_answer_to_a_question_continues_the_same_task(client):
    first_context, _ = echo(client.send_task("hello", session_id="trip"))
    assert client.send_task("ask", session_id="trip") == "which city?"
    assert continued(client.send_task("lisbon", session_id="trip"), "lisbon")
    context, text = echo(client.send_task("thanks", session_id="trip"))
    assert (context, text) == (first_context, "thanks")


def test_the_streaming_answer_to_a_question_continues_the_same_task(client):
    assert "".join(client.stream_task("ask", session_id="trip-s")) == "which city?"
    assert continued("".join(client.stream_task("porto", session_id="trip-s")), "porto")


def test_the_answer_to_an_auth_request_continues_the_same_task(client):
    assert client.send_task("authreq", session_id="bank") == "login please"
    assert continued(client.send_task("token-9", session_id="bank"), "token-9")


def test_a_task_the_agent_purged_keeps_the_conversation(start_a2a_agent):
    agent, url = start_a2a_agent()
    with A2AClient(url, timeout=10) as c:
        context, _ = echo(c.send_task("hello", session_id="trip"))
        assert c.send_task("ask", session_id="trip") == "which city?"
        agent.kill()
        agent.wait(timeout=10)
        start_a2a_agent(port=int(url.rsplit(":", 1)[1]))
        with pytest.raises(TaskNotFoundError):
            c.send_task("lisbon", session_id="trip")
        assert echo(c.send_task("hello", session_id="trip")) == (context, "hello")


def test_a_task_finished_out_of_band_keeps_the_conversation(client):
    assert client.send_task("remember gold", session_id="vault") == "stored"
    assert client.send_task("authoob 1", session_id="vault") == "open the link to log in"
    time.sleep(2)
    with pytest.raises(UnsupportedOperationError):
        client.send_task("token-9", session_id="vault")
    assert client.send_task("recall", session_id="vault") == "gold"


@pytest.mark.parametrize("command, error", [
    ("forget", InvalidParamsError),
    ("expire", TaskNotFoundError),
    ("expire-terminal", UnsupportedOperationError),
])
@pytest.mark.parametrize("streaming", [False, True])
def test_a_rejected_context_starts_a_new_conversation(client, command, error, streaming):
    session = f"studio-{command}-{streaming}"

    def send(text):
        return "".join(client.stream_task(text, session_id=session)) if streaming \
            else client.send_task(text, session_id=session)

    assert send("remember blue") == "stored"
    context, _ = echo(send("hello"))
    assert send(command) == "forgotten"
    with pytest.raises(error):
        send("recall")
    assert send("recall") == "nothing"
    assert echo(send("hello"))[0] != context


def test_a_timeout_keeps_the_open_task(a2a_agent_url):
    with A2AClient(a2a_agent_url, timeout=1) as c:
        assert c.send_task("ask", session_id="patient") == "which city?"
        with pytest.raises(A2AClientTimeoutError):
            c.send_task("stall", session_id="patient")
        assert continued(c.send_task("lisbon", session_id="patient"), "lisbon")


def test_a_timeout_keeps_the_session(a2a_agent_url):
    with A2AClient(a2a_agent_url, timeout=1) as c:
        context, _ = echo(c.send_task("hello", session_id="flaky"))
        with pytest.raises(A2AClientTimeoutError):
            c.send_task("slow 3 x", session_id="flaky")
        assert echo(c.send_task("again", session_id="flaky"))[0] == context


def test_send_task_joins_appended_artifact_chunks(client):
    assert client.send_task("chunks alpha beta gamma") == "alpha beta gamma "


def test_stream_task_yields_each_artifact_chunk_once(client):
    assert list(client.stream_task("chunks alpha beta gamma")) == ["alpha ", "beta ", "gamma "]


def test_stream_task_does_not_repeat_the_final_task_snapshot(client):
    assert list(client.stream_task("snapshot a b")) == ["a ", "b "]


def test_send_task_reads_the_final_task_snapshot(client):
    assert client.send_task("snapshot a b") == "a b "


def test_stream_task_yields_an_agent_message(client):
    assert echo("".join(client.stream_task("hi", session_id="s-2")))[1] == "hi"


@pytest.mark.parametrize("text, answer", [
    ("ask", "which city?"),
    ("authreq", "login please"),
    ("done", "final words"),
])
def test_send_task_answers_with_the_task_status_message(client, text, answer):
    assert client.send_task(text) == answer


@pytest.mark.parametrize("text, answer", [("ask", "which city?"), ("done", "final words")])
def test_stream_task_answers_with_the_task_status_message(client, text, answer):
    assert "".join(client.stream_task(text)) == answer


def test_failed_task_raises_with_the_agent_reason(client):
    with pytest.raises(RuntimeError, match="TASK_STATE_FAILED: backend exploded"):
        client.send_task("fail")


def test_failed_streaming_task_raises(client):
    with pytest.raises(RuntimeError, match="TASK_STATE_FAILED: backend exploded"):
        list(client.stream_task("fail"))


def test_unicode_round_trip(client):
    text = "olá 👋 日本語"
    assert echo(client.send_task(text, session_id="u"))[1] == text


def test_auth_header_reaches_the_agent(a2a_agent_url):
    with A2AClient(a2a_agent_url, auth_header="Bearer s3cret", timeout=10) as c:
        c.send_task("hello")
    assert httpx.get(a2a_agent_url + "/last-auth").text == "Bearer s3cret"


def test_auth_header_reaches_the_agent_on_the_streaming_path(a2a_agent_url):
    with A2AClient(a2a_agent_url, auth_header="Bearer str3am", timeout=10) as c:
        assert echo("".join(c.stream_task("hello")))[1] == "hello"
    assert httpx.get(a2a_agent_url + "/last-auth").text == "Bearer str3am"


@pytest.mark.parametrize("status", [401, 429, 500])
@pytest.mark.parametrize("streaming", [False, True])
def test_http_error_status_raises_a_client_error(a2a_agent_url, status, streaming):
    with A2AClient(a2a_agent_url, auth_header=f"Bearer status-{status}", timeout=10) as c:
        with pytest.raises(A2AClientError, match=str(status)):
            list(c.stream_task("hello")) if streaming else c.send_task("hello")


def test_slow_agent_raises_a_timeout(a2a_agent_url):
    with A2AClient(a2a_agent_url, timeout=1) as c:
        start = time.monotonic()
        with pytest.raises(A2AClientTimeoutError):
            c.send_task("slow 5 x")
        assert time.monotonic() - start < 4


def test_unreachable_agent_raises_a_card_error():
    with A2AClient("http://127.0.0.1:1", timeout=3) as c:
        with pytest.raises(AgentCardResolutionError):
            c.send_task("hello")


def test_agent_killed_mid_call_raises(a2a_agent_process):
    proc, url = a2a_agent_process
    with A2AClient(url, timeout=10) as c:
        c.fetch_agent_card()
        start = time.monotonic()
        killer = threading.Timer(1.0, proc.kill)
        killer.start()
        with pytest.raises(A2AClientError):
            c.send_task("slow 5 x")
        killer.join()
        assert time.monotonic() - start < 5


def test_an_agent_upgraded_in_place_is_reached_after_one_error(start_a2a_agent):
    old, url = start_a2a_agent("--a2a-0.3")
    with A2AClient(url, timeout=10) as c:
        assert echo(c.send_task("hello"))[1] == "hello"
        old.kill()
        old.wait(timeout=10)
        start_a2a_agent(port=int(url.rsplit(":", 1)[1]))
        with pytest.raises(A2AError):
            c.send_task("hello")
        assert echo(c.send_task("hello"))[1] == "hello"


def loop_threads() -> int:
    return sum(t.name == "a2a-client-loop" for t in threading.enumerate())


def test_a_dropped_client_stops_its_thread(a2a_agent_url):
    gc.collect()
    before = loop_threads()
    for _ in range(5):
        A2AClient(a2a_agent_url, timeout=10).send_task("hello")
    gc.collect()
    deadline = time.monotonic() + 5
    while loop_threads() > before and time.monotonic() < deadline:
        time.sleep(0.1)
    assert loop_threads() == before


def test_close_is_idempotent(a2a_agent_url):
    c = A2AClient(a2a_agent_url)
    c.send_task("hello")
    c.close()
    c.close()


def test_stream_task_speaks_a_repeating_working_status_message_once(client):
    """A working status message that the final artifact repeats (the shape of Google ADK)."""
    assert list(client.stream_task("adk")) == ["The answer is 42. Have a nice day."]


def test_stream_task_skips_working_progress_messages_when_the_answer_comes(client):
    assert list(client.stream_task("progress")) == ["The answer is 42. Have a nice day."]


def test_stream_task_yields_only_the_new_text_of_a_replaced_artifact(client):
    chunks = list(client.stream_task("cumulative"))
    assert "".join(chunks) == "The answer is 42. Have a nice day."


def test_stream_task_separates_distinct_artifacts(client):
    assert "".join(client.stream_task("artifacts")) == "First part.\nSecond part."
