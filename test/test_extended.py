"""Extended unit tests — coverage for gaps identified after initial test pass.

Covers:
- agent-card with missing / extra / multiple-skill fields
- malformed JSON-RPC responses (no result, error object variants)
- HTTP 401 / 429 / 500 handling
- timeout behaviour
- SSE stream malformed events / connection drop mid-stream
- auth header propagation in streaming
- multi-skill agent cards
- unicode / emoji payload round-trip
- history with alternating roles + double-user edge
- config validation (missing url, bad timeout type)
- _extract_stream_chunk variants
- A2AClient context-manager protocol
"""

from __future__ import annotations

import json
import pytest
import httpx
import respx

from ovos_a2a_solver.client import A2AClient, AgentCard, AgentSkill
from ovos_a2a_solver.engine import A2AChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole


BASE_URL = "https://agent.test"

CARD_STREAMING = {
    "name": "StreamAgent",
    "description": "Streaming-capable agent",
    "url": BASE_URL,
    "version": "1.0",
    "capabilities": {"streaming": True},
    "skills": [],
}

TASK_OK = {
    "jsonrpc": "2.0",
    "id": "x",
    "result": {
        "artifacts": [{"parts": [{"type": "text", "text": "Hello!"}]}]
    },
}

# ---------------------------------------------------------------------------
# Agent-card edge cases
# ---------------------------------------------------------------------------

def test_card_with_extra_unknown_fields():
    """Unknown top-level fields in agent card are accepted (stored in raw)."""
    d = {
        "name": "Exotic",
        "description": "agent",
        "url": "https://x.test",
        "version": "3.0",
        "unknown_field": "ignored",
        "another": 42,
    }
    card = AgentCard.from_dict(d)
    assert card.name == "Exotic"
    assert card.raw["unknown_field"] == "ignored"


def test_card_missing_name_defaults_to_empty():
    card = AgentCard.from_dict({"description": "d", "url": "https://x.test"})
    assert card.name == ""


def test_card_missing_description_defaults_to_empty():
    card = AgentCard.from_dict({"name": "N", "url": "https://x.test"})
    assert card.description == ""


def test_card_missing_url_defaults_to_empty():
    card = AgentCard.from_dict({"name": "N", "description": "d"})
    assert card.url == ""


def test_card_multi_skill():
    d = {
        "name": "Multi",
        "description": "multi skill agent",
        "url": "https://multi.test",
        "version": "1.0",
        "skills": [
            {"id": "alpha", "name": "Alpha", "tags": ["a"]},
            {"id": "beta", "name": "Beta", "tags": ["b"], "examples": ["eg1", "eg2"]},
            {"id": "gamma", "name": "Gamma"},
        ],
    }
    card = AgentCard.from_dict(d)
    assert len(card.skills) == 3
    assert card.skills[1].examples == ["eg1", "eg2"]
    assert card.skills[2].tags == []


def test_skill_all_defaults():
    skill = AgentSkill.from_dict({})
    assert skill.id == ""
    assert skill.name == ""
    assert skill.description == ""
    assert skill.tags == []
    assert skill.examples == []


# ---------------------------------------------------------------------------
# Malformed JSON-RPC responses
# ---------------------------------------------------------------------------

@respx.mock
def test_send_task_no_result_key_returns_empty():
    """Response missing both 'result' and 'error' → empty string, no raise."""
    body = {"jsonrpc": "2.0", "id": "1"}
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=body))
    client = A2AClient(BASE_URL)
    result = client.send_task("hello")
    assert result == ""


@respx.mock
def test_send_task_error_with_code_and_message():
    body = {"jsonrpc": "2.0", "id": "1",
            "error": {"code": -32601, "message": "Method not found"}}
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=body))
    client = A2AClient(BASE_URL)
    with pytest.raises(RuntimeError, match="Method not found"):
        client.send_task("test")


@respx.mock
def test_send_task_error_missing_message_field():
    """Error object without 'message' key should still raise RuntimeError."""
    body = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32700}}
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=body))
    client = A2AClient(BASE_URL)
    with pytest.raises(RuntimeError):
        client.send_task("oops")


@respx.mock
def test_send_task_artifacts_empty_parts():
    """Artifact present but parts list is empty → fall through to message."""
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "artifacts": [{"parts": []}],
            "message": {"parts": [{"type": "text", "text": "fallback"}]},
        },
    }
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=body))
    client = A2AClient(BASE_URL)
    assert client.send_task("q") == "fallback"


@respx.mock
def test_send_task_non_text_parts_skipped():
    """Parts with type != 'text' are skipped; text part wins."""
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "artifacts": [
                {
                    "parts": [
                        {"type": "image", "url": "https://img.test/x.png"},
                        {"type": "text", "text": "found it"},
                    ]
                }
            ]
        },
    }
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=body))
    client = A2AClient(BASE_URL)
    assert client.send_task("q") == "found it"


# ---------------------------------------------------------------------------
# HTTP error status codes
# ---------------------------------------------------------------------------

@respx.mock
def test_http_401_raises():
    respx.post(BASE_URL).mock(return_value=httpx.Response(401))
    client = A2AClient(BASE_URL)
    with pytest.raises(httpx.HTTPStatusError):
        client.send_task("secret")


@respx.mock
def test_http_429_raises():
    respx.post(BASE_URL).mock(return_value=httpx.Response(429))
    client = A2AClient(BASE_URL)
    with pytest.raises(httpx.HTTPStatusError):
        client.send_task("too many")


@respx.mock
def test_http_500_raises():
    respx.post(BASE_URL).mock(return_value=httpx.Response(500))
    client = A2AClient(BASE_URL)
    with pytest.raises(httpx.HTTPStatusError):
        client.send_task("error")


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_timeout_stored_on_client():
    client = A2AClient(BASE_URL, timeout=5.0)
    assert client.timeout == 5.0


# ---------------------------------------------------------------------------
# Unicode / emoji round-trip
# ---------------------------------------------------------------------------

@respx.mock
def test_unicode_emoji_payload_round_trip():
    payload_text = "こんにちは 🎉 ñoño"
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "artifacts": [{"parts": [{"type": "text", "text": payload_text}]}]
        },
    }
    captured = {}

    def capture(request: httpx.Request, *_):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=body)

    respx.post(BASE_URL).mock(side_effect=capture)
    client = A2AClient(BASE_URL)
    result = client.send_task(payload_text)
    assert result == payload_text
    sent_text = captured["body"]["params"]["message"]["parts"][0]["text"]
    assert sent_text == payload_text


# ---------------------------------------------------------------------------
# Auth header propagation in streaming
# ---------------------------------------------------------------------------

@respx.mock
def test_auth_header_forwarded_in_streaming():
    """Authorization header must be included in streaming requests too."""
    sse_lines = [
        'data: {"jsonrpc":"2.0","id":"x","result":{"delta":{"parts":[{"type":"text","text":"Hi"}]}}}',
        "data: [DONE]",
    ]
    sse_body = "\n".join(sse_lines) + "\n"

    captured = {}

    def capture(request: httpx.Request, *_):
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text=sse_body,
                              headers={"Content-Type": "text/event-stream"})

    respx.post(BASE_URL).mock(side_effect=capture)
    client = A2AClient(BASE_URL, auth_header="Bearer stream-token", streaming=True)
    chunks = list(client.stream_task("hello"))
    assert captured.get("auth") == "Bearer stream-token"
    assert "Hi" in chunks


# ---------------------------------------------------------------------------
# SSE stream parsing
# ---------------------------------------------------------------------------

@respx.mock
def test_sse_malformed_json_skipped():
    """Lines with invalid JSON are silently skipped; valid events still yielded."""
    sse_lines = [
        "data: not-valid-json",
        'data: {"jsonrpc":"2.0","id":"x","result":{"delta":{"parts":[{"type":"text","text":"ok"}]}}}',
        "data: [DONE]",
    ]
    sse_body = "\n".join(sse_lines) + "\n"
    respx.post(BASE_URL).mock(return_value=httpx.Response(
        200, text=sse_body, headers={"Content-Type": "text/event-stream"}
    ))
    client = A2AClient(BASE_URL, streaming=True)
    chunks = list(client.stream_task("test"))
    assert chunks == ["ok"]


@respx.mock
def test_sse_empty_lines_and_comments_skipped():
    """Blank lines and comment lines (':') are not yielded."""
    sse_lines = [
        "",
        ": keep-alive",
        'data: {"jsonrpc":"2.0","id":"x","result":{"artifact":{"parts":[{"type":"text","text":"chunk"}]}}}',
        "data: [DONE]",
    ]
    sse_body = "\n".join(sse_lines) + "\n"
    respx.post(BASE_URL).mock(return_value=httpx.Response(
        200, text=sse_body, headers={"Content-Type": "text/event-stream"}
    ))
    client = A2AClient(BASE_URL, streaming=True)
    chunks = list(client.stream_task("hi"))
    assert chunks == ["chunk"]


@respx.mock
def test_sse_connection_drop_mid_stream():
    """A network error mid-stream propagates as an exception."""
    def drop(*_):
        raise httpx.RemoteProtocolError("connection reset", request=None)

    respx.post(BASE_URL).mock(side_effect=drop)
    client = A2AClient(BASE_URL, streaming=True)
    with pytest.raises(Exception):
        list(client.stream_task("hello"))


# ---------------------------------------------------------------------------
# _extract_stream_chunk variants
# ---------------------------------------------------------------------------

def test_extract_stream_chunk_delta():
    event = {
        "result": {
            "delta": {"parts": [{"type": "text", "text": "delta-chunk"}]}
        }
    }
    assert A2AClient._extract_stream_chunk(event) == "delta-chunk"


def test_extract_stream_chunk_artifact():
    event = {
        "result": {
            "artifact": {"parts": [{"type": "text", "text": "art-chunk"}]}
        }
    }
    assert A2AClient._extract_stream_chunk(event) == "art-chunk"


def test_extract_stream_chunk_no_text_returns_empty():
    assert A2AClient._extract_stream_chunk({}) == ""
    assert A2AClient._extract_stream_chunk({"result": {}}) == ""


def test_extract_stream_chunk_non_text_part_skipped():
    event = {
        "result": {
            "delta": {"parts": [{"type": "image", "url": "x"}]}
        }
    }
    assert A2AClient._extract_stream_chunk(event) == ""


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------

@respx.mock
def test_context_manager_closes_cleanly():
    respx.post(BASE_URL).mock(return_value=httpx.Response(200, json=TASK_OK))
    with A2AClient(BASE_URL) as client:
        result = client.send_task("ping")
    assert result == "Hello!"


# ---------------------------------------------------------------------------
# History — alternating roles + double-user edge
# ---------------------------------------------------------------------------

def test_split_messages_double_user_consecutive():
    """Two consecutive user messages: first goes to history, second is active."""
    messages = [
        AgentMessage(role=MessageRole.USER, content="First"),
        AgentMessage(role=MessageRole.USER, content="Second"),
    ]
    user_text, history = A2AChatEngine._split_messages(messages)
    assert user_text == "Second"
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "First"


def test_split_messages_alternating_full():
    messages = [
        AgentMessage(role=MessageRole.USER, content="u1"),
        AgentMessage(role=MessageRole.ASSISTANT, content="a1"),
        AgentMessage(role=MessageRole.USER, content="u2"),
        AgentMessage(role=MessageRole.ASSISTANT, content="a2"),
        AgentMessage(role=MessageRole.USER, content="u3"),
    ]
    user_text, history = A2AChatEngine._split_messages(messages)
    assert user_text == "u3"
    assert len(history) == 4
    roles = [h["role"] for h in history]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_split_messages_assistant_only():
    messages = [AgentMessage(role=MessageRole.ASSISTANT, content="only assistant")]
    user_text, history = A2AChatEngine._split_messages(messages)
    assert user_text == ""
    assert history == [{"role": "assistant", "content": "only assistant"}]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_missing_agent_url_raises_value_error():
    with pytest.raises(ValueError, match="agent_url"):
        A2AChatEngine(config={})


def test_bad_timeout_type_coerces():
    """timeout from config is coerced via float(); string '30' should work."""
    with respx.mock:
        respx.get(BASE_URL + "/.well-known/agent.json").mock(
            return_value=httpx.Response(404)
        )
        engine = A2AChatEngine(config={"agent_url": BASE_URL, "timeout": "30"})
    assert engine._client.timeout == 30.0


def test_engine_streaming_disabled_when_card_says_no():
    """Engine downgrades to blocking when card reports streaming=False."""
    card_no_stream = {
        "name": "NoStream",
        "description": "agent",
        "url": BASE_URL,
        "version": "1.0",
        "capabilities": {"streaming": False},
        "skills": [],
    }
    with respx.mock:
        respx.get(BASE_URL + "/.well-known/agent.json").mock(
            return_value=httpx.Response(200, json=card_no_stream)
        )
        engine = A2AChatEngine(config={"agent_url": BASE_URL, "streaming": True})
    # After fetching card with streaming=False, engine falls back
    assert engine._streaming is False


# ---------------------------------------------------------------------------
# Engine streaming path
# ---------------------------------------------------------------------------

@respx.mock
def test_engine_stream_tokens_via_sse():
    """Engine.stream_tokens uses SSE when streaming=True and card agrees."""
    sse_lines = [
        'data: {"jsonrpc":"2.0","id":"x","result":{"delta":{"parts":[{"type":"text","text":"tok1"}]}}}',
        'data: {"jsonrpc":"2.0","id":"x","result":{"delta":{"parts":[{"type":"text","text":"tok2"}]}}}',
        "data: [DONE]",
    ]
    sse_body = "\n".join(sse_lines) + "\n"
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_STREAMING)
    )
    respx.post(BASE_URL).mock(return_value=httpx.Response(
        200, text=sse_body, headers={"Content-Type": "text/event-stream"}
    ))
    engine = A2AChatEngine(config={"agent_url": BASE_URL, "streaming": True})
    chunks = list(engine.stream_tokens(
        [AgentMessage(role=MessageRole.USER, content="hello")]
    ))
    assert "tok1" in chunks
    assert "tok2" in chunks


@respx.mock
def test_engine_stream_tokens_no_user_message_yields_nothing():
    """stream_tokens with only system message and streaming=True yields nothing."""
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_STREAMING)
    )
    engine = A2AChatEngine(config={"agent_url": BASE_URL, "streaming": True})
    chunks = list(engine.stream_tokens(
        [AgentMessage(role=MessageRole.SYSTEM, content="sys")]
    ))
    assert chunks == []
