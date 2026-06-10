"""Tests for A2AClient request/response mapping using a mocked server."""

import json
import pytest
import httpx
import respx

from ovos_a2a_solver.client import A2AClient, AgentCard


BASE_URL = "https://agent.test"

CARD_RESPONSE = {
    "name": "MockAgent",
    "description": "Mocked",
    "url": BASE_URL,
    "version": "1.0",
    "capabilities": {"streaming": False},
    "skills": [],
}

# Typical tasks/send response
TASK_RESPONSE = {
    "jsonrpc": "2.0",
    "id": "ignored",
    "result": {
        "id": "task-1",
        "status": {"state": "completed"},
        "artifacts": [
            {
                "parts": [{"type": "text", "text": "Hello from the agent!"}]
            }
        ],
    },
}

# tasks/send response with message fallback (no artifacts)
TASK_RESPONSE_MSG_FALLBACK = {
    "jsonrpc": "2.0",
    "id": "ignored",
    "result": {
        "id": "task-2",
        "status": {"state": "completed"},
        "message": {
            "role": "agent",
            "parts": [{"type": "text", "text": "Fallback answer"}],
        },
    },
}

TASK_ERROR_RESPONSE = {
    "jsonrpc": "2.0",
    "id": "ignored",
    "error": {"code": -32600, "message": "Invalid Request"},
}


@respx.mock
def test_fetch_agent_card():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    client = A2AClient(BASE_URL)
    card = client.fetch_agent_card()
    assert isinstance(card, AgentCard)
    assert card.name == "MockAgent"
    assert card.streaming is False


@respx.mock
def test_send_task_artifact_path():
    respx.get(BASE_URL + "/.well-known/agent.json").mock(
        return_value=httpx.Response(200, json=CARD_RESPONSE)
    )
    respx.post(BASE_URL).mock(
        return_value=httpx.Response(200, json=TASK_RESPONSE)
    )
    client = A2AClient(BASE_URL)
    result = client.send_task("What time is it?")
    assert result == "Hello from the agent!"


@respx.mock
def test_send_task_message_fallback():
    respx.post(BASE_URL).mock(
        return_value=httpx.Response(200, json=TASK_RESPONSE_MSG_FALLBACK)
    )
    client = A2AClient(BASE_URL)
    result = client.send_task("Question")
    assert result == "Fallback answer"


@respx.mock
def test_send_task_rpc_error_raises():
    respx.post(BASE_URL).mock(
        return_value=httpx.Response(200, json=TASK_ERROR_RESPONSE)
    )
    client = A2AClient(BASE_URL)
    with pytest.raises(RuntimeError, match="Invalid Request"):
        client.send_task("Broken")


@respx.mock
def test_send_task_http_error_raises():
    respx.post(BASE_URL).mock(return_value=httpx.Response(500))
    client = A2AClient(BASE_URL)
    with pytest.raises(httpx.HTTPStatusError):
        client.send_task("Oops")


@respx.mock
def test_auth_header_forwarded():
    captured = {}

    def capture(request: httpx.Request, *_):
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=TASK_RESPONSE)

    respx.post(BASE_URL).mock(side_effect=capture)
    client = A2AClient(BASE_URL, auth_header="Bearer secret-token")
    client.send_task("Hello")
    assert captured.get("auth") == "Bearer secret-token"


def test_extract_text_empty_result():
    result = A2AClient._extract_text(
        {"jsonrpc": "2.0", "id": "x", "result": {}}, "x"
    )
    assert result == ""
