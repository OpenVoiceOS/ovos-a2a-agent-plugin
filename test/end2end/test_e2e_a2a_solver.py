# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end tests for ovos-a2a-solver-plugin.

Spins up a tiny FastAPI mock A2A server (agent card + tasks/send + SSE
streaming) on a free local port, then runs A2AClient and A2AChatEngine
against it to verify the full request-response cycle.

Run in isolation::

    pytest test/end2end/test_e2e_a2a_solver.py -v --timeout=30
"""
from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Mock A2A server
# ---------------------------------------------------------------------------

MOCK_AGENT_CARD: Dict[str, Any] = {
    "name": "MockAgent",
    "description": "A mock A2A agent for testing.",
    "url": "",  # filled in at fixture creation
    "version": "1.0",
    "capabilities": {"streaming": True},
    "skills": [
        {
            "id": "echo",
            "name": "Echo",
            "description": "Echoes back the user message.",
            "tags": ["echo", "test"],
            "examples": ["hello", "what time is it"],
        }
    ],
}

MOCK_RESPONSE_TEXT = "Echo: {msg}"


def _build_mock_a2a_app(base_url: str) -> FastAPI:
    """Build a minimal A2A-compliant FastAPI server."""
    app = FastAPI(title="mock-a2a")

    card = dict(MOCK_AGENT_CARD)
    card["url"] = base_url

    @app.get("/.well-known/agent.json")
    def agent_card() -> JSONResponse:
        return JSONResponse(content=card)

    @app.post("/")
    async def rpc_endpoint(request: Request) -> Any:
        body = await request.json()
        method = body.get("method", "")
        rpc_id = body.get("id", str(uuid.uuid4()))
        params = body.get("params", {})

        # Extract user message text
        msg_text = ""
        msg = params.get("message", {})
        for part in msg.get("parts", []):
            if part.get("type") == "text":
                msg_text = part["text"]
                break

        if method == "tasks/send":
            reply_text = MOCK_RESPONSE_TEXT.format(msg=msg_text)
            result = {
                "id": params.get("id", rpc_id),
                "artifacts": [
                    {
                        "parts": [{"type": "text", "text": reply_text}]
                    }
                ],
            }
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": result,
            })

        if method == "tasks/sendSubscribe":
            reply_text = MOCK_RESPONSE_TEXT.format(msg=msg_text)

            def sse_stream():
                chunk = {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "delta": {
                            "parts": [{"type": "text", "text": reply_text}]
                        }
                    },
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
            )

        return JSONResponse(
            content={"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "Method not found"}},
            status_code=200,
        )

    return app


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_a2a_server():
    """Start the mock A2A server and return its base URL."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    app = _build_mock_a2a_app(base_url)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            httpx.get(f"{base_url}/.well-known/agent.json", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.skip("Mock A2A server did not start in time")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Agent card discovery
# ---------------------------------------------------------------------------

class TestAgentCardDiscovery:
    def test_agent_card_fetch(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server) as client:
            card = client.fetch_agent_card()
        assert card.name == "MockAgent"
        assert card.url == mock_a2a_server
        assert card.streaming is True

    def test_agent_card_skills_parsed(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server) as client:
            card = client.fetch_agent_card()
        assert len(card.skills) == 1
        assert card.skills[0].id == "echo"

    def test_agent_card_version(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server) as client:
            card = client.fetch_agent_card()
        assert card.version == "1.0"


# ---------------------------------------------------------------------------
# tasks/send (blocking)
# ---------------------------------------------------------------------------

class TestTaskSend:
    def test_send_task_returns_echo(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server) as client:
            text = client.send_task("hello world")
        assert "hello world" in text

    def test_send_task_with_session_id(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server) as client:
            text = client.send_task("ping", session_id="sess-001")
        assert "ping" in text

    def test_send_task_with_history(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        with A2AClient(base_url=mock_a2a_server) as client:
            text = client.send_task("follow-up", history=history)
        assert "follow-up" in text

    def test_send_task_empty_message(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server) as client:
            text = client.send_task("")
        # Empty message echo is valid — just check it returned something
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# tasks/sendSubscribe (streaming)
# ---------------------------------------------------------------------------

class TestStreamTask:
    def test_stream_task_yields_text(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server, streaming=True) as client:
            chunks = list(client.stream_task("stream me"))
        assert chunks, "Expected at least one chunk from stream_task"
        full = "".join(chunks)
        assert "stream me" in full

    def test_stream_task_multiple_calls(self, mock_a2a_server):
        from ovos_a2a_solver.client import A2AClient
        with A2AClient(base_url=mock_a2a_server, streaming=True) as client:
            for msg in ("one", "two", "three"):
                chunks = list(client.stream_task(msg))
                assert any(msg in c for c in chunks), f"Expected {msg!r} in chunks {chunks}"


# ---------------------------------------------------------------------------
# A2AChatEngine end-to-end
# ---------------------------------------------------------------------------

class TestA2AChatEngineE2E:
    def test_engine_init_fetches_card(self, mock_a2a_server):
        from ovos_a2a_solver.engine import A2AChatEngine
        engine = A2AChatEngine(config={"agent_url": mock_a2a_server})
        assert engine._card is not None
        assert engine._card.name == "MockAgent"

    def test_continue_chat_returns_assistant_message(self, mock_a2a_server):
        from ovos_a2a_solver.engine import A2AChatEngine
        from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

        engine = A2AChatEngine(config={"agent_url": mock_a2a_server})
        messages = [
            AgentMessage(role=MessageRole.USER, content="hello engine"),
        ]
        reply = engine.continue_chat(messages, session_id="e2e-test")
        assert reply.role == MessageRole.ASSISTANT
        assert "hello engine" in reply.content

    def test_continue_chat_with_history(self, mock_a2a_server):
        from ovos_a2a_solver.engine import A2AChatEngine
        from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

        engine = A2AChatEngine(config={"agent_url": mock_a2a_server})
        messages = [
            AgentMessage(role=MessageRole.USER, content="first"),
            AgentMessage(role=MessageRole.ASSISTANT, content="second"),
            AgentMessage(role=MessageRole.USER, content="third turn"),
        ]
        reply = engine.continue_chat(messages)
        assert reply.role == MessageRole.ASSISTANT
        assert "third turn" in reply.content

    def test_stream_tokens_non_streaming_fallback(self, mock_a2a_server):
        """When streaming=False, stream_tokens falls back to continue_chat."""
        from ovos_a2a_solver.engine import A2AChatEngine
        from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

        engine = A2AChatEngine(config={"agent_url": mock_a2a_server, "streaming": False})
        messages = [AgentMessage(role=MessageRole.USER, content="fallback test")]
        tokens = list(engine.stream_tokens(messages))
        assert tokens, "Expected at least one token"
        assert "fallback test" in "".join(tokens)

    def test_stream_tokens_streaming(self, mock_a2a_server):
        """When streaming=True and agent reports streaming, use SSE path."""
        from ovos_a2a_solver.engine import A2AChatEngine
        from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

        engine = A2AChatEngine(config={"agent_url": mock_a2a_server, "streaming": True})
        messages = [AgentMessage(role=MessageRole.USER, content="stream test")]
        tokens = list(engine.stream_tokens(messages))
        assert tokens
        assert "stream test" in "".join(tokens)

    def test_no_user_message_returns_empty(self, mock_a2a_server):
        """continue_chat with only system messages returns empty content."""
        from ovos_a2a_solver.engine import A2AChatEngine
        from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

        engine = A2AChatEngine(config={"agent_url": mock_a2a_server})
        messages = [
            AgentMessage(role=MessageRole.SYSTEM, content="be helpful"),
        ]
        reply = engine.continue_chat(messages)
        assert reply.role == MessageRole.ASSISTANT
        assert reply.content == ""
