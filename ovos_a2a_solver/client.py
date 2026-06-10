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

"""A2A protocol client — agent-card discovery and task submission."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urljoin

import httpx

from ovos_utils.log import LOG


@dataclass
class AgentSkill:
    """A capability advertised in an agent card."""
    id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentSkill":
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            examples=d.get("examples", []),
        )


@dataclass
class AgentCard:
    """
    Parsed representation of an A2A agent card (``/.well-known/agent.json``).

    The A2A agent-card is the discovery document that every compliant A2A server
    exposes.  It describes the agent's identity, supported capabilities, and the
    URL at which tasks are accepted.
    """
    name: str
    description: str
    url: str
    version: str = "1.0"
    skills: List[AgentSkill] = field(default_factory=list)
    streaming: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentCard":
        skills = [AgentSkill.from_dict(s) for s in d.get("skills", [])]
        caps = d.get("capabilities", {})
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            url=d.get("url", ""),
            version=d.get("version", "1.0"),
            skills=skills,
            streaming=caps.get("streaming", False),
            raw=d,
        )


class A2AClient:
    """
    Minimal A2A JSON-RPC 2.0 client.

    Handles:
    - Agent-card discovery (``GET /.well-known/agent.json``)
    - Task submission via ``tasks/send`` (blocking)
    - Streaming via ``tasks/sendSubscribe`` (SSE)

    Args:
        base_url: Root URL of the A2A server (e.g. ``https://agent.example.com``).
        auth_header: Optional ``Authorization`` header value (e.g. ``Bearer <token>``).
        timeout: HTTP timeout in seconds (default 60).
        streaming: Prefer SSE streaming when True.
    """

    AGENT_CARD_PATH = "/.well-known/agent.json"
    JSONRPC_VERSION = "2.0"

    def __init__(
        self,
        base_url: str,
        auth_header: Optional[str] = None,
        timeout: float = 60.0,
        streaming: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.streaming = streaming
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header
        self._http = httpx.Client(headers=headers, timeout=timeout)

    # ------------------------------------------------------------------
    # Agent-card discovery
    # ------------------------------------------------------------------

    def fetch_agent_card(self) -> AgentCard:
        """
        Fetch and parse the agent card from ``/.well-known/agent.json``.

        Returns:
            Parsed :class:`AgentCard`.

        Raises:
            httpx.HTTPError: On network/HTTP failure.
            ValueError: If the response body is not valid JSON.
        """
        url = self.base_url + self.AGENT_CARD_PATH
        LOG.debug(f"A2AClient: fetching agent card from {url}")
        resp = self._http.get(url)
        resp.raise_for_status()
        data = resp.json()
        card = AgentCard.from_dict(data)
        LOG.debug(f"A2AClient: discovered agent '{card.name}' at {card.url}")
        return card

    # ------------------------------------------------------------------
    # Task send (blocking)
    # ------------------------------------------------------------------

    def send_task(
        self,
        message_text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Submit a task to the agent and return the final text response.

        Constructs a ``tasks/send`` JSON-RPC 2.0 request, sends it, and
        extracts the assistant-role ``text`` part from the response.

        Args:
            message_text: The user's message text.
            session_id: Conversation context identifier (optional).
            history: Prior turns as ``[{"role": ..., "content": ...}]`` (optional).

        Returns:
            The agent's response text.

        Raises:
            httpx.HTTPError: On network failure.
            RuntimeError: If the RPC response contains an error object.
        """
        rpc_id = str(uuid.uuid4())
        parts: List[Dict[str, Any]] = [{"type": "text", "text": message_text}]
        msg: Dict[str, Any] = {"role": "user", "parts": parts}
        params: Dict[str, Any] = {
            "id": rpc_id,
            "message": msg,
        }
        if session_id:
            params["sessionId"] = session_id
        if history:
            params["history"] = [
                {"role": t["role"], "parts": [{"type": "text", "text": t["content"]}]}
                for t in history
            ]

        payload = {
            "jsonrpc": self.JSONRPC_VERSION,
            "id": rpc_id,
            "method": "tasks/send",
            "params": params,
        }

        url = urljoin(self.base_url + "/", "")
        # A2A servers accept JSON-RPC at their root or at a /tasks path.
        # Try root first; callers can override by subclassing.
        resp = self._http.post(self.base_url, content=json.dumps(payload))
        resp.raise_for_status()
        body = resp.json()
        return self._extract_text(body, rpc_id)

    # ------------------------------------------------------------------
    # Streaming (SSE) send
    # ------------------------------------------------------------------

    def stream_task(
        self,
        message_text: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """
        Submit a task and yield text chunks as they arrive (SSE streaming).

        Uses ``tasks/sendSubscribe``.  Each SSE ``data:`` line is expected to be
        a JSON-RPC response fragment with an ``artifact`` or ``delta`` payload.

        Args:
            message_text: The user's message text.
            session_id: Conversation context identifier (optional).
            history: Prior turns (optional).

        Yields:
            Text chunks from the streaming response.
        """
        rpc_id = str(uuid.uuid4())
        parts: List[Dict[str, Any]] = [{"type": "text", "text": message_text}]
        msg: Dict[str, Any] = {"role": "user", "parts": parts}
        params: Dict[str, Any] = {
            "id": rpc_id,
            "message": msg,
        }
        if session_id:
            params["sessionId"] = session_id
        if history:
            params["history"] = [
                {"role": t["role"], "parts": [{"type": "text", "text": t["content"]}]}
                for t in history
            ]

        payload = {
            "jsonrpc": self.JSONRPC_VERSION,
            "id": rpc_id,
            "method": "tasks/sendSubscribe",
            "params": params,
        }

        with self._http.stream("POST", self.base_url,
                               content=json.dumps(payload),
                               headers={"Accept": "text/event-stream"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                chunk = self._extract_stream_chunk(event)
                if chunk:
                    yield chunk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(body: Dict[str, Any], rpc_id: str) -> str:
        """Extract the assistant text from a ``tasks/send`` response."""
        if "error" in body:
            err = body["error"]
            raise RuntimeError(
                f"A2A RPC error {err.get('code')}: {err.get('message')}"
            )
        result = body.get("result", {})
        # Navigate: result → artifacts[] → parts[] → text
        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    return part["text"]
        # Fallback: some servers put the answer directly in result.message
        msg = result.get("message", {})
        for part in msg.get("parts", []):
            if part.get("type") == "text":
                return part["text"]
        LOG.warning(f"A2AClient: could not extract text from response: {body}")
        return ""

    @staticmethod
    def _extract_stream_chunk(event: Dict[str, Any]) -> str:
        """Extract incremental text from a streaming SSE event."""
        result = event.get("result", {})
        # streaming events carry delta or artifact
        for key in ("delta", "artifact"):
            artifact = result.get(key, {})
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    return part["text"]
        return ""

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> "A2AClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
