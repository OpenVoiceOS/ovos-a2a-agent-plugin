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

"""A2A protocol client: agent-card discovery and message exchange through a2a-sdk."""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Dict, Generator, List, Optional, TypeVar

import httpx
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.helpers.proto_helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, StreamResponse, TaskState
from a2a.utils.errors import A2AError, InvalidParamsError, TaskNotFoundError, UnsupportedOperationError
from google.protobuf.json_format import MessageToDict

from ovos_utils.log import LOG

T = TypeVar("T")

_FAILED_STATES = (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_REJECTED,
                  TaskState.TASK_STATE_CANCELED)

_OPEN_STATES = (TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED)
_TERMINAL_STATES = (TaskState.TASK_STATE_COMPLETED,) + _FAILED_STATES

# Task errors: an unknown, expired or purged task (-32001), or a task already in
# a terminal state (-32004). When the message named a task, that task is gone
# but the conversation goes on, so only the taskId is cleared (A2A 1.0.1
# section 3.4.3: a contextId without a taskId starts a new task in the same
# conversation). When the message named no task, the error can only be about
# the context, which the agent may have expired (section 3.4.1), so the session
# is dropped.
_TASK_GONE = (TaskNotFoundError, UnsupportedOperationError)
# The agent answers this when it rejects the context, or the context does not
# match the task (-32602). The same code also marks a message the agent finds
# invalid, so one such message ends the conversation: that is the price of never
# resending a context the agent refused.
_CONTEXT_GONE = (InvalidParamsError,)
# Timeouts, HTTP and network errors and internal errors are transient and keep both.

# One entry per local session that talked to the agent. Far more than the
# devices one OVOS or HiveMind node serves, and each entry is two short strings.
MAX_SESSIONS = 1024


@dataclass
class _Conversation:
    """The agent-assigned ``contextId`` of a session, and its task waiting for input."""
    context_id: str
    task_id: str = ""


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
    Parsed A2A agent card (``/.well-known/agent-card.json``).

    The service URL comes from the first entry of ``supportedInterfaces``
    (A2A 1.0), or from the top-level ``url`` that A2A 0.3 cards carry.
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
        interfaces = d.get("supportedInterfaces") or []
        url = interfaces[0].get("url", "") if interfaces else d.get("url", "")
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            url=url,
            version=d.get("version", "1.0"),
            skills=[AgentSkill.from_dict(s) for s in d.get("skills", [])],
            streaming=d.get("capabilities", {}).get("streaming", False),
            raw=d,
        )


def response_text(event: StreamResponse) -> str:
    """
    Return the text an A2A response event carries.

    The text parts of one message or one artifact are joined without a
    separator, because a streamed artifact arrives as appended chunks.
    Separate artifacts of a task are joined with a newline. A task with no
    artifact text answers with its status message, for example the question
    of a task in TASK_STATE_INPUT_REQUIRED.
    """
    if event.HasField("message"):
        return get_message_text(event.message, delimiter="")
    if event.HasField("artifact_update"):
        return get_artifact_text(event.artifact_update.artifact, delimiter="")
    if event.HasField("status_update") and event.status_update.status.HasField("message"):
        return get_message_text(event.status_update.status.message, delimiter="")
    if event.HasField("task"):
        texts = [t for t in (get_artifact_text(a, delimiter="") for a in event.task.artifacts) if t]
        if texts:
            return "\n".join(texts)
        if event.task.status.HasField("message"):
            return get_message_text(event.task.status.message, delimiter="")
    return ""


def _shutdown(loop: asyncio.AbstractEventLoop, thread: threading.Thread,
              resources: Dict[str, Any]) -> None:
    """Close the HTTP client, stop the loop and join its thread."""
    if loop.is_closed():
        return
    if threading.current_thread() is thread:
        loop.call_soon(loop.stop)
        return
    http = resources.get("http")
    if http is not None and thread.is_alive():
        asyncio.run_coroutine_threadsafe(http.aclose(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


class A2AClient:
    """
    Blocking A2A client on top of the a2a-sdk asynchronous client.

    The a2a-sdk client resolves the agent card, selects the transport the card
    advertises (A2A 1.0, or 0.3 for older servers) and sends each request to the
    URL in the card. This class runs it on a private event loop thread, so OVOS
    and HiveMind code can call it from any thread. The thread stops on
    :meth:`close`, or when the client is garbage collected.

    The client never invents an A2A ``contextId``. The first message of a
    session goes without one, the agent assigns it, and the client sends that
    value with every later message of the same session. When a task stops in
    TASK_STATE_INPUT_REQUIRED or TASK_STATE_AUTH_REQUIRED, the next message of
    the session also carries that ``taskId``, until the task reaches a terminal
    state. The map keeps the ``MAX_SESSIONS`` most recently used sessions.
    When the agent answers that the waiting task is gone, the session keeps
    its ``contextId`` and its next message starts a new task in the same
    conversation. When the agent rejects the context, with
    ``InvalidParamsError`` or with a task error to a message that named no
    task, the session is forgotten and its next message starts a new
    conversation.

    After an A2A error, from the transport or from the agent, the negotiated
    transport is dropped, so the next call reads the agent card again. An agent
    that changes its protocol version or URL is then reached without a restart.

    Args:
        base_url: Root URL of the A2A agent, where ``/.well-known/agent-card.json`` lives.
        auth_header: Optional ``Authorization`` header value (e.g. ``Bearer <token>``).
        timeout: HTTP timeout in seconds (default 60).
        streaming: Use ``SendStreamingMessage`` when the agent card allows it.
    """

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
        self._headers: Dict[str, str] = {"Authorization": auth_header} if auth_header else {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name="a2a-client-loop")
        self._thread.start()
        self._resources: Dict[str, Any] = {}
        self._clients: Dict[bool, Any] = {}
        # session id -> conversation, touched on the loop thread only
        self._sessions: "OrderedDict[str, _Conversation]" = OrderedDict()
        self._finalizer = weakref.finalize(self, _shutdown, self._loop, self._thread, self._resources)

    def _run(self, coro: Awaitable[T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _http_client(self) -> httpx.AsyncClient:
        if "http" not in self._resources:
            self._resources["http"] = httpx.AsyncClient(headers=self._headers, timeout=self.timeout)
        return self._resources["http"]
    async def _client(self, streaming: bool):
        if streaming not in self._clients:
            config = ClientConfig(httpx_client=self._http_client(), streaming=streaming)
            self._clients[streaming] = await ClientFactory(config).create_from_url(self.base_url)
        return self._clients[streaming]

    # ------------------------------------------------------------------
    # Agent-card discovery
    # ------------------------------------------------------------------

    def fetch_agent_card(self) -> AgentCard:
        """
        Fetch and parse the agent card.

        Raises:
            a2a.client.errors.A2AClientError: On network or HTTP failure, or an invalid card.
        """
        async def fetch():
            return await A2ACardResolver(self._http_client(), self.base_url).get_agent_card()

        card = AgentCard.from_dict(MessageToDict(self._run(fetch())))
        LOG.debug(f"A2AClient: discovered agent '{card.name}' at {card.url}")
        return card

    # ------------------------------------------------------------------
    # Message exchange
    # ------------------------------------------------------------------

    def _request(self, message_text: str, session_id: Optional[str]) -> SendMessageRequest:
        conversation = self._sessions.get(session_id) if session_id else None
        if conversation:
            self._sessions.move_to_end(session_id)
        message = new_text_message(message_text, role=Role.ROLE_USER,
                                   context_id=conversation.context_id if conversation else None,
                                   task_id=(conversation.task_id or None) if conversation else None)
        return SendMessageRequest(message=message)

    def _remember(self, session_id: Optional[str], event: StreamResponse) -> None:
        kind = event.WhichOneof("payload")
        if not session_id or not kind:
            return
        payload = getattr(event, kind)
        if not payload.context_id:
            return
        conversation = self._sessions.get(session_id)
        if conversation is None or conversation.context_id != payload.context_id:
            conversation = self._sessions[session_id] = _Conversation(payload.context_id)
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > MAX_SESSIONS:
            self._sessions.popitem(last=False)
        if kind == "task":
            task_id, state = payload.id, payload.status.state
        elif kind == "status_update":
            task_id, state = payload.task_id, payload.status.state
        else:
            return
        if state in _OPEN_STATES:
            conversation.task_id = task_id
        elif state in _TERMINAL_STATES:
            conversation.task_id = ""

    @staticmethod
    def _raise_on_failure(event: StreamResponse) -> None:
        status = None
        if event.HasField("task"):
            status = event.task.status
        elif event.HasField("status_update"):
            status = event.status_update.status
        if status is not None and status.state in _FAILED_STATES:
            detail = get_message_text(status.message) if status.HasField("message") else ""
            raise RuntimeError(f"A2A task ended in {TaskState.Name(status.state)}: {detail}")

    async def _events(self, message_text: str, session_id: Optional[str],
                      streaming: bool) -> AsyncIterator[StreamResponse]:
        client = await self._client(streaming)
        try:
            async for event in client.send_message(self._request(message_text, session_id)):
                self._remember(session_id, event)
                self._raise_on_failure(event)
                yield event
        except A2AError as error:
            self._clients.pop(streaming, None)
            if session_id and isinstance(error, _CONTEXT_GONE):
                self._sessions.pop(session_id, None)
            elif session_id and isinstance(error, _TASK_GONE) and session_id in self._sessions:
                if self._sessions[session_id].task_id:
                    self._sessions[session_id].task_id = ""
                else:
                    self._sessions.pop(session_id)
            raise

    def send_task(self, message_text: str, session_id: Optional[str] = None) -> str:
        """
        Send one user message with ``SendMessage`` and return the agent's answer text.

        Args:
            message_text: The user's message text.
            session_id: Local conversation id. Its messages share the agent's ``contextId``.

        Raises:
            a2a.utils.errors.A2AError: On network, HTTP or JSON-RPC failure.
                ``a2a.client.errors.A2AClientError`` is the transport subclass.
            RuntimeError: When the remote task fails, is rejected or is cancelled.
        """
        async def send() -> str:
            text = ""
            async for event in self._events(message_text, session_id, streaming=False):
                text = response_text(event) or text
            return text

        text = self._run(send())
        if not text:
            LOG.warning("A2AClient: the agent answered without any text part")
        return text

    def stream_task(self, message_text: str,
                    session_id: Optional[str] = None) -> Generator[str, None, None]:
        """
        Send one user message with ``SendStreamingMessage`` and yield text as it arrives.

        Text comes from agent messages, artifact updates and status-update
        messages. An artifact update that replaces its artifact yields only
        the text added to it, and distinct artifacts are separated by a
        newline. A replacement that does not extend the earlier text of its
        artifact yields nothing at once: the replaced text is final, and it is
        yielded in full, in artifact order, after the stream ends. A message
        of a working or completed task is yielded only when the stream carried
        no other text, as is the task snapshot.

        Raises:
            a2a.utils.errors.A2AError: On network, HTTP or JSON-RPC failure.
                ``a2a.client.errors.A2AClientError`` is the transport subclass.
            RuntimeError: When the remote task fails, is rejected or is cancelled.
        """
        events = self._events(message_text, session_id, streaming=True).__aiter__()
        yielded = False
        snapshot = progress = closing = ""
        artifacts: Dict[str, str] = {}
        rewritten: List[str] = []
        last_artifact = ""
        while True:
            try:
                event = self._run(events.__anext__())
            except StopAsyncIteration:
                break
            text = response_text(event)
            if event.HasField("task"):
                snapshot = text or snapshot
                continue
            if event.HasField("artifact_update"):
                update = event.artifact_update
                artifact_id = update.artifact.artifact_id
                seen = artifacts.get(artifact_id)
                if update.append or seen is None:
                    artifacts[artifact_id] = (seen or "") + text
                else:
                    artifacts[artifact_id] = text
                    if text.startswith(seen):
                        text = text[len(seen):]
                    elif text.startswith(seen.rstrip()):
                        text = text[len(seen.rstrip()):].lstrip()
                    elif artifact_id not in rewritten:
                        rewritten.append(artifact_id)
                if artifact_id in rewritten:
                    text = ""
                if text and last_artifact and artifact_id != last_artifact:
                    text = "\n" + text
                if text:
                    last_artifact = artifact_id
            elif event.HasField("status_update"):
                state = event.status_update.status.state
                if state == TaskState.TASK_STATE_WORKING:
                    progress = text or progress
                    continue
                if state == TaskState.TASK_STATE_COMPLETED:
                    closing = text or closing
                    continue
            if text:
                yielded = True
                yield text
        for artifact_id in [a for a in artifacts if a in rewritten]:
            yielded = True
            yield "\n" + artifacts[artifact_id]
        if not yielded and (closing or snapshot or progress):
            yield closing or snapshot or progress

    def close(self) -> None:
        """Close the HTTP connections and stop the event loop thread."""
        self._finalizer()

    def __enter__(self) -> "A2AClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
