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

"""A2AChatEngine — OPM ChatEngine plugin that delegates to an A2A agent."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole, ToolsArg
from ovos_utils.log import LOG

from ovos_a2a_solver.client import A2AClient, AgentCard


class A2AChatEngine(ChatEngine):
    """
    ``ChatEngine`` plugin that proxies all conversations to an external
    `Agent2Agent (A2A) <https://google.github.io/A2A/>`_ agent.

    The plugin discovers the remote agent via its agent card
    (``/.well-known/agent.json``), converts the OVOS message history to the
    A2A task format, and returns the agent's response as an
    :class:`~ovos_plugin_manager.templates.agents.AgentMessage`.

    **Entry point group:** ``opm.agents.chat``
    **Entry point id:** ``ovos-a2a-solver``

    Configuration keys (``persona["engine_config"]``):

    .. code-block:: yaml

        engine: ovos-a2a-solver
        engine_config:
          agent_url: "https://my-a2a-agent.example.com"
          auth_header: "Bearer <token>"    # optional
          timeout: 60                      # seconds, default 60
          streaming: false                 # set true to use SSE streaming

    Multiple agents can be chained by stacking personas; each plugin instance
    targets one ``agent_url``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        agent_url: str = self.config.get("agent_url", "")
        if not agent_url:
            raise ValueError(
                "A2AChatEngine: 'agent_url' is required in engine_config"
            )
        auth_header: Optional[str] = self.config.get("auth_header")
        timeout: float = float(self.config.get("timeout", 60))
        streaming: bool = bool(self.config.get("streaming", False))

        self._client = A2AClient(
            base_url=agent_url,
            auth_header=auth_header,
            timeout=timeout,
            streaming=streaming,
        )
        self._card: Optional[AgentCard] = None
        self._streaming = streaming
        self._fetch_card()

    # ------------------------------------------------------------------
    # Agent-card bootstrap
    # ------------------------------------------------------------------

    def _fetch_card(self) -> None:
        try:
            self._card = self._client.fetch_agent_card()
            LOG.info(
                f"A2AChatEngine: connected to A2A agent '{self._card.name}' "
                f"(streaming={self._card.streaming})"
            )
            # Honour server-side streaming capability; local config can disable it.
            if self._streaming and not self._card.streaming:
                LOG.warning(
                    "A2AChatEngine: streaming requested but agent card reports "
                    "streaming=False; falling back to blocking send"
                )
                self._streaming = False
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                f"A2AChatEngine: could not fetch agent card: {exc}; "
                "continuing without it — tasks will still be attempted"
            )

    # ------------------------------------------------------------------
    # ChatEngine contract
    # ------------------------------------------------------------------

    def continue_chat(
        self,
        messages: List[AgentMessage],
        session_id: str = "default",
        lang: Optional[str] = None,
        units: Optional[str] = None,
        tools: ToolsArg = None,
    ) -> AgentMessage:
        """
        Send the conversation to the A2A agent and return the response.

        The last ``user`` message becomes the A2A task message; prior turns are
        forwarded as ``history``.

        Args:
            messages: Full conversation history (system + user + assistant turns).
            session_id: Session identifier forwarded to the A2A server.
            lang: Ignored (A2A agents handle language internally).
            units: Ignored.
            tools: Ignored. Tool selection lives on the remote A2A agent's side
                of the protocol; there is no seam here for injecting external
                tool schemas. ``supports_tools`` stays False so the agentic
                loop falls back to its text-based ReAct path.

        Returns:
            :class:`AgentMessage` with ``role=ASSISTANT`` containing the
            agent's response.
        """
        user_text, history = self._split_messages(messages)
        if not user_text:
            LOG.warning("A2AChatEngine.continue_chat: no user message found")
            return AgentMessage(role=MessageRole.ASSISTANT, content="")

        try:
            text = self._client.send_task(
                message_text=user_text,
                session_id=session_id,
                history=history,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error(f"A2AChatEngine: task failed: {exc}")
            raise

        return AgentMessage(role=MessageRole.ASSISTANT, content=text)

    def stream_tokens(
        self,
        messages: List[AgentMessage],
        session_id: str = "default",
        lang: Optional[str] = None,
        units: Optional[str] = None,
    ) -> Iterable[str]:
        """
        Stream response tokens from the A2A agent.

        Falls back to :meth:`continue_chat` when streaming is disabled.

        Args:
            messages: Full conversation history.
            session_id: Session identifier.
            lang: Ignored.
            units: Ignored.

        Yields:
            Text chunks as they arrive.
        """
        if not self._streaming:
            reply = self.continue_chat(messages, session_id, lang, units)
            yield reply.content
            return

        user_text, history = self._split_messages(messages)
        if not user_text:
            return

        try:
            yield from self._client.stream_task(
                message_text=user_text,
                session_id=session_id,
                history=history,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error(f"A2AChatEngine: streaming task failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_messages(
        messages: List[AgentMessage],
    ) -> tuple[str, List[Dict[str, str]]]:
        """
        Extract the latest user message text and build a history list.

        System messages are prepended as a first user turn (many A2A servers do
        not have a dedicated system role).  Returns ``(user_text, history)``
        where ``history`` excludes the final user message.
        """
        history: List[Dict[str, str]] = []
        user_text = ""
        system_parts: List[str] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_parts.append(msg.content)
            elif msg.role == MessageRole.USER:
                if user_text:
                    history.append({"role": "user", "content": user_text})
                user_text = msg.content
            elif msg.role == MessageRole.ASSISTANT:
                if user_text:
                    history.append({"role": "user", "content": user_text})
                    user_text = ""
                history.append({"role": "assistant", "content": msg.content})

        if system_parts:
            system_turn = {"role": "user", "content": "\n".join(system_parts)}
            history.insert(0, system_turn)

        return user_text, history
