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
"""Full-pipeline persona end-to-end test using ovoscope.

Validates that the complete OVOS intent pipeline can route utterances
through ovos-persona-pipeline-plugin into A2AChatEngine and produce a
``speak`` bus message — without any network access or running A2A server.

Stubbing strategy
-----------------
Two boundaries are monkeypatched before the pipeline thread starts:

1. ``A2AClient.fetch_agent_card`` — called in ``A2AChatEngine.__init__``
   via ``_fetch_card()``.  Returns a dummy :class:`AgentCard` so the engine
   constructs without touching a real URL.

2. ``A2AClient.send_task`` — the single method that performs the outbound
   HTTP request in the blocking (non-streaming) path.  Returns the fixed
   string ``"stubbed a2a reply"`` so the engine produces a deterministic
   response through the full persona pipeline.

These stubs are applied at *module level* (before any test fixture runs) so
the pipeline worker thread always sees the patched versions.  This validates
the integration path — OVOS session → persona pipeline plugin → ChatEngine
→ ``speak`` bus message — without coupling the test to any remote service.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

ovoscope = pytest.importorskip("ovoscope")

from ovoscope import (  # noqa: E402
    PERSONA_PIPELINE,
    CaptureSession,
    get_minicroft,
    is_pipeline_available,
)

if not is_pipeline_available(PERSONA_PIPELINE):
    pytest.skip("ovos-persona-pipeline-plugin not installed", allow_module_level=True)

# ---------------------------------------------------------------------------
# Stub the A2A network boundary BEFORE the pipeline is started.
# ---------------------------------------------------------------------------

from ovos_a2a_solver.client import A2AClient, AgentCard  # noqa: E402

_STUB_REPLY = "stubbed a2a reply"
_STUB_CARD = AgentCard(
    name="StubAgent",
    description="Deterministic stub for CI",
    url="http://stub.invalid",
    version="0.0",
    skills=[],
    streaming=False,
)

# Save originals so the patch can be undone after the module fixture tears down.
_REAL_FETCH_AGENT_CARD = A2AClient.fetch_agent_card
_REAL_SEND_TASK = A2AClient.send_task


def _stub_fetch_agent_card(self: A2AClient) -> AgentCard:
    """Return a static AgentCard; no HTTP request is made."""
    return _STUB_CARD


def _stub_send_task(
    self: A2AClient,
    message_text: str,
    session_id=None,
    history=None,
) -> str:
    """Return a fixed reply; no HTTP request is made."""
    return _STUB_REPLY

# Disable per-test timeout for the whole module: in a dev environment with many
# heavy pipeline plugins (m2v, markov, …) the IntentService bootstrap can take
# several minutes.  In a clean CI environment (only test-extras installed) it
# completes in < 30 s.  The global timeout in pyproject.toml (600 s) still
# applies as a safety net.
pytestmark = pytest.mark.timeout(0)

# ---------------------------------------------------------------------------
# Import bus helpers after stubs are installed.
# ---------------------------------------------------------------------------
from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session, SessionManager  # noqa: E402

# ---------------------------------------------------------------------------
# Persona fixture files
# ---------------------------------------------------------------------------

PERSONA_NAME = "A2ABot"

_PERSONAS_DIR = tempfile.mkdtemp()
_PERSONA_JSON = {
    "name": PERSONA_NAME,
    "handlers": ["ovos-a2a-solver"],
    "ovos-a2a-solver": {
        # Minimal engine_config — agent_url is required by A2AChatEngine.__init__
        # but fetch_agent_card and send_task are both stubbed above so no real
        # HTTP connection is attempted.
        "agent_url": "http://stub.invalid",
    },
}
with open(os.path.join(_PERSONAS_DIR, f"{PERSONA_NAME}.json"), "w") as _fh:
    json.dump(_PERSONA_JSON, _fh)

_PIPELINE_CONFIG = {
    "persona": {
        "personas_path": _PERSONAS_DIR,
        "default_persona": PERSONA_NAME,
        "short-term-memory": True,
        "handle_fallback": True,
        "ignore_plugin_personas": True,
    }
}

_TEST_PIPELINE = [
    "ovos-persona-pipeline-plugin-high",
    "ovos-persona-pipeline-plugin-low",
]

# ---------------------------------------------------------------------------
# Shared MiniCroft fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mc():
    # Apply stubs at class level so every instance the pipeline worker creates
    # sees them automatically.  Restore originals on teardown so other test
    # modules (collected in the same pytest session) are not affected.
    A2AClient.fetch_agent_card = _stub_fetch_agent_card  # type: ignore[method-assign]
    A2AClient.send_task = _stub_send_task  # type: ignore[method-assign]
    try:
        croft = get_minicroft(
            skill_ids=[],
            default_pipeline=_TEST_PIPELINE,
            pipeline_config=_PIPELINE_CONFIG,
            max_wait=480,  # IntentService loads all installed pipeline plugins (~90 s on this machine)
        )
        yield croft
        croft.stop()
    finally:
        A2AClient.fetch_agent_card = _REAL_FETCH_AGENT_CARD  # type: ignore[method-assign]
        A2AClient.send_task = _REAL_SEND_TASK  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utterance_msg(utterance: str, sess: Session) -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": sess.lang},
        {"session": sess.serialize()},
    )


def _drive_utterance(croft, sess: Session, utterance: str, timeout: int = 30):
    cap = CaptureSession(
        croft,
        eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"],
    )
    cap.capture(_utterance_msg(utterance, sess), timeout=timeout)
    return cap.finish()


def _get_persona_service(croft):
    return croft.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]


# ---------------------------------------------------------------------------
# Test 1: utterance flows through full pipeline → speak
# ---------------------------------------------------------------------------


class TestA2APersonaSpeaksThroughPipeline:
    """An utterance must traverse the full OVOS pipeline, reach the A2A
    persona engine (with its network call stubbed), and produce a non-empty
    ``speak`` message.  This validates the integration path without any
    real A2A server or API key.
    """

    def test_pipeline_produces_speak(self, mc):
        sess = Session(session_id="a2a-e2e-speak-1")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "hello a2a bot", timeout=30)
        msg_types = [m.msg_type for m in messages]
        speak_msgs = [m for m in messages if m.msg_type == "speak"]

        assert speak_msgs, (
            f"Expected at least one 'speak' message; got msg_types: {msg_types}"
        )
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), (
            f"'speak' message had an empty utterance; data={speak_msgs[0].data}"
        )

    def test_speak_contains_stubbed_reply(self, mc):
        """The spoken text must originate from the stubbed A2A engine reply."""
        sess = Session(session_id="a2a-e2e-speak-2")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "what can you do", timeout=30)
        speak_msgs = [m for m in messages if m.msg_type == "speak"]

        assert speak_msgs, "No speak message produced"
        spoken = speak_msgs[0].data.get("utterance", "")
        assert _STUB_REPLY in spoken, (
            f"Expected stub reply {_STUB_REPLY!r} in spoken text {spoken!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: per-session memory is recorded
# ---------------------------------------------------------------------------


class TestA2APersonaMemory:
    """PersonaService accumulates USER + ASSISTANT turns per session.

    The live PersonaService is obtained from the MiniCroft pipeline registry.
    Drives utterances through the real pipeline; the A2A network boundary is
    stubbed so the ASSISTANT turn is deterministic.
    """

    def test_user_turn_recorded_in_memory(self, mc):
        svc = _get_persona_service(mc)
        sess = Session(session_id="a2a-e2e-mem-user")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None, f"Persona '{PERSONA_NAME}' not loaded"
        assert persona.memory is not None, "Persona must have short-term memory enabled"

        _drive_utterance(mc, sess, "remember this utterance please", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        contents = [m.content for m in history]
        assert any("remember this utterance please" in c for c in contents), (
            f"User utterance not found in memory for session {sess.session_id}. "
            f"History: {contents}"
        )

    def test_assistant_response_recorded_in_memory(self, mc):
        svc = _get_persona_service(mc)
        sess = Session(session_id="a2a-e2e-mem-assistant")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        _drive_utterance(mc, sess, "say something back please", timeout=30)

        from ovos_plugin_manager.templates.agents import MessageRole

        history = persona.memory.get_history(sess.session_id)
        roles = [m.role for m in history]
        assert MessageRole.ASSISTANT in roles, (
            f"No ASSISTANT turn recorded in memory. History roles: {roles}"
        )

    def test_unknown_session_has_empty_history(self, mc):
        svc = _get_persona_service(mc)
        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        # ensure at least one known session exists
        sess = Session(session_id="a2a-e2e-mem-known")
        SessionManager.sessions[sess.session_id] = sess
        _drive_utterance(mc, sess, "hello known session", timeout=30)

        unknown_history = persona.memory.get_history("session-that-never-existed-a2a")
        assert unknown_history == [], (
            f"Expected empty history for unknown session, got: {unknown_history}"
        )

    def test_same_session_accumulates_turns(self, mc):
        """Two utterances on the same session must yield at least 2 history entries."""
        svc = _get_persona_service(mc)
        sess = Session(session_id="a2a-e2e-mem-accumulate")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None
        # clear any prior history for isolation
        persona.memory.session2history.pop(sess.session_id, None)

        _drive_utterance(mc, sess, "first a2a question", timeout=30)
        _drive_utterance(mc, sess, "second a2a question", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        assert len(history) >= 2, (
            f"Expected at least 2 history entries after two turns, got {len(history)}: "
            f"{[m.content for m in history]}"
        )
