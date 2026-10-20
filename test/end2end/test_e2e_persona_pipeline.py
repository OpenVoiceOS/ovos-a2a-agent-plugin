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

An utterance travels through the OVOS intent pipeline, the
ovos-persona-pipeline-plugin and A2AChatEngine to a real a2a-sdk agent
(test/a2a_agent.py), and the agent's answer comes back as a ``speak`` bus
message. The agent answers ``echo[<contextId>]: <text>``. The contextId is the
one the agent assigned, never the OVOS session id.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time

import pytest

from ovoscope import (  # noqa: E402
    PERSONA_PIPELINE,
    CaptureSession,
    get_minicroft,
    is_pipeline_available,
)

assert is_pipeline_available(PERSONA_PIPELINE), (
    "ovos-persona-pipeline-plugin must be installed (ships with ovos-persona)"
)

# Disable per-test timeout for the whole module: in a dev environment with many
# heavy pipeline plugins (m2v, markov, …) the IntentService bootstrap can take
# several minutes.  In a clean CI environment (only test-extras installed) it
# completes in < 30 s.  The global timeout in pyproject.toml (600 s) still
# applies as a safety net.
pytestmark = pytest.mark.timeout(0)

# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------
from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session, SessionManager  # noqa: E402

# ---------------------------------------------------------------------------
# Persona fixture files
# ---------------------------------------------------------------------------

PERSONA_NAME = "A2ABot"
STREAM_PERSONA = "A2AStreamBot"
SLOW_PERSONA = "A2ASlowBot"
DOWN_PERSONA = "A2ADownBot"
SLOW_STREAM_PERSONA = "A2ASlowStreamBot"
DOWN_STREAM_PERSONA = "A2ADownStreamBot"

_PERSONAS_DIR = tempfile.mkdtemp()

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
def mc(a2a_agent_url):
    engines = {
        PERSONA_NAME: {"agent_url": a2a_agent_url},
        STREAM_PERSONA: {"agent_url": a2a_agent_url, "streaming": True},
        SLOW_PERSONA: {"agent_url": a2a_agent_url, "timeout": 1},
        DOWN_PERSONA: {"agent_url": "http://127.0.0.1:1", "timeout": 3},
        SLOW_STREAM_PERSONA: {"agent_url": a2a_agent_url, "timeout": 1, "streaming": True},
        DOWN_STREAM_PERSONA: {"agent_url": "http://127.0.0.1:1", "timeout": 3, "streaming": True},
    }
    for name, engine_config in engines.items():
        persona = {"name": name, "handlers": ["ovos-a2a-solver"], "ovos-a2a-solver": engine_config}
        with open(os.path.join(_PERSONAS_DIR, f"{name}.json"), "w") as fh:
            json.dump(persona, fh)
    croft = get_minicroft(
        skill_ids=[],
        default_pipeline=_TEST_PIPELINE,
        pipeline_config=_PIPELINE_CONFIG,
        max_wait=480,  # IntentService loads all installed pipeline plugins (~90 s on this machine)
    )
    yield croft
    croft.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utterance_msg(utterance: str, sess: Session) -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": sess.lang},
        {"session": sess.serialize()},
    )


def _session(session_id: str, persona: str = PERSONA_NAME) -> Session:
    """A registered session whose active persona is ``persona``."""
    sess = Session(session_id=session_id)
    sess.persona_id = persona
    SessionManager.sessions[sess.session_id] = sess
    return sess


def _speaks(messages):
    """The spoken messages, one per utterance, in whichever spelling the stack emits."""
    for msg_type in ("ovos.utterance.speak", "speak"):
        spoken = [m for m in messages if m.msg_type == msg_type]
        if spoken:
            return spoken
    return []


def _spoken_texts(messages):
    """Spoken sentences, without the space the utterance pipeline puts before punctuation."""
    return [re.sub(r"\s+([.,?!])", r"\1", m.data["utterance"]).strip() for m in _speaks(messages)]


def _context_of(spoken: str) -> str:
    return re.fullmatch(r"echo\[([^\]]+)\]: .*", spoken).group(1)


def _drive_utterance(croft, sess: Session, utterance: str, timeout: int = 30):
    cap = CaptureSession(
        croft,
        eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"],
    )
    cap.capture(_utterance_msg(utterance, sess), timeout=timeout)
    return cap.finish()


def _get_persona_service(croft):
    return croft.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]


# Legacy and namespaced spelling of the "speak" bus message. Depending on
# the resolved ovos-bus-client version, the stack may emit either (or both).
_SPEAK_MSG_TYPES = {"speak", "ovos.utterance.speak"}


# ---------------------------------------------------------------------------
# Test 1: utterance flows through full pipeline → speak
# ---------------------------------------------------------------------------


class TestA2APersonaSpeaksThroughPipeline:
    """An utterance must traverse the full OVOS pipeline, reach the A2A
    persona engine, reach the real A2A agent, and produce a ``speak``
    message with the agent's answer.
    """

    def test_pipeline_produces_speak(self, mc):
        sess = Session(session_id="a2a-e2e-speak-1")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "hello a2a bot", timeout=30)
        msg_types = [m.msg_type for m in messages]
        speak_msgs = [m for m in messages if m.msg_type in _SPEAK_MSG_TYPES]

        assert speak_msgs, (
            f"Expected at least one of {_SPEAK_MSG_TYPES} message; got msg_types: {msg_types}"
        )
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), (
            f"'speak' message had an empty utterance; data={speak_msgs[0].data}"
        )

    def test_speak_is_the_agent_answer_for_this_session(self, mc):
        """The spoken text is the agent's echo, in a context the agent assigned."""
        sess = Session(session_id="a2a-e2e-speak-2")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "what can you do", timeout=30)
        speak_msgs = [m for m in messages if m.msg_type in _SPEAK_MSG_TYPES]

        assert speak_msgs, "No speak message produced"
        spoken = speak_msgs[0].data.get("utterance", "")
        context, text = re.fullmatch(r"echo\[([^\]]+)\]: (.*)", spoken).groups()
        assert text == "what can you do"
        assert context != "a2a-e2e-speak-2"


# ---------------------------------------------------------------------------
# Test 2: per-session memory is recorded
# ---------------------------------------------------------------------------


class TestA2APersonaMemory:
    """PersonaService accumulates USER + ASSISTANT turns per session.

    The live PersonaService is obtained from the MiniCroft pipeline registry.
    Drives utterances through the real pipeline to the real A2A agent, whose
    echo answer makes the ASSISTANT turn deterministic.
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


# ---------------------------------------------------------------------------
# Test 3: the follow-up reaches the remote conversation of the first turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persona", [PERSONA_NAME, STREAM_PERSONA], ids=["blocking", "streaming"])
class TestA2AFollowUpKeepsTheRemoteConversation:
    def test_follow_up_reuses_the_context_the_agent_assigned(self, mc, persona):
        sess = _session("a2a-e2e-context-1-" + persona, persona)

        first = _speaks(_drive_utterance(mc, sess, "first question here"))
        second = _speaks(_drive_utterance(mc, sess, "second question here"))

        assert len(first) == 1 and len(second) == 1
        assert _context_of(first[0].data["utterance"]) == _context_of(second[0].data["utterance"])

    def test_two_sessions_get_two_remote_conversations(self, mc, persona):
        one = _speaks(_drive_utterance(mc, _session("a2a-e2e-context-2a-" + persona, persona), "hello from one"))
        two = _speaks(_drive_utterance(mc, _session("a2a-e2e-context-2b-" + persona, persona), "hello from two"))

        assert _context_of(one[0].data["utterance"]) != _context_of(two[0].data["utterance"])

    def test_remote_agent_remembers_across_turns(self, mc, persona):
        sess = _session("a2a-e2e-context-3-" + persona, persona)

        _drive_utterance(mc, sess, "remember purple")
        recalled = _speaks(_drive_utterance(mc, sess, "recall"))

        assert [m.data["utterance"] for m in recalled] == ["purple"]


# ---------------------------------------------------------------------------
# Test 4: a failing remote agent is spoken about once
# ---------------------------------------------------------------------------


class TestA2AFailuresAreSpokenOnce:
    def _one_error_dialog(self, messages):
        spoken = [m.data["utterance"] for m in _speaks(messages)]
        assert len(spoken) == 1, spoken
        assert spoken[0].strip() and not spoken[0].startswith("echo["), spoken
        return spoken[0]

    @pytest.mark.parametrize("persona", [PERSONA_NAME, STREAM_PERSONA], ids=["blocking", "streaming"])
    def test_failed_task(self, mc, persona):
        self._one_error_dialog(_drive_utterance(mc, _session("a2a-e2e-fail-" + persona, persona), "fail"))

    @pytest.mark.parametrize("persona", [SLOW_PERSONA, SLOW_STREAM_PERSONA], ids=["blocking", "streaming"])
    def test_agent_slower_than_the_timeout(self, mc, persona):
        sess = _session("a2a-e2e-slow-" + persona, persona)
        self._one_error_dialog(_drive_utterance(mc, sess, "slow 5 answer", timeout=30))

    @pytest.mark.parametrize("persona", [DOWN_PERSONA, DOWN_STREAM_PERSONA], ids=["blocking", "streaming"])
    def test_agent_down(self, mc, persona):
        sess = _session("a2a-e2e-down-" + persona, persona)
        self._one_error_dialog(_drive_utterance(mc, sess, "are you there", timeout=30))


# ---------------------------------------------------------------------------
# Test 5: streaming through the pipeline speaks sentence by sentence
# ---------------------------------------------------------------------------


class TestA2AStreamingSpeaksSentences:
    ANSWER = "Hello there. How are you today? I am fine, thank you"
    SENTENCES = ["Hello there.", "How are you today?", "I am fine, thank you"]

    def test_streaming_answer_is_spoken_as_separate_sentences(self, mc):
        sess = _session("a2a-e2e-stream-1", STREAM_PERSONA)

        messages = _drive_utterance(mc, sess, f"chunks {self.ANSWER}")

        assert _spoken_texts(messages) == self.SENTENCES

    def test_first_sentence_is_spoken_before_the_agent_has_finished(self, mc):
        sess = _session("a2a-e2e-stream-2", STREAM_PERSONA)
        cap = CaptureSession(mc, eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"])
        by_type = {"ovos.utterance.speak": [], "speak": []}
        for msg_type, times in by_type.items():
            mc.bus.on(msg_type, lambda m, times=times: times.append(time.monotonic()))

        start = time.monotonic()
        cap.capture(_utterance_msg(f"drip {self.ANSWER}", sess), timeout=30)
        cap.finish()
        finished = time.monotonic()
        arrivals = by_type["ovos.utterance.speak"] or by_type["speak"]

        # the agent drips 11 chunks, half a second apart: about 5.5 s in all
        assert len(arrivals) == len(self.SENTENCES)
        assert arrivals[0] - start < 3
        assert finished - start > 4.5

    def test_without_streaming_the_whole_answer_is_one_speak(self, mc):
        sess = _session("a2a-e2e-stream-3")

        messages = _drive_utterance(mc, sess, f"chunks {self.ANSWER}")

        assert _spoken_texts(messages) == [self.ANSWER]
