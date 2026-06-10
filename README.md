# ovos-a2a-solver-plugin

An [OVOS](https://openvoiceos.org) `ChatEngine` plugin that lets an [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) delegate its reasoning to any external agent that speaks the [Agent2Agent (A2A) protocol](https://google.github.io/A2A/).

Implements the consumer side of [OpenVoiceOS/ovos-persona#164](https://github.com/OpenVoiceOS/ovos-persona/issues/164).

---

## What is A2A?

Agent2Agent (A2A) is an open protocol that lets software agents interoperate regardless of their internal implementation. An A2A-compliant server:

1. Exposes a discovery document at `GET /.well-known/agent.json` (the *agent card*) that describes its identity, skills, and capabilities.
2. Accepts tasks at its root URL as [JSON-RPC 2.0](https://www.jsonrpc.org/specification) requests (`tasks/send` for blocking, `tasks/sendSubscribe` for SSE streaming).
3. Returns structured responses with typed *artifact parts* (text, file, data).

This plugin implements the **client (consumer) side** only — it calls remote A2A agents from within an OVOS persona.

---

## Installation

```bash
pip install ovos-a2a-solver-plugin
```

Requires Python ≥ 3.10 and `ovos-plugin-manager >= 2.3.0a1`.

---

## Configuration

Add the plugin as the engine for an ovos-persona persona:

```yaml
# ~/.config/mycroft/personas/my-a2a-persona.yaml
name: my-a2a-persona
engine: ovos-a2a-solver
engine_config:
  agent_url: "https://my-a2a-agent.example.com"   # required
  auth_header: "Bearer <token>"                    # optional
  timeout: 60                                      # seconds (default 60)
  streaming: false                                 # set true to use SSE streaming
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent_url` | `str` | — | **Required.** Root URL of the A2A server. |
| `auth_header` | `str` | `None` | Full `Authorization` header value, e.g. `Bearer <token>`. |
| `timeout` | `float` | `60` | HTTP timeout in seconds. |
| `streaming` | `bool` | `False` | Use `tasks/sendSubscribe` (SSE) instead of blocking `tasks/send`. Server must advertise `capabilities.streaming: true`. |

---

## Persona wiring example

```python
from ovos_persona import PersonaService

# PersonaService loads persona YAML files automatically.
# Once the plugin is installed, setting engine: ovos-a2a-solver in the
# persona YAML is all that is needed.
svc = PersonaService(config={})
reply = svc.chat("What is the capital of Portugal?", persona="my-a2a-persona")
print(reply)  # "Lisbon."
```

Or use the engine directly:

```python
from ovos_a2a_solver import A2AChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

engine = A2AChatEngine(config={
    "agent_url": "https://my-a2a-agent.example.com",
})

messages = [
    AgentMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    AgentMessage(role=MessageRole.USER, content="Hello!"),
]
reply = engine.continue_chat(messages)
print(reply.content)
```

For streaming:

```python
engine = A2AChatEngine(config={
    "agent_url": "https://my-a2a-agent.example.com",
    "streaming": True,
})

for chunk in engine.stream_tokens(messages):
    print(chunk, end="", flush=True)
```

---

## A2AClient — low-level usage

The `A2AClient` class can be used independently of OPM:

```python
from ovos_a2a_solver import A2AClient

with A2AClient("https://my-a2a-agent.example.com") as client:
    card = client.fetch_agent_card()
    print(card.name, card.skills)

    answer = client.send_task(
        "Summarise the A2A spec in one sentence.",
        session_id="my-session",
    )
    print(answer)
```

---

## Protocol notes

### Agent-card discovery

On `__init__` the engine fetches `GET {agent_url}/.well-known/agent.json` and
logs the agent's name. A failure is non-fatal — the engine continues and will
attempt tasks anyway (useful when the card endpoint is behind auth).

### Message history

OVOS `ChatEngine` receives the full conversation list. The plugin maps it to
A2A history as follows:

| OVOS role | A2A mapping |
|-----------|-------------|
| `system` | Injected as the first `user`-role history turn (A2A has no system role) |
| `user` | `user` history turns; the **last** user message becomes the task message |
| `assistant` | `assistant` history turns |

### Streaming

When `streaming: true` the plugin calls `tasks/sendSubscribe` and yields text
chunks via SSE. If the server's agent card reports `capabilities.streaming: false`
the plugin falls back to blocking `tasks/send` and warns.

---

## Development

```bash
git clone https://github.com/TigreGotico/ovos-a2a-solver-plugin
cd ovos-a2a-solver-plugin
pip install -e ".[test]"
pytest test/ -v
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
