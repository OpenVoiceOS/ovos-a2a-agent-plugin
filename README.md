# ovos-a2a-solver-plugin

An [OVOS](https://openvoiceos.org) `ChatEngine` plugin. It lets an [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) send its reasoning to an external agent that speaks the [Agent2Agent (A2A) protocol](https://google.github.io/A2A/).

This plugin implements the consumer side of [OpenVoiceOS/ovos-persona#164](https://github.com/OpenVoiceOS/ovos-persona/issues/164).

---

## What is A2A?

Agent2Agent (A2A) is an open protocol. It lets software agents work together no matter how each one is built inside. An A2A-compliant server does three things:

1. It exposes a discovery document at `GET /.well-known/agent.json` (the *agent card*). This card describes the server's identity and its skills and capabilities.
2. It accepts tasks at its root URL as [JSON-RPC 2.0](https://www.jsonrpc.org/specification) requests (`tasks/send` for a blocking call, `tasks/sendSubscribe` for SSE streaming).
3. It returns structured responses with typed *artifact parts*: text, file, or data.

This plugin implements only the **client (consumer) side**. It calls remote A2A agents from inside an OVOS persona.

---

## Installation

```bash
pip install ovos-a2a-solver-plugin
```

This plugin needs Python 3.10 or later, and `ovos-plugin-manager >= 2.3.0a1`.

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
| `agent_url` | `str` | (none) | Required. Root URL of the A2A server. |
| `auth_header` | `str` | `None` | Full `Authorization` header value, for example `Bearer <token>`. |

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `timeout` | `float` | `60` | HTTP timeout in seconds. |
| `streaming` | `bool` | `False` | Use `tasks/sendSubscribe` (SSE) instead of the blocking `tasks/send`. The server must advertise `capabilities.streaming: true`. |

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

You can also use the engine directly:

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

## A2AClient (low-level usage)

Use the `A2AClient` class on its own, without OPM:

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

On `__init__` the engine fetches `GET {agent_url}/.well-known/agent.json` and logs the agent's name. A failure here is not fatal. The engine continues and still attempts tasks. This matters when the card endpoint sits behind auth.

### Message history

The OVOS `ChatEngine` receives the full conversation list. The plugin maps it to A2A history as follows:

| OVOS role | A2A mapping |
|-----------|-------------|
| `system` | Injected as the first `user`-role history turn (A2A has no system role) |
| `user` | `user` history turns. The last user message becomes the task message |
| `assistant` | `assistant` history turns |

### Streaming

When `streaming: true`, the plugin calls `tasks/sendSubscribe` and yields text chunks over SSE. If the server's agent card reports `capabilities.streaming: false`, the plugin falls back to the blocking `tasks/send` and logs a warning.

---

## Development

```bash
git clone https://github.com/TigreGotico/ovos-a2a-solver-plugin
cd ovos-a2a-solver-plugin
pip install -e ".[test]"
pytest test/ -v
```

---

## Related projects

- [OpenVoiceOS/ovos-persona](https://github.com/OpenVoiceOS/ovos-persona): the persona service that loads this plugin as a `ChatEngine`.
- [OpenVoiceOS/ovos-persona#164](https://github.com/OpenVoiceOS/ovos-persona/issues/164): the issue this plugin implements the consumer side of.

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
