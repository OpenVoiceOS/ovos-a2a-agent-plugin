# ovos-a2a-solver-plugin

An [OVOS](https://openvoiceos.org) `ChatEngine` plugin. It lets an [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) answer through a remote agent that speaks the [Agent2Agent (A2A) protocol](https://a2a-protocol.org).

The repository is `ovos-a2a-agent-plugin`. The PyPI package is `ovos-a2a-solver-plugin`, the Python module is `ovos_a2a_solver`, and the plugin id is `ovos-a2a-solver`.

- [What you can do](#what-you-can-do)
- [Answer through a remote A2A agent](#answer-through-a-remote-a2a-agent)
- [Use the A2A endpoint of ovos-persona-server](#use-the-a2a-endpoint-of-ovos-persona-server)
- [Handle a remote agent that is down or slow](#handle-a-remote-agent-that-is-down-or-slow)
- [Reference](#reference)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## What you can do

A2A is an open protocol that lets one agent call another, whatever framework each one uses inside. An A2A agent publishes an *agent card* at `/.well-known/agent-card.json`. The card names the agent, its skills, and the URL that accepts messages.

This plugin is the client side. A persona that uses it sends each user message to one A2A agent and speaks the agent's answer. The plugin uses the [a2a-sdk](https://pypi.org/project/a2a-sdk/) client. It reads the card, picks the transport the card offers, and speaks A2A 1.0 or 0.3, whichever the agent supports.

The plugin never invents an A2A `contextId`. The first message of an OVOS session goes without one, the agent assigns it, and every later message of that session carries the agent's value. The remote agent keeps the conversation for each `contextId`, so a follow-up question in the same session reaches the same conversation. A2A 1.0.1 section 3.4.1 says a client SHOULD NOT send a `contextId` it generated itself. Each device that runs the plugin keeps its own map, so two devices never share a remote conversation, even when both use the OVOS session `default`. The plugin sends only the newest user message. A2A has no field for the client to send earlier turns.

## Answer through a remote A2A agent

Use this setup when you run OVOS and have an A2A agent somewhere: a LangGraph or CrewAI service, another company's assistant, or a second OVOS box. The persona hands each question to that agent.

1. Install the plugin next to ovos-persona:

   ```bash
   pip install --pre ovos-a2a-solver-plugin
   ```

2. Write a persona file. ovos-persona reads every `*.json` file in `~/.config/ovos_persona/`. The `name` field is the persona name, and the file name is used when `name` is missing. `handlers` lists the plugins that answer, in order. The key with the plugin id holds its settings:

   ```json
   {
     "name": "remote agent",
     "handlers": ["ovos-a2a-solver"],
     "ovos-a2a-solver": {
       "agent_url": "http://127.0.0.1:8337/a2a",
       "timeout": 30
     }
   }
   ```

3. Restart ovos-core and say "ask remote agent about the speed of light". To send every unmatched utterance to this persona, set it as the default in `mycroft.conf`:

   ```json
   {"intents": {"persona": {"default_persona": "remote agent", "handle_fallback": true}}}
   ```

On success, the agent's answer is spoken. With a test agent that echoes its input, the OVOS persona pipeline spoke:

```text
echo[f9e2fee0-1f79-4f38-99a3-d73a091659dd]: ask remote agent about the speed of light
```

The id in brackets is the `contextId` the agent assigned to this OVOS session.

When the agent needs more, it can stop the task in `TASK_STATE_INPUT_REQUIRED` and ask. The persona speaks the question, and the user's next utterance goes to that same task, with its `taskId` and `contextId`. With a test agent that asks for a city, the persona pipeline spoke:

```text
USER: ask      -> which city?
USER: lisbon   -> continued[0f1ff863-13c5-4468-8c46-ad0d0e44f1f6]: lisbon
```

When the agent no longer knows the task, for example after a restart, or has already finished it, it answers with an error. The persona speaks its error dialog once. The next utterance starts a new task in the same remote conversation, so the agent keeps the earlier turns it still holds. With an agent that finished a login task on its own, `recall` after the error still answered `gold`, stored two turns before.

An agent may also expire or refuse the conversation itself. A2A requires it to answer with an error, but names no error code for this. The plugin treats these answers as a refused conversation, and the next utterance starts a new remote conversation:

| Agent answer | Plugin action |
|---|---|
| `TaskNotFoundError` (-32001) or `UnsupportedOperationError` (-32004), to a message that named a task | Forget the task, keep the conversation. |
| `TaskNotFoundError` (-32001) or `UnsupportedOperationError` (-32004), to a message that named no task | Forget the conversation. |
| `InvalidParamsError` (-32602) | Forget the conversation. |
| Timeout, HTTP error, network error, `InternalError` | Keep both. |

In each case the persona speaks its error dialog once. With an agent that expired the conversation, the next `recall` answered `nothing`, and every later utterance worked. A2A also uses -32602 for a single message the agent finds invalid, so one such message ends the conversation too. This is deliberate: the plugin never sends a context again after the agent refused it.

To try the persona without ovos-core, call it from Python:

```python
from ovos_bus_client.session import Session
from ovos_persona import Persona
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

persona = Persona("remote agent", {
    "name": "remote agent",
    "handlers": ["ovos-a2a-solver"],
    "ovos-a2a-solver": {"agent_url": "http://127.0.0.1:8337/a2a"},
})
print(persona.chat([AgentMessage(role=MessageRole.USER, content="what is the speed of light?")],
                   Session("kitchen")))
```

## Use the A2A endpoint of ovos-persona-server

[ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server) can publish any persona as an A2A agent. This makes one OVOS box the brain of another:

```bash
pip install --pre "ovos-persona-server[a2a]" ovos-solver-rivescript-plugin "httpx<1"
echo '{"name": "rive", "solvers": ["ovos-solver-rivescript-plugin"]}' > rive.json
ovos-persona-server --persona rive.json --host 0.0.0.0 --port 8337 \
    --a2a-base-url http://127.0.0.1:8337/a2a
```

Set `agent_url` to the `--a2a-base-url` value. Use the address the OVOS box can reach, not `localhost`, when the server runs on another machine. The Python example above, pointed at this server, printed the rivescript persona's answer:

```text
Does that question interest you?
```

## Handle a remote agent that is down or slow

When the agent cannot answer, the plugin raises an error, and the persona service speaks one line of its error dialog, `persona_error.dialog`. The lines are picked at random. In both cases below, the OVOS persona pipeline spoke, for example:

```text
We encountered an error while chatting with remote agent.
```

| Situation | What the log shows |
|---|---|
| Agent unreachable | `A2AChatEngine: task failed: Network communication error fetching agent card from http://127.0.0.1:1/.well-known/agent-card.json: All connection attempts failed` |
| Agent slower than `timeout` | `A2AChatEngine: task failed: Client Request timed out` |
| Agent task failed | `A2A task ended in TASK_STATE_FAILED: <the agent's reason>` |

To answer something other than the error dialog, add a second handler. The persona tries the handlers in order:

```json
{
  "name": "remote agent",
  "handlers": ["ovos-a2a-solver", "ovos-solver-failure-plugin"],
  "ovos-a2a-solver": {"agent_url": "http://127.0.0.1:8337/a2a"}
}
```

With the agent down, this persona answered with one line of the failure plugin's `no_brain.dialog`, for example `404 brain not found`. The other lines are `please install a brain before asking me questions` and `critical error, brain not available`.

## Reference

### Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent_url` | `str` | (required) | Base URL of the agent: the part before `/.well-known/agent-card.json`. |
| `auth_header` | `str` | none | Full `Authorization` header value, for example `Bearer <token>`. It goes on every request, the card request included. |
| `timeout` | `float` | `60` | Seconds to wait for each HTTP request. |
| `streaming` | `bool` | `false` | Use `SendStreamingMessage`, so `stream_tokens` yields text as the agent sends it. When the agent card says `streaming: false`, the plugin logs a warning and uses `SendMessage`. |

### What is sent and received

| OVOS | A2A |
|---|---|
| Newest user message with no assistant answer after it | `SendMessage` with one text part, role `ROLE_USER` |
| `session_id` | the `contextId` the agent assigned to that session, after its first message, and the `taskId` of a task waiting for the user's answer |
| System and earlier messages | not sent |
| Agent message, task artifacts, artifact updates, status-update messages | the answer text |
| Task status message, when the task has no artifact text | the answer text, for example the question of a task in `TASK_STATE_INPUT_REQUIRED` |
| Task state `FAILED`, `REJECTED` or `CANCELED` | `RuntimeError` with the agent's status message |

### Python API

`A2AChatEngine(config)` is the plugin. `continue_chat(messages, session_id)` returns an `AgentMessage`, and `stream_tokens(messages, session_id)` yields text.

`A2AClient(base_url, auth_header=None, timeout=60, streaming=False)` works without OPM:

```python
from ovos_a2a_solver import A2AClient

with A2AClient("http://127.0.0.1:8337/a2a") as client:
    print(client.fetch_agent_card().name)
    print(client.send_task("hello", session_id="my-session"))
    for chunk in client.stream_task("hello", session_id="my-session"):
        print(chunk, end="")
```

`fetch_agent_card()` returns an `AgentCard` with `name`, `description`, `url`, `version`, `skills` and `streaming`. Errors are the a2a-sdk errors: `AgentCardResolutionError`, `A2AClientTimeoutError` and `A2AClientError`.

## Deployment

The plugin runs inside ovos-core, or inside any process that loads ovos-persona, such as ovos-persona-server. It opens outbound HTTP connections only. It needs no open port.

A2A has no built-in encryption. Use an `https://` `agent_url` when the agent is on another machine, and `auth_header` when the agent asks for a token.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Network communication error fetching agent card` | The agent is down, or `agent_url` is wrong. `curl <agent_url>/.well-known/agent-card.json` must return JSON. |
| `Failed to fetch agent card from <url>/.well-known/agent-card.json (HTTP 404)` | `agent_url` points at the wrong path, or the agent publishes only `/.well-known/agent.json`, the A2A 0.2 path. The plugin speaks A2A 0.3 and 1.0 only. |
| `Client Request timed out` | Increase `timeout`, or make the agent answer faster. |
| The agent forgets earlier turns | The agent must keep history per `contextId`. The plugin sends only the newest message. |
| `AttributeError: module 'httpx' has no attribute 'AsyncClient'` | A prerelease of httpx 1.0 is installed. Install `httpx<1`. |

## Development

```bash
git clone https://github.com/OpenVoiceOS/ovos-a2a-agent-plugin
cd ovos-a2a-agent-plugin
uv venv && uv pip install --prerelease=allow -e ".[test]"
.venv/bin/pytest test/
```

The tests start a real a2a-sdk agent (`test/a2a_agent.py`) in a subprocess. Run it alone to try the plugin by hand: `python test/a2a_agent.py 9999`.

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

## License

Apache License 2.0. See [LICENSE](LICENSE).
