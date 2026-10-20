# ovos-a2a-solver-plugin — Agent Skill Card

## What this plugin does

It sends OVOS persona conversations to an external agent that speaks the
[Agent2Agent (A2A)](https://a2a-protocol.org) protocol, version 1.0 or 0.3.

An A2A agent publishes a discovery card at `/.well-known/agent-card.json`.
The plugin reads the card with the a2a-sdk client and sends each user message
with `SendMessage`, or `SendStreamingMessage` when streaming is on.

## Accessibility angle

Voice-only and blind users talk to OVOS personas that do not know every
domain. A persona that points at a specialist A2A agent, such as a medical
Q&A agent, a transit planner or a recipe assistant, gets that domain with no
code change. The user speaks naturally, and the persona routes the question.

## Quick config (`~/.config/ovos_persona/<name>.json`)

```json
{
  "name": "remote agent",
  "handlers": ["ovos-a2a-solver"],
  "ovos-a2a-solver": {
    "agent_url": "https://my-a2a-agent.example.com",
    "auth_header": "Bearer <token>",
    "timeout": 60,
    "streaming": false
  }
}
```

## Key methods

| Method | What it does |
|--------|-------------|
| `A2AClient.fetch_agent_card()` | Fetch and parse `/.well-known/agent-card.json` |
| `A2AClient.send_task(text, session_id)` | Send one message; messages of one `session_id` share the agent-assigned `contextId` |
| `A2AClient.stream_task(text, session_id)` | Send one message and yield text as it arrives |
| `A2AChatEngine.continue_chat(messages, session_id)` | OPM ChatEngine entry point |
| `A2AChatEngine.stream_tokens(messages, session_id)` | Streaming OPM entry point |

## How to speak results

The plugin returns the agent's plain-text answer. The OVOS TTS pipeline
speaks the text as it is.
