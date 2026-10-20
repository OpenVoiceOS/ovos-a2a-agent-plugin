"""A real A2A agent built on a2a-sdk, run as a subprocess by the test fixtures.

usage: python a2a_agent.py PORT [--a2a-0.3]

``--a2a-0.3`` publishes a card whose interface declares protocol version 0.3,
and accepts the 0.3 method names next to the 1.0 ones. Without it the agent
accepts the 1.0 method names only.

The user text selects the behaviour:

- ``fail``         the task ends in TASK_STATE_FAILED with a status message
- ``ask``          the task stops in TASK_STATE_INPUT_REQUIRED, asking ``which city?``.
                   The next message to that task completes it with
                   ``continued[<taskId>]: <text>``
- ``authreq``      the task stops in TASK_STATE_AUTH_REQUIRED, asking ``login please``.
                   The next message to that task completes it like ``ask``
- ``authoob N``    the task stops in TASK_STATE_AUTH_REQUIRED, asking ``open the link to
                   log in``, and completes on its own N seconds later, as after a
                   login out of band (A2A 1.0.1 section 7.6.1)
- ``remember X``   stores X for this contextId and answers ``stored``
- ``recall``       answers what this contextId stored, or ``nothing``
- ``forget``       answers ``forgotten``; every later message that names this
                   contextId is rejected with InvalidParamsError (-32602)
- ``expire``       the same, rejected with TaskNotFoundError (-32001), as an
                   agent that expired the context (A2A 1.0.1 section 3.4.1)
- ``expire-terminal`` the same, rejected with UnsupportedOperationError (-32004)
- ``done``         the task completes with the status message ``final words`` and no artifact
- ``slow N ...``   waits N seconds, then echoes
- ``chunks ...``   answers as a task, one artifact chunk per word
- ``drip ...``     like ``chunks``, with half a second between the chunks
- ``adk``          a working status message with the answer, then an artifact that repeats it
- ``progress``     a working status message ``Looking up the answer...``, then the answer as an artifact
- ``cumulative``   the answer as artifact updates that each replace the artifact with all the text so far
- ``artifacts``    two artifacts, ``First part.`` and ``Second part.``
- ``rewrite``      two replacing artifact updates, ``... is Sydney.`` then ``... is Canberra.``
- ``interim``      the artifact ``Thinking...`` replaced by the answer
- ``helloworld``   a working status, the answer as an artifact, then a completed status message
                   ``Request is completed!`` (the a2a-samples hello world agent)
- ``nopunct``      two artifacts, ``First part`` and ``Second part``, without final punctuation
- ``wsrewrite``    the answer as two appended chunks that end in a space, then a replacement
                   with the whole answer without the trailing space
- ``tworewrites``  the artifacts ``Alpha old.`` and ``Beta old.``, then ``Beta new.`` and
                   ``Alpha new.`` as replacements, in that order
- ``poem``         one artifact of four lines without final punctuation, separated by newlines
- ``snapshot ...`` streams one artifact chunk per word, then a completed task
                   snapshot that carries the whole answer again
- anything else    one agent message: ``echo[<contextId>]: <text>``

``GET /last-auth`` returns the Authorization header of the last request. A
JSON-RPC request with ``Authorization: Bearer status-NNN`` is answered with
HTTP status NNN. A message whose text is ``stall`` never reaches the agent:
the request waits 5 seconds and is answered with HTTP 503, so a client with a
shorter timeout sees a timeout and the agent's state does not change.
"""
import asyncio
import json
import sys

import uvicorn
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from a2a.helpers.proto_helpers import get_message_text, new_task, new_text_message, new_text_part
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Artifact, Task,
                       TaskState, TaskStatus)
from a2a.utils.errors import InvalidParamsError, TaskNotFoundError, UnsupportedOperationError

PORT = int(sys.argv[1])
A2A_0_3 = "--a2a-0.3" in sys.argv
BASE_URL = f"http://127.0.0.1:{PORT}"
last_auth = {"value": ""}
REJECTIONS = {"forget": InvalidParamsError, "expire": TaskNotFoundError,
              "expire-terminal": UnsupportedOperationError}
memory: dict[str, str] = {}
# contextId -> the error every later message that names it is answered with
rejected_contexts: dict[str, type] = {}


SHAPES = ("adk", "progress", "cumulative", "artifacts", "rewrite", "interim", "helloworld", "nopunct",
          "wsrewrite", "tworewrites", "poem")
ANSWER = "The answer is 42. Have a nice day."


async def send_shape(shape: str, updater: TaskUpdater) -> None:
    """Answer the way agents built on other frameworks stream."""
    if shape == "adk":
        await updater.update_status(TaskState.TASK_STATE_WORKING,
                                    updater.new_agent_message([new_text_part(ANSWER)]))
        await updater.add_artifact([new_text_part(ANSWER)], artifact_id="final", last_chunk=True)
    elif shape == "progress":
        await updater.update_status(TaskState.TASK_STATE_WORKING,
                                    updater.new_agent_message([new_text_part("Looking up the answer...")]))
        await updater.add_artifact([new_text_part(ANSWER)], artifact_id="final", last_chunk=True)
    elif shape == "cumulative":
        so_far = ""
        for piece in ("The answer is 42. ", "Have a ", "nice day."):
            so_far += piece
            await updater.add_artifact([new_text_part(so_far)], artifact_id="answer", append=False)
    elif shape == "artifacts":
        await updater.add_artifact([new_text_part("First part.")], artifact_id="one")
        await updater.add_artifact([new_text_part("Second part.")], artifact_id="two")
    elif shape == "rewrite":
        for city in ("Sydney", "Canberra"):
            await updater.add_artifact([new_text_part(f"The capital of Australia is {city}.")],
                                       artifact_id="answer", append=False)
    elif shape == "interim":
        await updater.add_artifact([new_text_part("Thinking...")], artifact_id="answer", append=False)
        await updater.add_artifact([new_text_part(ANSWER)], artifact_id="answer", append=False)
    elif shape == "helloworld":
        await updater.update_status(TaskState.TASK_STATE_WORKING,
                                    updater.new_agent_message([new_text_part("Processing request...")]))
        await updater.add_artifact([new_text_part("Hello, World! I am your agent.")], artifact_id="hello")
        await updater.update_status(TaskState.TASK_STATE_COMPLETED,
                                    updater.new_agent_message([new_text_part("Request is completed!")]))
        return
    elif shape == "nopunct":
        await updater.add_artifact([new_text_part("First part")], artifact_id="one")
        await updater.add_artifact([new_text_part("Second part")], artifact_id="two")
    elif shape == "wsrewrite":
        await updater.add_artifact([new_text_part("The answer is 42. ")], artifact_id="answer", append=False)
        await updater.add_artifact([new_text_part("Have a nice day. ")], artifact_id="answer", append=True)
        await updater.add_artifact([new_text_part(ANSWER)], artifact_id="answer", append=False)
    elif shape == "tworewrites":
        for artifact_id, text in (("alpha", "Alpha old."), ("beta", "Beta old."),
                                  ("beta", "Beta new."), ("alpha", "Alpha new.")):
            await updater.add_artifact([new_text_part(text)], artifact_id=artifact_id, append=False)
    elif shape == "poem":
        await updater.add_artifact(
            [new_text_part("Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you.")],
            artifact_id="poem", append=False)
    await updater.complete()


class TestAgent(AgentExecutor):
    async def execute(self, context, event_queue):
        text = get_message_text(context.message)
        if context.current_task and context.current_task.status.state in (
                TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED):
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await updater.complete(updater.new_agent_message(
                [new_text_part(f"continued[{context.task_id}]: {text}")]))
            return
        words = text.split()
        if words[:1] == ["remember"]:
            memory[context.context_id] = " ".join(words[1:])
            text = "stored"
        elif text == "recall":
            text = memory.get(context.context_id, "nothing")
        elif text in REJECTIONS:
            rejected_contexts[context.context_id] = REJECTIONS[text]
            text = "forgotten"
        if words[:1] in (["remember"], ["recall"]) or text == "forgotten":
            await event_queue.enqueue_event(new_text_message(
                text, context_id=context.context_id, task_id=context.task_id))
            return
        if words[:1] == ["authoob"]:
            await event_queue.enqueue_event(new_task(context.task_id, context.context_id,
                                                     TaskState.TASK_STATE_WORKING))
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await updater.requires_auth(updater.new_agent_message([new_text_part("open the link to log in")]))
            await asyncio.sleep(float(words[1]))
            await updater.complete(updater.new_agent_message([new_text_part("logged in out of band")]))
            return
        if words and words[0] == "slow":
            await asyncio.sleep(float(words[1]))
        if text in ("fail", "ask", "authreq", "done", *SHAPES) or words[:1] in (["chunks"], ["drip"], ["snapshot"]):
            await event_queue.enqueue_event(new_task(context.task_id, context.context_id,
                                                     TaskState.TASK_STATE_WORKING))
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            if text == "fail":
                await updater.failed(updater.new_agent_message([new_text_part("backend exploded")]))
                return
            if text == "ask":
                await updater.requires_input(updater.new_agent_message([new_text_part("which city?")]))
                return
            if text == "authreq":
                await updater.requires_auth(updater.new_agent_message([new_text_part("login please")]))
                return
            if text == "done":
                await updater.complete(updater.new_agent_message([new_text_part("final words")]))
                return
            if text in SHAPES:
                await send_shape(text, updater)
                return
            for i, word in enumerate(words[1:]):
                await updater.add_artifact([new_text_part(word + " ")], artifact_id="answer",
                                           append=i > 0, last_chunk=i == len(words) - 2)
                if words[0] == "drip":
                    await asyncio.sleep(0.5)
            if words[0] == "snapshot":
                answer = "".join(word + " " for word in words[1:])
                await event_queue.enqueue_event(Task(
                    id=context.task_id, context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                    artifacts=[Artifact(artifact_id="answer", parts=[new_text_part(answer)])]))
                return
            await updater.complete()
            return
        await event_queue.enqueue_event(new_text_message(
            f"echo[{context.context_id}]: {text}",
            context_id=context.context_id, task_id=context.task_id))

    async def cancel(self, context, event_queue):
        pass


card = AgentCard(
    name="test-agent", description="a2a-sdk test agent", version="1.0.0",
    supported_interfaces=[AgentInterface(url=BASE_URL, protocol_binding="JSONRPC",
                                         protocol_version="0.3" if A2A_0_3 else "1.0")],
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain"], default_output_modes=["text/plain"],
    skills=[AgentSkill(id="echo", name="Echo", description="Echoes the text", tags=["test"])],
)
class Handler(DefaultRequestHandlerV2):
    @staticmethod
    def _reject_forgotten_context(params):
        error = rejected_contexts.get(params.message.context_id)
        if error:
            raise error(message=f"context {params.message.context_id} is no longer accepted")

    async def on_message_send(self, params, context):
        self._reject_forgotten_context(params)
        return await super().on_message_send(params, context)

    async def on_message_send_stream(self, params, context):
        self._reject_forgotten_context(params)
        async for event in super().on_message_send_stream(params, context):
            yield event


handler = Handler(agent_executor=TestAgent(), task_store=InMemoryTaskStore(), agent_card=card)


async def show_last_auth(request):
    return PlainTextResponse(last_auth["value"])


starlette_app = Starlette(routes=[Route("/last-auth", show_last_auth)]
                          + create_agent_card_routes(card)
                          + create_jsonrpc_routes(handler, rpc_url="/", enable_v0_3_compat=A2A_0_3))


def message_text(body: bytes) -> str:
    try:
        parts = json.loads(body)["params"]["message"]["parts"]
    except (ValueError, KeyError, TypeError):
        return ""
    return "".join(part.get("text", "") for part in parts)


async def app(scope, receive, send):
    """Record the Authorization header, force HTTP statuses, and stall ``stall`` messages."""
    if scope["type"] == "http" and scope["path"] != "/last-auth":
        auth = dict(scope["headers"]).get(b"authorization", b"").decode()
        last_auth["value"] = auth
        if scope["method"] == "POST" and auth.startswith("Bearer status-"):
            await PlainTextResponse("forced status", status_code=int(auth[len("Bearer status-"):]))(
                scope, receive, send)
            return
        if scope["method"] == "POST":
            body, more = b"", True
            while more:
                chunk = await receive()
                body += chunk.get("body", b"")
                more = chunk.get("more_body", False)
            if message_text(body) == "stall":
                await asyncio.sleep(5)
                await PlainTextResponse("stalled", status_code=503)(scope, receive, send)
                return
            replayed, upstream = False, receive

            async def receive():
                nonlocal replayed
                if replayed:
                    return await upstream()
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
    await starlette_app(scope, receive, send)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
