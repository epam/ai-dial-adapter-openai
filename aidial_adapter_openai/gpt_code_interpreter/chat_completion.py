from typing import Any, List, Tuple

from fastapi.responses import Response
from openai import AsyncAssistantEventHandler
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.code_interpreter_tool_param import (
    CodeInterpreterToolParam,
)
from openai.types.beta.thread_create_params import Message as ThreadMessage
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import RunStep, RunStepDelta
from typing_extensions import override

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.gpt_code_interpreter.bing_search import (
    search_bing_tool,
)
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    chat_completions_parser,
)
from aidial_adapter_openai.utils.streaming import create_single_message_response


class EventHandler(AsyncAssistantEventHandler):
    @override
    async def on_event(self, event: AssistantStreamEvent) -> None:
        if event.event == "thread.run.step.created":
            details = event.data.step_details
            if details.type == "tool_calls":
                print("Generating code to interpret:\n\n```py")
        elif event.event == "thread.message.created":
            print("\nResponse:\n")

    @override
    async def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        print(delta.value, end="", flush=True)

    @override
    async def on_run_step_done(self, run_step: RunStep) -> None:
        details = run_step.step_details
        if details.type == "tool_calls":
            for tool in details.tool_calls:
                if tool.type == "code_interpreter":
                    print("\n```\nExecuting code...")

    @override
    async def on_run_step_delta(
        self, delta: RunStepDelta, snapshot: RunStep
    ) -> None:
        details = delta.step_details
        if details is not None and details.type == "tool_calls":
            for tool in details.tool_calls or []:
                if (
                    tool.type == "code_interpreter"
                    and tool.code_interpreter
                    and tool.code_interpreter.input
                ):
                    print(tool.code_interpreter.input, end="", flush=True)


def parse_dial_messages(
    dial_messages: List[dict],
) -> Tuple[str | None, List[ThreadMessage]]:
    instructions: str | None = None
    if len(dial_messages) > 0 and dial_messages[0]["role"] == "system":
        instructions = dial_messages[0]["content"]
        dial_messages = dial_messages[1:]

    thread_messages: List[ThreadMessage] = []
    for message in dial_messages:
        role = message["role"]
        if role == "system":
            raise HTTPException(
                "System messages other than the first one are not allowed",
                400,
                "invalid_request_error",
            )
        content = message.get("content") or ""
        thread_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return instructions, thread_messages


async def chat_completion(
    request: Any,
    stream: bool,
    endpoint: str,
    creds: OpenAICreds,
    api_version: str,
) -> Response:

    openai_endpoint = chat_completions_parser.parse(endpoint)

    if not isinstance(openai_endpoint, AzureOpenAIEndpoint):
        raise HTTPException(
            "Invalid upstream endpoint format", 400, "invalid_request_error"
        )

    model = openai_endpoint.azure_deployment

    client = openai_endpoint.get_assistants_client(
        {**creds, "api_version": api_version, "timeout": DEFAULT_TIMEOUT}
    )

    instructions, thread_messages = parse_dial_messages(request["messages"])

    code_interpreter: CodeInterpreterToolParam = {"type": "code_interpreter"}

    assistant = await client.beta.assistants.create(model=model)

    try:
        # TODO: should save the thread_id in the state and
        # add a message to a thread here instead of recreating the thread
        # Recreate only when the thread has been cancelled or expired...
        thread = await client.beta.threads.create(messages=thread_messages)

        # run = await client.beta.threads.runs.create(
        #     thread_id=thread.id,
        #     assistant_id=assistant.id,
        # )

        async with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions=instructions,
            tools=[code_interpreter, search_bing_tool],
            event_handler=EventHandler(),
        ) as run_stream:
            await run_stream.until_done()

        # await poll_run_till_completion(
        #     client=client,
        #     thread_id=thread.id,
        #     run_id=run.id,
        #     available_functions={"search_bing": search_bing},
        # )

        # content = await get_response(
        #     client=client,
        #     thread_id=thread.id,
        #     run_id=run.id,
        #     out_dir="data",
        # )

    finally:
        await client.beta.assistants.delete(assistant.id)

    return create_single_message_response("WHAT?", stream)
