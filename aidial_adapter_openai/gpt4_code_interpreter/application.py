from typing import List, Tuple

from aidial_sdk.chat_completion import (
    ChatCompletion,
    Choice,
    Request,
    Response,
    Stage,
)
from openai import AsyncAssistantEventHandler
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.code_interpreter_tool_param import (
    CodeInterpreterToolParam,
)
from openai.types.beta.thread_create_params import Message as ThreadMessage
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.image_file import ImageFile
from openai.types.beta.threads.message import Message as MessageRaw
from openai.types.beta.threads.message_delta import MessageDelta
from openai.types.beta.threads.runs import RunStep, RunStepDelta
from openai.types.beta.threads.runs.code_interpreter_logs import (
    CodeInterpreterLogs,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import (
    CodeInterpreterToolCallDelta,
)
from openai.types.beta.threads.runs.run_step import Usage
from openai.types.beta.threads.runs.tool_call import ToolCall
from openai.types.beta.threads.runs.tool_call_delta import ToolCallDelta
from typing_extensions import override

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.gpt4_code_interpreter.bing_search import (
    search_bing_tool,
)
from aidial_adapter_openai.utils.auth import OpenAICreds, get_credentials
from aidial_adapter_openai.utils.exceptions import (
    HTTPException,
    handle_exceptions,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    chat_completions_parser,
)
from aidial_adapter_openai.utils.version import get_api_version


class EventHandler(AsyncAssistantEventHandler):
    response: Response
    choice: Choice
    tool_calls_input: dict[str, Stage]
    tool_calls_output: dict[str, Stage]
    usage: Usage

    def __init__(self, response: Response, choice: Choice) -> None:
        super().__init__()
        self.response = response
        self.choice = choice
        self.tool_calls_input = {}
        self.tool_calls_output = {}
        self.usage = Usage(
            prompt_tokens=0,
            total_tokens=0,
            completion_tokens=0,
        )

    async def accumulate_usage(self, usage: Usage) -> None:
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.total_tokens += usage.total_tokens
        self.usage.completion_tokens += usage.completion_tokens

    def get_tool_calls_input(self, id: str) -> Stage:
        if id in self.tool_calls_input:
            return self.tool_calls_input[id]
        else:
            idx = len(self.tool_calls_input) + 1
            stage = self.choice.create_stage(f"Calling tool #{idx}")
            stage.open()
            stage.append_content("```python\n")
            self.tool_calls_input[id] = stage
            return stage

    def get_tool_calls_output(self, id: str) -> Stage:
        if id in self.tool_calls_output:
            return self.tool_calls_output[id]
        else:
            idx = len(self.tool_calls_output) + 1
            stage = self.choice.create_stage(f"Tool output #{idx}")
            stage.open()
            stage.append_content("```\n")
            self.tool_calls_output[id] = stage
            return stage

    def close_tool_calls_input(self, id: str) -> None:
        if id in self.tool_calls_input:
            self.tool_calls_input[id].append_content("\n```")
            self.tool_calls_input[id].close()
            del self.tool_calls_input[id]

    def close_tool_calls_output(self, id: str) -> None:
        if id in self.tool_calls_output:
            self.tool_calls_output[id].append_content("\n```")
            self.tool_calls_output[id].close()
            del self.tool_calls_output[id]

    def cleanup_stages(self) -> None:
        ids = list(self.tool_calls_input.keys())
        for id in ids:
            self.close_tool_calls_input(id)

        ids = list(self.tool_calls_output.keys())
        for id in ids:
            self.close_tool_calls_output(id)

    @override
    async def on_event(self, event: AssistantStreamEvent) -> None:
        # Print every single event. This is useful for debugging.
        logger.debug(f"on_event: {event.json(exclude_none=True)}\n")
        return

    @override
    async def on_run_step_created(self, run_step: RunStep) -> None:
        logger.debug(
            f"on_run_step_created: {run_step.json(exclude_none=True)}\n"
        )

    @override
    async def on_run_step_delta(
        self, delta: RunStepDelta, snapshot: RunStep
    ) -> None:
        logger.debug(
            f"on_run_step_delta: {delta.json(exclude_none=True)}\n{snapshot.json(exclude_none=True)}\n"
        )

    @override
    async def on_run_step_done(self, run_step: RunStep) -> None:
        logger.debug(f"on_run_step_done: {run_step.json(exclude_none=True)}\n")

        if run_step.usage is not None:
            await self.accumulate_usage(run_step.usage)

    @override
    async def on_tool_call_created(self, tool_call: ToolCall) -> None:
        logger.debug(
            f"on_tool_call_created: {tool_call.json(exclude_none=True)}\n"
        )

    @override
    async def on_tool_call_delta(
        self, delta: ToolCallDelta, snapshot: ToolCall
    ) -> None:
        # Print to a tool call stage
        # Input: delta.code_interpreter.input
        # Output: delta.code_interpreter.outputs[*].logs
        logger.debug(
            f"on_tool_call_delta: {delta.json(exclude_none=True)}\n{snapshot.json(exclude_none=True)}\n"
        )

        id = snapshot.id

        match delta:
            case CodeInterpreterToolCallDelta(code_interpreter=ci):
                if ci is None:
                    return

                if ci.input is not None:
                    self.get_tool_calls_input(id).append_content(ci.input)

                for output in ci.outputs or []:
                    match output:
                        case CodeInterpreterLogs(logs=logs):
                            if logs is not None:
                                self.get_tool_calls_output(id).append_content(
                                    logs
                                )
                        case _:
                            raise ValueError(
                                f"Unsupported code interpreter output type: {output.type}"
                            )
            case _:
                raise ValueError(
                    f"Unsupported tool call delta type: {delta.type}"
                )

    @override
    async def on_tool_call_done(self, tool_call: ToolCall) -> None:
        logger.debug(
            f"on_tool_call_done: {tool_call.json(exclude_none=True)}\n"
        )

        id = tool_call.id
        self.close_tool_calls_input(id)
        self.close_tool_calls_output(id)

    @override
    async def on_message_created(self, message: MessageRaw) -> None:
        logger.debug(f"on_message_created: {message.json(exclude_none=True)}\n")

    @override
    async def on_message_delta(
        self, delta: MessageDelta, snapshot: MessageRaw
    ) -> None:
        logger.debug(
            f"on_message_delta: {delta.json(exclude_none=True)}\n{snapshot.json(exclude_none=True)}\n"
        )

    @override
    async def on_message_done(self, message: MessageRaw) -> None:
        logger.debug(f"on_message_done: {message.json(exclude_none=True)}\n")

    @override
    async def on_text_created(self, text: Text) -> None:
        logger.debug(f"on_text_created: {text.json(exclude_none=True)}\n")

    @override
    async def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        logger.debug(
            f"on_text_delta: {delta.json(exclude_none=True)}\n{snapshot.json(exclude_none=True)}\n"
        )

        if delta.value is not None:
            self.choice.append_content(delta.value)

    @override
    async def on_text_done(self, text: Text) -> None:
        logger.debug(f"on_text_done: {text.json(exclude_none=True)}\n")

    @override
    async def on_image_file_done(self, image_file: ImageFile) -> None:
        logger.debug(
            f"on_image_file_done: {image_file.json(exclude_none=True)}\n"
        )

    async def report_usage(self) -> None:
        self.response.set_usage(
            self.usage.prompt_tokens, self.usage.completion_tokens
        )

    @override
    async def on_exception(self, exception: Exception) -> None:
        logger.error(f"on_exception: {str(exception)}\n")
        raise exception

    @override
    async def on_timeout(self) -> None:
        logger.error("on_timeout:\n")
        raise TimeoutError("The run has timed out")

    @override
    async def on_end(self) -> None:
        logger.debug("on_end:\n")
        self.cleanup_stages()
        await self.report_usage()


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


class CodeInterpreterApplication(ChatCompletion):
    deployment_id: str

    def __init__(self, deployment_id: str) -> None:
        self.deployment_id = deployment_id

    async def chat_completion(self, request: Request, response: Response):

        if request.n is not None and request.n > 1:
            raise ValueError("n>1 isn't supported")

        creds = await get_credentials(request.headers.get("X-UPSTREAM-KEY"))
        endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

        api_version = get_api_version(request.api_version)

        data = request.dict()
        for message in data["messages"]:
            message["role"] = message["role"].value

        data["model"] = self.deployment_id

        await handle_exceptions(
            chat_completion(data, endpoint, creds, api_version, response)
        )


async def chat_completion(
    request: dict,
    endpoint: str,
    creds: OpenAICreds,
    api_version: str,
    response: Response,
) -> None:

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

        with response.create_single_choice() as choice:
            async with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                instructions=instructions,
                tools=[code_interpreter, search_bing_tool],
                event_handler=EventHandler(response, choice),
            ) as run_stream:
                await run_stream.until_done()

    finally:
        await client.beta.assistants.delete(assistant.id)
