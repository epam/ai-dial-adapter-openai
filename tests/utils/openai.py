import json
from typing import Any, Callable, List

from aidial_sdk.utils.merge_chunks import (
    cleanup_indices,
    merge_chat_completion_chunks,
)
from openai import APIError, AsyncAzureOpenAI, AsyncStream, NotGiven
from openai._types import NOT_GIVEN
from openai.types import CompletionUsage, ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as ToolFunction,
)
from openai.types.chat.completion_create_params import Function, ResponseFormat
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from aidial_adapter_openai.utils.resource.base import Resource
from tests.utils.json import match_objects


def sys(content: str) -> ChatCompletionSystemMessageParam:
    return {"role": "system", "content": content}


def ai(content: str) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "content": content}


def ai_function(
    function_call: ToolFunction,
) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "function_call": function_call}


def ai_tools(
    tool_calls: List[ChatCompletionMessageToolCallParam],
) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "tool_calls": tool_calls}


def user(
    content: str | List[ChatCompletionContentPartParam],
    **kwargs,
) -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": content, **kwargs}  # type: ignore


def user_with_attachment_data(
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": content,
        "custom_content": {  # type: ignore
            "attachments": [
                {"type": resource.type, "data": resource.data_base64}
            ]
        },
    }


def user_with_file_content_part(
    content: str, name: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": content},
            {
                "type": "file",
                "file": {
                    "filename": name,
                    "file_data": resource.to_data_url(),
                },
            },
        ],
    }


def user_with_attachment_url(
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": content,
        "custom_content": {  # type: ignore
            "attachments": [
                {
                    "type": resource.type,
                    "url": resource.to_data_url(),
                }
            ]
        },
    }


def user_with_image_content_part(
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": content},
            {
                "type": "image_url",
                "image_url": {"url": resource.to_data_url()},
            },
        ],
    }


def function_request(name: str, args: Any) -> ToolFunction:
    return {"name": name, "arguments": json.dumps(args)}


def tool_request(
    id: str, name: str, args: Any
) -> ChatCompletionMessageToolCallParam:
    return {
        "id": id,
        "type": "function",
        "function": function_request(name, args),
    }


def function_response(
    name: str, content: str
) -> ChatCompletionFunctionMessageParam:
    return {"role": "function", "name": name, "content": content}


def tool_response(id: str, content: str) -> ChatCompletionToolMessageParam:
    return {"role": "tool", "tool_call_id": id, "content": content}


def function_to_tool(function: FunctionDefinition) -> ChatCompletionToolParam:
    return {"type": "function", "function": function}


class ChatCompletionResult(BaseModel):
    response: ChatCompletion

    @property
    def message(self) -> ChatCompletionMessage:
        return self.response.choices[0].message

    @property
    def stages(self) -> list[dict]:
        return self.response.choices[0].message.dict()["custom_content"][
            "stages"
        ]

    @property
    def all_attachments(self) -> list[list[dict]]:
        return [
            choice.message.dict()["custom_content"]["attachments"]
            for choice in self.response.choices
        ]

    @property
    def attachments(self) -> list[dict]:
        return self.all_attachments[0]

    @property
    def content(self) -> str:
        return self.message.content or ""

    @property
    def contents(self) -> List[str]:
        return [
            choice.message.content or "" for choice in self.response.choices
        ]

    @property
    def usage(self) -> CompletionUsage | None:
        return self.response.usage

    @property
    def function_call(self) -> FunctionCall | None:
        return self.message.function_call

    @property
    def tool_calls(self) -> List[ChatCompletionMessageToolCall] | None:
        return self.message.tool_calls

    def content_contains_all(self, matches: List[Any]) -> bool:
        return all(
            str(match).lower() in self.content.lower() for match in matches
        )


async def chat_completion(
    client: AsyncAzureOpenAI,
    *,
    deployment_id: str,
    messages: List[ChatCompletionMessageParam],
    stream: bool,
    stop: List[str] | NotGiven = NOT_GIVEN,
    max_completion_tokens: int | NotGiven = NOT_GIVEN,
    max_tokens: int | NotGiven = NOT_GIVEN,
    n: int | NotGiven = NOT_GIVEN,
    functions: List[Function] | NotGiven = NOT_GIVEN,
    tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
    temperature: float | NotGiven = NOT_GIVEN,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    response_format: ResponseFormat | NotGiven = NOT_GIVEN,
    extra_body: dict | None = None,
) -> ChatCompletionResult:
    async def get_response() -> ChatCompletion:
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=messages,
            stream=stream,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            function_call="auto" if functions is not NOT_GIVEN else NOT_GIVEN,
            functions=functions,
            tool_choice="auto" if tools is not NOT_GIVEN else NOT_GIVEN,
            tools=tools or NOT_GIVEN,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            extra_body=extra_body,
        )

        if isinstance(response, AsyncStream):
            chunks: List[dict] = [
                {}
            ]  # workaround for https://github.com/epam/ai-dial-sdk/pull/269
            async for chunk in response:
                chunks.append(chunk.dict())

            response_dict = merge_chat_completion_chunks(*chunks)

            for choice in response_dict["choices"]:
                choice["message"] = cleanup_indices(choice["delta"])
                del choice["delta"]

            response_dict["object"] = "chat.completion"

            return ChatCompletion.parse_obj(response_dict)
        else:
            return response

    response = await get_response()
    return ChatCompletionResult(response=response)


GET_WEATHER_FUNCTION: FunctionDefinition = {
    "name": "get_current_weather",
    "description": "Get the current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the users location.",
            },
        },
        "required": ["location", "format"],
    },
}


def is_valid_function_call(
    call: FunctionCall | None, expected_name: str, expected_args: Any
) -> bool:
    assert call is not None
    assert call.name == expected_name
    obj = json.loads(call.arguments)
    match_objects(expected_args, obj)
    return True


def is_valid_tool_call(
    calls: List[ChatCompletionMessageToolCall] | None,
    tool_call_idx: int,
    check_tool_id: Callable[[str], bool],
    expected_name: str,
    expected_args: dict,
) -> bool:
    assert calls is not None

    call = calls[tool_call_idx]

    function = call.function
    assert check_tool_id(call.id)
    assert expected_name == function.name

    actual_args = json.loads(function.arguments)
    match_objects(expected_args, actual_args)
    return True


class ExpectedException(BaseModel):
    type: type[APIError]
    message: str | None = None
    display_message: str | None = None
    status_code: int | None = None
