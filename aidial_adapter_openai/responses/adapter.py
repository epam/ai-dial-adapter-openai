import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from aidial_sdk.exceptions import RequestValidationError
from openai import (
    AsyncAzureOpenAI,
    AsyncBedrockOpenAI,
    AsyncOpenAI,
    AsyncStream,
    BaseModel,
    omit,
)
from openai.types.responses import ResponseTextConfigParam
from openai.types.shared_params import Reasoning
from pydantic import Field

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.converter import (
    _DEPRECATED_FUNCTION_API,
    convert_messages,
    convert_response,
    convert_tool_choice,
    convert_tools,
)
from aidial_adapter_openai.responses.event_handler import EventHandler
from aidial_adapter_openai.responses.request import convert_response_format
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from aidial_adapter_openai.utils.streaming import (
    map_stream,
    map_stream_generator,
)


def _validate_request(request: dict[str, Any]) -> None:
    errors: list[str] = []

    if (n := request.get("n")) not in [None, 1]:
        errors.append(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    unsupported_params: list[str] = []
    for param in [
        "stop",
        "seed",
        "top_logprobs",
        "logprobs",
        "presence_penalty",
    ]:
        if request.get(param) is not None:
            unsupported_params.append(param)

    if unsupported_params:
        suffix = "s" if len(unsupported_params) > 1 else ""
        errors.append(
            f"The deployment doesn't support {', '.join(unsupported_params)} request parameter{suffix}."
        )

    if (
        request.get("function_call") is not None
        or request.get("functions") is not None
    ):
        errors.append(_DEPRECATED_FUNCTION_API)

    if not request.get("messages"):
        errors.append("The request doesn't contain any messages.")

    if errors:
        raise RequestValidationError(" ".join(errors))


def _to_dict(x: BaseModel) -> dict:
    ret = x.model_dump()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"chat completion API response: {json.dumps(ret)}")
    return ret


class ResponsesConfig(ExtraAllowedModel):
    reasoning: Reasoning | None = Field(
        default=None,
        description="Configuration options for [reasoning models](https://platform.openai.com/docs/guides/reasoning).",
    )


def _get_configuration(request: dict[str, Any]) -> ResponsesConfig:
    configuration = (
        parse_configuration(ResponsesConfig, request) or ResponsesConfig()
    )

    if configuration.reasoning is None and (
        reasoning_effort := request.get("reasoning_effort")
    ):
        configuration.reasoning = Reasoning(effort=reasoning_effort)

    logger.debug(f"configuration: {configuration.model_dump_json()}")
    return configuration


async def chat_completion(
    *,
    request: dict[str, Any],
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI,
    file_storage: FileStorage | None,
) -> AsyncIterator[dict] | dict:
    _validate_request(request)

    is_stream = bool(request.get("stream"))
    model_name = request["model"]
    messages = request["messages"]

    transformed_messages = await ResourceProcessor(
        file_storage=file_storage,
    ).transform_messages(messages)

    input_messages = convert_messages(
        [m.raw_message for m in transformed_messages]  # type: ignore
    )

    res_tools = []
    if tools := request.get("tools"):
        res_tools.extend(convert_tools(tools))

    if "web_search_options" in request:
        res_tools.append(
            {"type": "web_search", **request["web_search_options"]}
        )

    res_tool_choice = omit
    if tool_choice := request.get("tool_choice"):
        res_tool_choice = convert_tool_choice(tool_choice)

    max_output_tokens = (
        request.get("max_tokens")
        or request.get("max_completion_tokens")
        or omit
    )

    configuration = _get_configuration(request)

    text = omit
    if response_format := request.get("response_format"):
        text = ResponseTextConfigParam(
            format=convert_response_format(response_format)
        )

    response = await client.responses.create(
        model=model_name,
        stream=is_stream,
        input=input_messages,
        tools=res_tools or omit,
        tool_choice=res_tool_choice,
        top_p=request.get("top_p") or omit,
        temperature=request.get("temperature") or omit,
        max_output_tokens=max_output_tokens,
        parallel_tool_calls=request.get("parallel_tool_calls") or omit,
        text=text,
        **configuration.model_dump(),
    )

    if isinstance(response, AsyncStream):
        handler = EventHandler()
        return map_stream(
            _to_dict, map_stream_generator(handler.handle, response)
        )
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"responses API response: {json.dumps(response.model_dump())}"
            )
        return _to_dict(convert_response(response))
