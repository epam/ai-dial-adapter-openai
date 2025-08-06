import json
import logging
from typing import Any, AsyncIterator, Dict, List

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import RequestValidationError
from fastapi.responses import Response as FastAPIResponse
from openai import (
    NOT_GIVEN,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AsyncStream,
    BaseModel,
)

from aidial_adapter_openai.chat_completions.input import SupportedInputs
from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.converter import (
    _DEPRECATED_FUNCTION_API,
    convert_messages,
    convert_response,
    convert_tool_choice,
    convert_tools,
)
from aidial_adapter_openai.responses.event_handler import EventHandler
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.streaming import (
    create_response_from_chunk,
    create_stage_chunk,
    map_stream,
)


def _validate_request(request: Dict[str, Any]) -> None:
    errors: List[str] = []

    if (n := request.get("n")) not in [None, 1]:
        errors.append(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    unsupported_params: List[str] = []
    for param in [
        "stop",
        "response_format",
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

    if not (request.get("messages")):
        errors.append("The request doesn't contain any messages.")

    if errors:
        raise RequestValidationError(" ".join(errors))


def _to_dict(x: BaseModel) -> dict:
    ret = x.dict()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"chat completion API response: {json.dumps(ret)}")
    return ret


async def chat_completion(
    request: Dict[str, Any],
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    supported_inputs: SupportedInputs,
) -> AsyncIterator[dict] | dict | FastAPIResponse:
    _validate_request(request)

    is_stream = bool(request.get("stream"))
    model_name = request["model"]
    messages = request["messages"]

    transformed_messages = await ResourceProcessor(
        file_storage=file_storage,
        supported_image_types=supported_inputs.input_types,
    ).transform_messages(messages)

    if isinstance(transformed_messages, DialException):
        logger.error(
            f"Failed to prepare request: {transformed_messages.message}"
        )
        chunk = None
        if (usage := supported_inputs.usage_message) is not None:
            chunk = create_stage_chunk(
                model_name=model_name,
                stage_title="Usage",
                stage_content=usage,
                stream=is_stream,
            )
        return create_response_from_chunk(
            chunk=chunk, exc=transformed_messages, stream=is_stream
        )

    input_messages = convert_messages(
        [m.raw_message for m in transformed_messages]  # type: ignore
    )

    res_tools = NOT_GIVEN
    if tools := request.get("tools"):
        res_tools = convert_tools(tools)

    res_tool_choice = NOT_GIVEN
    if tool_choice := request.get("tool_choice"):
        res_tool_choice = convert_tool_choice(tool_choice)

    response = await client.responses.create(
        model=model_name,
        stream=is_stream,
        input=input_messages,
        tools=res_tools,
        tool_choice=res_tool_choice,
        top_p=request.get("top_p") or NOT_GIVEN,
        temperature=request.get("temperature") or NOT_GIVEN,
        max_output_tokens=request.get("max_tokens") or NOT_GIVEN,
        parallel_tool_calls=request.get("parallel_tool_calls") or NOT_GIVEN,
    )

    if isinstance(response, AsyncStream):
        handler = EventHandler()
        return map_stream(_to_dict, map_stream(handler.handle, response))
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"responses API response: {json.dumps(response.dict())}"
            )
        return _to_dict(convert_response(response))
