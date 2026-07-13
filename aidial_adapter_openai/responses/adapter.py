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
)

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.converter import (
    _DEPRECATED_FUNCTION_API,
    chat_completions_to_responses_request,
    convert_response,
)
from aidial_adapter_openai.responses.event_handler import EventHandler
from aidial_adapter_openai.utils.log_config import logger
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


async def chat_completion(
    *,
    request: dict[str, Any],
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI,
    file_storage: FileStorage | None,
) -> AsyncIterator[dict] | dict:
    _validate_request(request)

    _, create_request = await chat_completions_to_responses_request(
        request, file_storage
    )
    response = await client.responses.create(**create_request)

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
