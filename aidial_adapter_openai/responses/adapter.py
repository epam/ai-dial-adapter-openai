import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from aidial_sdk.exceptions import InvalidRequestError, RequestValidationError
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
from aidial_adapter_openai.responses.tokenizer import (
    ResponsesTokenizer,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.streaming import (
    map_stream,
    map_stream_generator,
)
from aidial_adapter_openai.utils.truncate_prompt import truncate_prompt
from aidial_adapter_openai.utils.truncation_types import (
    DiscardedMessages,
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


def _extract_max_prompt_tokens(request: dict[str, Any]) -> int | None:
    if (max_prompt_tokens := request.pop("max_prompt_tokens", None)) is None:
        return None

    if not isinstance(max_prompt_tokens, int):
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is not of type 'integer'",
            param="max_prompt_tokens",
        )

    if max_prompt_tokens < 1:
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is less than the minimum of 1",
            param="max_prompt_tokens",
        )

    return max_prompt_tokens


async def _truncate_prompt(
    max_prompt_tokens: int,
    request: dict[str, Any],
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI,
    file_storage: FileStorage | None,
) -> DiscardedMessages | None:
    (
        messages,
        discarded_messages,
        prompt_tokens,
    ) = await truncate_prompt(
        tokenizer=ResponsesTokenizer(client=client, file_storage=file_storage),
        original_request=request,
        messages=request["messages"],
        get_raw_message=lambda m: m,
        max_prompt_tokens=max_prompt_tokens,
    )
    request["messages"] = messages
    logger.debug(
        f"Responses estimated prompt tokens after truncation: {prompt_tokens}, "
        f"discarded messages indices: {discarded_messages}"
    )
    return discarded_messages


async def chat_completion(
    *,
    request: dict[str, Any],
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI,
    file_storage: FileStorage | None,
) -> AsyncIterator[dict] | dict:
    _validate_request(request)

    discarded_messages = None
    if (max_prompt_tokens := _extract_max_prompt_tokens(request)) is not None:
        discarded_messages = await _truncate_prompt(
            max_prompt_tokens=max_prompt_tokens,
            request=request,
            client=client,
            file_storage=file_storage,
        )

    _, create_request = await chat_completions_to_responses_request(
        request, file_storage
    )
    response = await client.responses.create(**create_request)

    if isinstance(response, AsyncStream):
        handler = EventHandler()
        stream = map_stream(
            _to_dict, map_stream_generator(handler.handle, response)
        )
        return _generate_stream(
            stream=stream, discarded_messages=discarded_messages
        )
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"responses API response: {json.dumps(response.model_dump())}"
            )
        body = _to_dict(convert_response(response))
        if discarded_messages is not None:
            body |= {"statistics": {"discarded_messages": discarded_messages}}
        return body


async def _generate_stream(
    *,
    stream: AsyncIterator[dict],
    discarded_messages: DiscardedMessages | None,
) -> AsyncIterator[dict]:
    last_chunk = None

    async for chunk in stream:
        if last_chunk is not None:
            yield last_chunk
        last_chunk = chunk

    if last_chunk is not None:
        if discarded_messages is not None:
            last_chunk["statistics"] = {
                "discarded_messages": discarded_messages
            }
        yield last_chunk
