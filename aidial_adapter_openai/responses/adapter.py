import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, assert_never, cast

from aidial_sdk.exceptions import RequestValidationError
from openai import (
    AsyncAzureOpenAI,
    AsyncBedrockOpenAI,
    AsyncOpenAI,
    AsyncStream,
    BaseModel,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
)

from aidial_adapter_openai.configuration.app_config import Vendor
from aidial_adapter_openai.dial_api.request import extract_max_prompt_tokens
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.converter import (
    _DEPRECATED_FUNCTION_API,
    chat_completions_to_responses_request,
    convert_response,
)
from aidial_adapter_openai.responses.event_handler import EventHandler
from aidial_adapter_openai.responses.tokenizer import ResponsesRequestTokenizer
from aidial_adapter_openai.utils.caching import (
    build_cache_headers,
    get_chat_completions_breakpoint_path,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
    add_statistics_to_response,
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


async def _truncate_prompt(
    max_prompt_tokens: int,
    request: dict[str, Any],
    vendor: Vendor,
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI,
    file_storage: FileStorage | None,
) -> DiscardedMessages | None:
    match vendor:
        case Vendor.OPENAI_PLATFORM:
            tokenizer = ResponsesRequestTokenizer(client, file_storage)
        case Vendor.AWS | Vendor.AZURE | Vendor.ALIBABA:
            logger.warning(
                "max_prompt_tokens is ignored for Responses API deployments "
                "backed by Azure OpenAI, Amazon Bedrock or Alibaba Cloud "
                "Model Studio, because the upstream doesn't support "
                "responses/input_tokens."
            )
            return None
        case Vendor.VLLM:
            raise ValueError(
                "Unexpected vendor backed by Responses API - VLLM."
            )
        case _:
            assert_never(vendor)

    (
        messages,
        discarded_messages,
        prompt_tokens,
    ) = await truncate_prompt(
        tokenizer=tokenizer,
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
    request_headers: Mapping[str, str],
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI,
    file_storage: FileStorage | None,
    vendor: Vendor,
) -> ResponseWithHeaders[AsyncIterator[dict] | dict]:
    _validate_request(request)

    # Computed upfront: the path must address the request body
    # as DIAL Core has sent it, before the truncation reindexes messages.
    response_headers = await build_cache_headers(
        request_headers=request_headers,
        breakpoint_path=get_chat_completions_breakpoint_path(
            cast(CompletionCreateParamsBase, request)
        ),
    )

    discarded_messages = None
    if (max_prompt_tokens := extract_max_prompt_tokens(request)) is not None:
        discarded_messages = await _truncate_prompt(
            max_prompt_tokens=max_prompt_tokens,
            request=request,
            vendor=vendor,
            client=client,
            file_storage=file_storage,
        )

    _, create_request = await chat_completions_to_responses_request(
        request, file_storage
    )
    response = await client.responses.create(**create_request)

    body: AsyncIterator[dict] | dict
    if isinstance(response, AsyncStream):
        handler = EventHandler()
        body = map_stream(
            _to_dict, map_stream_generator(handler.handle, response)
        )
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"responses API response: {json.dumps(response.model_dump())}"
            )
        body = _to_dict(convert_response(response))

    return ResponseWithHeaders(
        headers=response_headers,
        body=add_statistics_to_response(
            body, discarded_messages=discarded_messages
        ),
    )
