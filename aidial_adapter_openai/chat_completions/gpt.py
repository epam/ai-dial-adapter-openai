from collections.abc import AsyncIterator, Mapping
from typing import cast

from aidial_client.types.chat import ChatCompletionRequest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
)

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.configuration.app_config import Vendor
from aidial_adapter_openai.dial_api.request import extract_max_prompt_tokens
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.providers import alibaba
from aidial_adapter_openai.utils.caching import (
    build_cache_headers,
    get_chat_completions_breakpoint_path,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
    add_statistics_to_response,
    chunk_to_dict,
    debug_print,
    generate_stream,
    map_stream,
)
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.truncate_messages import truncate_messages
from aidial_adapter_openai.utils.truncation_types import (
    DiscardedMessages,
    TruncatedTokens,
)


async def truncate_gpt_prompt(
    *,
    request: dict,
    file_storage: FileStorage | None,
    max_prompt_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[list[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
    multi_modal_messages = await ResourceProcessor(
        file_storage=file_storage
    ).transform_messages(request["messages"])
    return await truncate_messages(
        messages=multi_modal_messages,
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message.raw_message["role"]
        == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=await tokenizer.tokenize_request(request, []),
    )


async def chat_completion(
    *,
    request: dict,
    request_headers: Mapping[str, str],
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: Tokenizer,
    eliminate_empty_choices: bool,
    vendor: Vendor,
) -> ResponseWithHeaders[AsyncIterator[dict] | dict]:
    n: int = request.get("n") or 1
    model_name = request["model"]

    # Computed upfront: the path must address the request body
    # as DIAL Core has sent it, before the truncation reindexes messages.
    breakpoint_path = get_chat_completions_breakpoint_path(
        cast(CompletionCreateParamsBase, request)
    )

    max_prompt_tokens = extract_max_prompt_tokens(request)
    discarded_messages: DiscardedMessages | None

    if max_prompt_tokens is not None:
        (
            multi_modal_messages,
            discarded_messages,
            prompt_tokens,
        ) = await truncate_gpt_prompt(
            request=request,
            file_storage=file_storage,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
        )

        logger.debug(
            f"estimated prompt tokens after truncation: {prompt_tokens}, "
            f"discarded messages indices: {discarded_messages}"
        )

        async def get_prompt_tokens() -> int:
            return prompt_tokens

    else:
        multi_modal_messages = await ResourceProcessor(
            file_storage=file_storage
        ).transform_messages(request["messages"])
        discarded_messages = None

        async def get_prompt_tokens() -> int:
            prompt_tokens = await tokenizer.tokenize_request(
                request, multi_modal_messages
            )
            logger.debug(f"estimated prompt tokens: {prompt_tokens}")
            return prompt_tokens

    request["messages"] = [m.raw_message for m in multi_modal_messages]

    # The upstream-specific rewrites go last, once the request is final
    if vendor == Vendor.ALIBABA:
        alibaba.convert_chat_completions_request(
            cast(ChatCompletionRequest, request)
        )

    response: (
        AsyncStream[ChatCompletionChunk] | ChatCompletion
    ) = await call_with_extra_body(client.chat.completions.create, request)

    if isinstance(response, AsyncStream):
        response_headers = await build_cache_headers(
            request_headers=request_headers,
            breakpoint_path=breakpoint_path,
            get_request_tokens=get_prompt_tokens,
        )

        body = generate_stream(
            n=n,
            stream=map_stream(chunk_to_dict, response),
            get_prompt_tokens=get_prompt_tokens,
            tokenize_response=tokenizer.tokenize_response,
            model=model_name,
            discarded_messages=discarded_messages,
            eliminate_empty_choices=eliminate_empty_choices,
        )

        return ResponseWithHeaders(headers=response_headers, body=body)
    else:
        body = response.to_dict()
        body = add_statistics_to_response(
            body, discarded_messages=discarded_messages
        )

        actual_prompt_tokens: int | None = None
        if usage := response.usage:
            actual_prompt_tokens = usage.prompt_tokens

        async def get_request_tokens():
            return actual_prompt_tokens or await get_prompt_tokens()

        response_headers = await build_cache_headers(
            request_headers=request_headers,
            breakpoint_path=breakpoint_path,
            get_request_tokens=get_request_tokens,
        )

        debug_print("response", body)
        return ResponseWithHeaders(headers=response_headers, body=body)
