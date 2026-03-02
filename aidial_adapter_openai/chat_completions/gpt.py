from typing import AsyncIterator, Callable, Coroutine, List, Mapping, Tuple

from aidial_sdk.exceptions import InvalidRequestError
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.caching import get_response_headers_for_caching
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
    chunk_to_dict,
    debug_print,
    generate_stream,
    map_stream,
)
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.truncate_prompt import (
    DiscardedMessages,
    TruncatedTokens,
    truncate_prompt,
)
from aidial_adapter_openai.utils.vllm_tokenizer import VllmTokenizer


async def multi_modal_truncate_prompt(
    request: dict,
    messages: List[MultiModalMessage],
    max_prompt_tokens: int,
    tokenizer: Tokenizer,
) -> Tuple[List[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
    return await truncate_prompt(
        messages=messages,
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message.raw_message["role"]
        == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=await tokenizer.tokenize_request(request, []),
    )


def _extract_max_prompt_tokens(request: dict) -> int | None:
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


async def _truncate_messages(
    request: dict, messages: List[MultiModalMessage], tokenizer: Tokenizer
) -> Tuple[
    List[MultiModalMessage],
    DiscardedMessages | None,
    Callable[[], Coroutine[None, None, TruncatedTokens]],
]:
    if (max_prompt_tokens := _extract_max_prompt_tokens(request)) is not None:
        (
            messages,
            discarded_indices,
            prompt_tokens,
        ) = await multi_modal_truncate_prompt(
            request=request,
            messages=messages,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
        )

        logger.debug(
            f"estimated prompt tokens after truncation: {prompt_tokens}, "
            f"discarded messages indices: {discarded_indices}"
        )

        async def get_prompt_tokens() -> int:
            return prompt_tokens

        return messages, discarded_indices, get_prompt_tokens
    else:

        async def get_prompt_tokens() -> int:
            estimated = await tokenizer.tokenize_request(request, messages)
            logger.debug(f"estimated prompt tokens: {estimated}")
            return estimated

        return (messages, None, get_prompt_tokens)


async def _truncate_messages_vllm(
    request: dict, messages: List[MultiModalMessage], tokenizer: VllmTokenizer
) -> Tuple[List[MultiModalMessage], DiscardedMessages | None]:
    """vLLM-specific truncation: sends the full message list to the vLLM
    tokenize endpoint on each iteration instead of counting per-message."""
    if (max_prompt_tokens := _extract_max_prompt_tokens(request)) is not None:
        (
            messages,
            discarded_indices,
            prompt_tokens,
        ) = await tokenizer.truncate_prompt(
            original_request=request,
            messages=messages,
            max_prompt_tokens=max_prompt_tokens,
        )

        logger.debug(
            f"vLLM estimated prompt tokens after truncation: {prompt_tokens}, "
            f"discarded messages indices: {discarded_indices}"
        )

        return messages, discarded_indices

    return messages, None


async def chat_completion(
    *,
    request: dict,
    request_headers: Mapping[str, str],
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: Tokenizer,
    eliminate_empty_choices: bool,
) -> ResponseWithHeaders[AsyncIterator[dict] | dict]:
    n: int = request.get("n") or 1
    messages: List[dict] = request["messages"]
    model_name = request["model"]

    multi_modal_messages = await ResourceProcessor(
        file_storage=file_storage
    ).transform_messages(messages)

    (
        multi_modal_messages,
        discarded_messages,
        get_prompt_tokens,
    ) = await _truncate_messages(request, multi_modal_messages, tokenizer)

    request["messages"] = [m.raw_message for m in multi_modal_messages]

    response: (
        AsyncStream[ChatCompletionChunk] | ChatCompletion
    ) = await call_with_extra_body(client.chat.completions.create, request)

    if isinstance(response, AsyncStream):
        response_headers = await get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
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
        if discarded_messages is not None:
            body |= {"statistics": {"discarded_messages": discarded_messages}}

        actual_prompt_tokens: int | None = None
        if usage := response.usage:
            actual_prompt_tokens = usage.prompt_tokens

        async def get_request_tokens():
            return actual_prompt_tokens or await get_prompt_tokens()

        response_headers = await get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=get_request_tokens,
        )

        debug_print("response", body)
        return ResponseWithHeaders(headers=response_headers, body=body)


async def vllm_chat_completion(
    *,
    request: dict,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: VllmTokenizer,
    eliminate_empty_choices: bool,
) -> ResponseWithHeaders[AsyncIterator[dict] | dict]:
    """Chat completion flow for vLLM deployments.

    Key differences from the standard GPT flow:
    - Truncation sends the full message list to the vLLM tokenize endpoint
      each iteration (no per-message token counting).
    - No adapter-side response tokenization or caching headers.
    - The request is proxied transparently to vLLM — usage handling
      is entirely between the client and the upstream server.

    Notes:
    - `eliminate_empty_choices` is still applied on the adapter side to keep
      behavior consistent with the standard GPT streaming implementation.
    """
    messages: List[dict] = request["messages"]

    multi_modal_messages = await ResourceProcessor(
        file_storage=file_storage
    ).transform_messages(messages)

    (
        multi_modal_messages,
        discarded_messages,
    ) = await _truncate_messages_vllm(request, multi_modal_messages, tokenizer)

    request["messages"] = [m.raw_message for m in multi_modal_messages]

    # vLLM: guarantee usage stats in streaming responses.
    # For streaming calls, vLLM includes usage only if requested via
    # stream_options.include_usage.
    if request.get("stream"):
        stream_options = request.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options["include_usage"] = True
        request["stream_options"] = stream_options

    response: (
        AsyncStream[ChatCompletionChunk] | ChatCompletion
    ) = await call_with_extra_body(client.chat.completions.create, request)

    if isinstance(response, AsyncStream):
        body = _vllm_generate_stream(
            stream=map_stream(chunk_to_dict, response),
            discarded_messages=discarded_messages,
            eliminate_empty_choices=eliminate_empty_choices,
        )
        return ResponseWithHeaders(headers=None, body=body)
    else:
        body = response.to_dict()
        if eliminate_empty_choices and isinstance(body.get("choices"), list):
            body["choices"] = [c for c in body["choices"] if c]

        if discarded_messages is not None:
            body |= {"statistics": {"discarded_messages": discarded_messages}}

        debug_print("response", body)
        return ResponseWithHeaders(headers=None, body=body)


async def _vllm_generate_stream(
    *,
    stream: AsyncIterator[dict],
    discarded_messages: DiscardedMessages | None,
    eliminate_empty_choices: bool,
) -> AsyncIterator[dict]:
    """Pass through streaming chunks from vLLM, injecting
    ``discarded_messages`` statistics into the last chunk if needed.

    If `eliminate_empty_choices` is True, empty chunk choices are removed
    (same intent as the standard GPT stream generator).
    """
    last_chunk = None

    async for chunk in stream:
        if eliminate_empty_choices and isinstance(chunk.get("choices"), list):
            chunk["choices"] = [c for c in chunk["choices"] if c]

        if last_chunk is not None:
            yield last_chunk
        last_chunk = chunk

    if last_chunk is not None:
        if discarded_messages is not None:
            last_chunk["statistics"] = {
                "discarded_messages": discarded_messages
            }
        yield last_chunk
