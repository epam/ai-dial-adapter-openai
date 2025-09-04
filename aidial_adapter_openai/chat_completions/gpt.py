from typing import AsyncIterator, Callable, List, Mapping, Tuple

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


def multi_modal_truncate_prompt(
    request: dict,
    messages: List[MultiModalMessage],
    max_prompt_tokens: int,
    tokenizer: Tokenizer,
) -> Tuple[List[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
    return truncate_prompt(
        messages=messages,
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message.raw_message["role"]
        == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=tokenizer.tokenize_request(request, []),
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


def _truncate_messages(
    request: dict, messages: List[MultiModalMessage], tokenizer: Tokenizer
) -> Tuple[
    List[MultiModalMessage],
    DiscardedMessages | None,
    Callable[[], TruncatedTokens],
]:
    if (max_prompt_tokens := _extract_max_prompt_tokens(request)) is not None:
        messages, discarded_indices, prompt_tokens = (
            multi_modal_truncate_prompt(
                request=request,
                messages=messages,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )

        logger.debug(
            f"estimated prompt tokens after truncation: {prompt_tokens}, "
            f"discarded messages indices: {discarded_indices}"
        )

        return messages, discarded_indices, lambda: prompt_tokens
    else:

        def get_prompt_tokens() -> int:
            prompt_tokens = tokenizer.tokenize_request(request, messages)
            logger.debug(f"estimated prompt tokens: {prompt_tokens}")
            return prompt_tokens

        return (messages, None, get_prompt_tokens)


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

    multi_modal_messages, discarded_messages, get_prompt_tokens = (
        _truncate_messages(request, multi_modal_messages, tokenizer)
    )

    request["messages"] = [m.raw_message for m in multi_modal_messages]

    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, request)
    )

    if isinstance(response, AsyncStream):
        response_headers = get_response_headers_for_caching(
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

        response_headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: actual_prompt_tokens
            or get_prompt_tokens(),
        )

        debug_print("response", body)
        return ResponseWithHeaders(headers=response_headers, body=body)
