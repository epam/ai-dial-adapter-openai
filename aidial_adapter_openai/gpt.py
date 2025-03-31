from functools import cache
from typing import AsyncIterator, List, Mapping, Tuple, cast

from aidial_sdk.exceptions import InvalidRequestError
from openai import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.caching import (
    get_response_headers_for_caching,
    get_response_prompt_tokens,
)
from aidial_adapter_openai.utils.parsers import chat_completions_parser
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    AppResponse,
    ResponseWithHeaders,
    chunk_to_dict,
    debug_print,
    generate_stream,
    map_stream,
)
from aidial_adapter_openai.utils.tokenizer import PlainTextTokenizer
from aidial_adapter_openai.utils.truncate_prompt import (
    DiscardedMessages,
    TruncatedTokens,
    truncate_prompt,
)


def plain_text_truncate_prompt(
    request: dict,
    messages: List[dict],
    max_prompt_tokens: int,
    tokenizer: PlainTextTokenizer,
) -> Tuple[List[dict], DiscardedMessages, TruncatedTokens]:
    return truncate_prompt(
        messages=messages,
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message["role"] == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=tokenizer.tokenize_request(request, []),
    )


async def gpt_chat_completion(
    request: dict,
    request_headers: Mapping[str, str],
    deployment_id: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    api_version: str,
    tokenizer: PlainTextTokenizer,
    eliminate_empty_choices: bool,
) -> AppResponse:
    discarded_messages = None
    estimated_prompt_tokens = None
    if "max_prompt_tokens" in request:
        max_prompt_tokens = request["max_prompt_tokens"]
        if not isinstance(max_prompt_tokens, int):
            raise InvalidRequestError(
                f"'{max_prompt_tokens}' is not of type 'integer' - 'max_prompt_tokens'",
            )
        if max_prompt_tokens < 1:
            raise InvalidRequestError(
                f"'{max_prompt_tokens}' is less than the minimum of 1 - 'max_prompt_tokens'",
            )
        del request["max_prompt_tokens"]

        request["messages"], discarded_messages, estimated_prompt_tokens = (
            plain_text_truncate_prompt(
                request=request,
                messages=cast(List[dict], request["messages"]),
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )

    client = chat_completions_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version}
    )
    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, request)
    )

    if isinstance(response, AsyncIterator):

        @cache
        def get_request_tokens() -> int:
            return estimated_prompt_tokens or tokenizer.tokenize_request(
                request, request["messages"]
            )

        body = generate_stream(
            stream=map_stream(chunk_to_dict, response),
            get_prompt_tokens=get_request_tokens,
            tokenize_response=tokenizer.tokenize_response,
            deployment=deployment_id,
            discarded_messages=discarded_messages,
            eliminate_empty_choices=eliminate_empty_choices,
        )

        headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=get_request_tokens,
        )

        return ResponseWithHeaders(body=body, headers=headers)
    else:
        body = response.to_dict()
        if discarded_messages is not None:
            body |= {"statistics": {"discarded_messages": discarded_messages}}
        debug_print("response", body)

        headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: get_response_prompt_tokens(body),
        )

        return ResponseWithHeaders(body=body, headers=headers)
