from typing import List, Tuple, cast

from aidial_sdk.exceptions import InvalidRequestError
from fastapi.responses import StreamingResponse
from openai import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import chat_completions_parser
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.streaming import (
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


def plain_text_truncate_prompt(
    messages: List[dict], max_prompt_tokens: int, tokenizer: Tokenizer
) -> Tuple[List[dict], DiscardedMessages, TruncatedTokens]:
    return truncate_prompt(
        message_holders=messages,
        message_tokens_getter=tokenizer.calculate_tokens_per_message,
        is_system_message_getter=lambda message: message["role"] == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=tokenizer.TOKENS_PER_REQUEST,
    )


async def gpt_chat_completion(
    data: dict,
    deployment_id: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    api_version: str,
    tokenizer: Tokenizer,
):
    discarded_messages = None
    prompt_tokens = None
    if "max_prompt_tokens" in data:
        max_prompt_tokens = data["max_prompt_tokens"]
        if not isinstance(max_prompt_tokens, int):
            raise InvalidRequestError(
                f"'{max_prompt_tokens}' is not of type 'integer' - 'max_prompt_tokens'",
            )
        if max_prompt_tokens < 1:
            raise InvalidRequestError(
                f"'{max_prompt_tokens}' is less than the minimum of 1 - 'max_prompt_tokens'",
            )
        del data["max_prompt_tokens"]

        data["messages"], discarded_messages, prompt_tokens = (
            plain_text_truncate_prompt(
                messages=cast(List[dict], data["messages"]),
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )

    client = chat_completions_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version}
    )
    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, data)
    )

    if isinstance(response, AsyncStream):
        return StreamingResponse(
            to_openai_sse_stream(
                generate_stream(
                    get_prompt_tokens=lambda: prompt_tokens
                    or tokenizer.calculate_prompt_tokens(data["messages"]),
                    tokenize=tokenizer.calculate_tokens,
                    deployment=deployment_id,
                    discarded_messages=discarded_messages,
                    stream=map_stream(chunk_to_dict, response),
                ),
            ),
            media_type="text/event-stream",
        )
    else:
        resp = response.to_dict()
        if discarded_messages is not None:
            resp |= {"statistics": {"discarded_messages": discarded_messages}}
        debug_print("response", resp)
        return resp
