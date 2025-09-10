from typing import AsyncIterator, List, Tuple, cast

from aidial_sdk.exceptions import InvalidRequestError
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
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


async def plain_text_truncate_prompt(
    request: dict,
    messages: List[dict],
    max_prompt_tokens: int,
    tokenizer: PlainTextTokenizer,
) -> Tuple[List[dict], DiscardedMessages, TruncatedTokens]:
    return await truncate_prompt(
        messages=messages,
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message["role"] == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=await tokenizer.tokenize_request(request, []),
    )


async def gpt_chat_completion(
    request: dict,
    deployment_id: str,
    endpoint: AzureOpenAIEndpoint | OpenAIEndpoint,
    creds: OpenAICreds,
    api_version: str,
    tokenizer: PlainTextTokenizer,
    eliminate_empty_choices: bool,
):
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
            await plain_text_truncate_prompt(
                request=request,
                messages=cast(List[dict], request["messages"]),
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )

    client = endpoint.get_client({**creds, "api_version": api_version})
    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, request)
    )

    if isinstance(response, AsyncIterator):

        async def get_prompt_tokens():
            return estimated_prompt_tokens or await tokenizer.tokenize_request(
                request, request["messages"]
            )

        return generate_stream(
            stream=map_stream(chunk_to_dict, response),
            get_prompt_tokens=get_prompt_tokens,
            tokenize_response=tokenizer.tokenize_response,
            deployment=deployment_id,
            discarded_messages=discarded_messages,
            eliminate_empty_choices=eliminate_empty_choices,
        )
    else:
        body = response.to_dict()
        if discarded_messages is not None:
            body |= {"statistics": {"discarded_messages": discarded_messages}}
        debug_print("response", body)
        return body
