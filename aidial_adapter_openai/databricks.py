from typing import Any, cast

from openai import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import (
    OpenAIParams,
    chat_completions_parser,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    chunk_to_dict,
    create_server_response,
    map_stream,
)


async def chat_completion(
    data: Any, upstream_endpoint: str, creds: OpenAICreds
):
    client = chat_completions_parser.parse(upstream_endpoint).get_client(
        cast(OpenAIParams, creds)
    )

    upstream_response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, data)
    )

    if isinstance(upstream_response, AsyncStream):
        response = map_stream(chunk_to_dict, upstream_response)
    else:
        response = upstream_response

    return create_server_response(response)
