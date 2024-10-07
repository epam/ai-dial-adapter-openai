from typing import Any

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import chunk_to_dict, map_stream


async def chat_completion(
    data: Any, upstream_endpoint: str, creds: OpenAICreds
):
    client = AsyncOpenAI(
        base_url=upstream_endpoint,
        api_key=creds.get("api_key"),
        http_client=get_http_client(),
    )

    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, data)
    )

    if isinstance(response, AsyncStream):
        return map_stream(chunk_to_dict, response)
    else:
        return response
