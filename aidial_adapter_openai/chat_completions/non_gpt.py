from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import chunk_to_dict, map_stream


async def chat_completion(
    request: dict, client: AsyncAzureOpenAI | AsyncOpenAI
):
    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, request)
    )

    if isinstance(response, AsyncStream):
        return map_stream(chunk_to_dict, response)
    else:
        return response
