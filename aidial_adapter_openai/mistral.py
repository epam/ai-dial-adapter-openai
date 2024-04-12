from typing import Any, AsyncIterator, cast

from fastapi.responses import StreamingResponse
from openai import ChatCompletion
from openai.openai_object import OpenAIObject

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.utils.sse_stream import END_CHUNK, format_chunk


async def generate_stream(
    stream: AsyncIterator[OpenAIObject],
) -> AsyncIterator[str]:
    async for chunk in stream:
        yield format_chunk(chunk.to_dict_recursive())
    yield END_CHUNK


async def chat_completion(
    data: Any,
    upstream_endpoint: str,
    api_key: str,
    api_type: str,
):
    data["model"] = "azureai"

    response = await ChatCompletion().acreate(
        api_key=api_key,
        api_base=upstream_endpoint,
        api_type=api_type,
        request_timeout=DEFAULT_TIMEOUT,
        **data,
    )

    if isinstance(response, AsyncIterator):
        response = cast(AsyncIterator[OpenAIObject], response)

        return StreamingResponse(
            generate_stream(response), media_type="text/event-stream"
        )
    else:
        return response
