from typing import Any, AsyncIterator

from fastapi.responses import StreamingResponse
from openai import ChatCompletion

from aidial_adapter_openai.utils.sse_stream import END_CHUNK, format_chunk


async def generate_stream(
    stream,
) -> AsyncIterator[str]:
    async for chunk in stream:
        yield format_chunk(chunk.to_dict_recursive())
    yield END_CHUNK


async def chat_completion(
    data: Any,
    upstream_endpoint: str,
    api_key: str,
    is_stream: bool,
):
    if "model" in data:
        del data["model"]

    response = await ChatCompletion().acreate(
        model="azureai",
        api_key=api_key,
        api_base=upstream_endpoint,
        request_timeout=(10, 600),  # connect timeout and total timeout
        **data,
    )

    if is_stream:
        return StreamingResponse(
            generate_stream(response), media_type="text/event-stream"
        )
    else:
        return response
