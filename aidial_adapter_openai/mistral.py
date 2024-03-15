from typing import Any, AsyncIterator

from fastapi.responses import Response, StreamingResponse
from openai import ChatCompletion

from aidial_adapter_openai.utils.sse_stream import END_CHUNK, format_chunk


async def generate_stream(
    stream: AsyncIterator[dict],
) -> AsyncIterator[str]:
    async for chunk in stream:
        yield format_chunk(chunk.to_dict_recursive())
    yield END_CHUNK


async def chat_completion(
    data: Any,
    upstream_endpoint: str,
    api_key: str,
    is_stream: bool,
) -> Response:
    response = await ChatCompletion().acreate(
        model="azureai",
        api_key="-",
        api_base=upstream_endpoint,
        request_timeout=(10, 600),  # connect timeout and total timeout
        headers={"Authorization": f"Bearer {api_key}"},
        **data,
    )

    if is_stream == False:
        return response
    else:
        return StreamingResponse(
            generate_stream(response), media_type="text/event-stream"
        )
