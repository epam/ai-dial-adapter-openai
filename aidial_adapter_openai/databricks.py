from typing import Any, cast

from fastapi.responses import StreamingResponse
from openai import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import (
    OpenAIParams,
    chat_completions_parser,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.streaming import map_stream


def debug_print(chunk):
    logger.debug(f"chunk: {chunk}")
    return chunk


async def chat_completion(
    data: Any, upstream_endpoint: str, creds: OpenAICreds
):
    client = chat_completions_parser.parse(upstream_endpoint).get_client(
        cast(OpenAIParams, creds)
    )

    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, data)
    )

    if isinstance(response, AsyncStream):
        return StreamingResponse(
            to_openai_sse_stream(
                map_stream(
                    debug_print,
                    map_stream(lambda chunk: chunk.to_dict(), response),
                )
            ),
            media_type="text/event-stream",
        )
    else:
        return response
