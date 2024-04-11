from typing import Any, AsyncGenerator, AsyncIterator, cast

from fastapi.responses import StreamingResponse
from openai import ChatCompletion
from openai.openai_object import OpenAIObject

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import chat_completions_parser
from aidial_adapter_openai.utils.sse_stream import (
    END_CHUNK,
    format_chunk,
    to_openai_sse_stream,
)
from aidial_adapter_openai.utils.streaming import map_stream


async def generate_stream(
    stream,
) -> AsyncIterator[str]:
    async for chunk in stream:
        yield format_chunk(chunk.to_dict_recursive())
    yield END_CHUNK


def debug_print(chunk):
    logger.debug(f"chunk: {chunk}")
    return chunk


async def chat_completion(
    data: Any, deployment_id: str, upstream_endpoint: str, api_key: str
):
    request_args = chat_completions_parser.parse(
        upstream_endpoint
    ).prepare_request_args(deployment_id)

    response = await ChatCompletion().acreate(
        **request_args,
        api_key=api_key,
        request_timeout=DEFAULT_TIMEOUT,
        **data,
    )

    if isinstance(response, AsyncGenerator):
        response = cast(AsyncGenerator[OpenAIObject, None], response)

        return StreamingResponse(
            to_openai_sse_stream(
                map_stream(
                    debug_print,
                    map_stream(
                        lambda chunk: chunk.to_dict_recursive(), response
                    ),
                )
            ),
            media_type="text/event-stream",
        )
    else:
        return response
