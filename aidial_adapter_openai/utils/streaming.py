from time import time
from typing import Any, AsyncIterator, Optional, TypeVar
from uuid import uuid4

import tiktoken
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.sse_stream import (
    OPENAI_END_MARKER,
    format_chunk,
)
from aidial_adapter_openai.utils.tokens import calculate_prompt_tokens

END_CHUNK = format_chunk(OPENAI_END_MARKER)


def generate_id():
    return "chatcmpl-" + str(uuid4())


def build_chunk(
    id: str,
    finish_reason: Optional[str],
    delta: Any,
    created: str,
    is_stream,
    **extra,
):
    choice_content_key = "delta" if is_stream else "message"

    return {
        "id": id,
        "object": "chat.completion.chunk" if is_stream else "chat.completion",
        "created": created,
        "choices": [
            {
                "index": 0,
                choice_content_key: delta,
                "finish_reason": finish_reason,
            }
        ],
        **extra,
    }


async def generate_stream(
    messages: list[Any],
    response,
    model: str,
    deployment: str,
    discarded_messages: Optional[int],
):
    encoding = tiktoken.encoding_for_model(model)

    prompt_tokens = calculate_prompt_tokens(messages, model, encoding)

    last_chunk = None
    stream_finished = False
    try:
        total_content = ""
        async for chunk in response:
            chunk_dict = chunk.to_dict_recursive()

            if chunk_dict["choices"][0]["finish_reason"] is not None:
                stream_finished = True
                completion_tokens = len(encoding.encode(total_content))
                chunk_dict["usage"] = {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                if discarded_messages is not None:
                    chunk_dict["statistics"] = {
                        "discarded_messages": discarded_messages
                    }
            else:
                content = chunk_dict["choices"][0]["delta"].get("content") or ""

                total_content += content

            last_chunk = chunk_dict
            yield format_chunk(chunk_dict)
    except OpenAIException as e:
        yield format_chunk(e.body)
        yield END_CHUNK

        return

    if not stream_finished:
        completion_tokens = len(encoding.encode(total_content))

        if last_chunk is not None:
            logger.warning("Didn't receive chunk with the finish reason")

            last_chunk["usage"] = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            last_chunk["choices"][0]["delta"]["content"] = ""
            last_chunk["choices"][0]["delta"]["finish_reason"] = "length"

            yield format_chunk(last_chunk)
        else:
            logger.warning("Received 0 chunks")

            yield format_chunk(
                {
                    "id": generate_id(),
                    "object": "chat.completion.chunk",
                    "created": str(int(time())),
                    "model": deployment,
                    "choices": [
                        {"index": 0, "finish_reason": "length", "delta": {}}
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens,
                    },
                }
            )

    yield END_CHUNK


T = TypeVar("T")


async def prepend_to_async_iterator(
    value: T, iterator: AsyncIterator[T]
) -> AsyncIterator[T]:
    yield value
    async for item in iterator:
        yield item


NO_USAGE = {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "total_tokens": 0,
}


def create_predefined_response(content: str, stream: bool) -> Response:
    id = generate_id()
    created = str(int(time()))

    chunk = build_chunk(
        id,
        "stop",
        {"role": "assistant", "content": content},
        created,
        stream,
        usage=NO_USAGE,
    )

    if not stream:
        return JSONResponse(content=chunk)

    async def generator() -> AsyncIterator[Any]:
        yield format_chunk(chunk)
        yield END_CHUNK

    return StreamingResponse(generator(), media_type="text/event-stream")
