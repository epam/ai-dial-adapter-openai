from time import time
from typing import Any, AsyncIterator, Callable, Optional, TypeVar
from uuid import uuid4

from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.sse_stream import (
    OPENAI_END_MARKER,
    format_chunk,
)
from aidial_adapter_openai.utils.tokens import Tokenizer

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
    prompt_tokens: int,
    stream: AsyncIterator[dict],
    tokenizer: Tokenizer,
    deployment: str,
    discarded_messages: Optional[int],
):
    last_chunk = None
    stream_finished = False
    try:
        total_content = ""
        async for chunk in stream:
            choice = chunk["choices"][0]

            if choice["finish_reason"] is not None:
                stream_finished = True
                completion_tokens = tokenizer.calculate_tokens(total_content)
                chunk["usage"] = {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                if discarded_messages is not None:
                    chunk["statistics"] = {
                        "discarded_messages": discarded_messages
                    }
            else:
                content = choice["delta"].get("content", "")
                total_content += content

            last_chunk = chunk
            yield format_chunk(chunk)
    except OpenAIException as e:
        yield format_chunk(e.body)
        yield END_CHUNK

        return

    if not stream_finished:
        completion_tokens = tokenizer.calculate_tokens(total_content)

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

            id = generate_id()
            created = str(int(time()))
            is_stream = True

            yield format_chunk(
                build_chunk(
                    id,
                    "length",
                    {},
                    created,
                    is_stream,
                    model=deployment,
                    usage={
                        "completion_tokens": 0,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens,
                    },
                )
            )

    yield END_CHUNK


def create_error_response(error_message: str, stream: bool) -> Response:
    id = generate_id()
    created = str(int(time()))

    error_stage = {
        "index": 0,
        "name": "Error",
        "content": error_message,
        "status": "failed",
    }

    custom_content = {"stages": [error_stage]}

    chunk = build_chunk(
        id,
        "stop",
        {"role": "assistant", "content": "", "custom_content": custom_content},
        created,
        stream,
        usage={
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    )

    return create_response_from_chunk(chunk, stream)


def create_response_from_chunk(chunk: dict, stream: bool) -> Response:
    if not stream:
        return JSONResponse(content=chunk)

    async def generator() -> AsyncIterator[Any]:
        yield format_chunk(chunk)
        yield END_CHUNK

    return StreamingResponse(generator(), media_type="text/event-stream")


T = TypeVar("T")
V = TypeVar("V")


async def prepend_to_async_iterator(
    value: T, iterator: AsyncIterator[T]
) -> AsyncIterator[T]:
    yield value
    async for item in iterator:
        yield item


async def map_async_iterator(
    func: Callable[[T], V], iterator: AsyncIterator[T]
) -> AsyncIterator[V]:
    async for item in iterator:
        yield func(item)
