from time import time
from typing import Any, AsyncIterator, Callable, Optional, TypeVar
from uuid import uuid4

from aidial_sdk.utils.merge_chunks import merge
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import APIError

from aidial_adapter_openai.utils.env import get_env_bool
from aidial_adapter_openai.utils.exceptions import create_error
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.sse_stream import END_CHUNK, format_chunk
from aidial_adapter_openai.utils.tokens import Tokenizer

fix_streaming_issues_in_new_api_versions = get_env_bool(
    "FIX_STREAMING_ISSUES_IN_NEW_API_VERSIONS", False
)


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
    discarded_messages: Optional[list[int]],
) -> AsyncIterator[dict]:
    last_chunk, temp_chunk = None, None
    stream_finished = False

    try:
        total_content = ""
        async for chunk in stream:
            if len(chunk["choices"]) > 0:
                if temp_chunk is not None:
                    chunk = merge(temp_chunk, chunk)
                    temp_chunk = None

                choice = chunk["choices"][0]

                if choice["finish_reason"] is not None:
                    stream_finished = True
                    completion_tokens = tokenizer.calculate_tokens(
                        total_content
                    )
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
                    content = choice["delta"].get("content") or ""
                    total_content += content

                yield chunk
            else:
                if fix_streaming_issues_in_new_api_versions:
                    temp_chunk = chunk
                else:
                    yield chunk

            last_chunk = chunk
    except APIError as e:
        yield create_error(
            message=e.message,
            type=e.type,
            param=e.param,
            code=e.code,
        )
        return

    if not stream_finished:
        if last_chunk is not None:
            logger.warning("Didn't receive chunk with the finish reason")

            completion_tokens = tokenizer.calculate_tokens(total_content)
            last_chunk["usage"] = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            last_chunk["choices"][0]["delta"]["content"] = ""
            last_chunk["choices"][0]["delta"]["finish_reason"] = "length"

            yield last_chunk
        else:
            logger.warning("Received 0 chunks")

            id = generate_id()
            created = str(int(time()))
            is_stream = True

            yield build_chunk(
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


def create_single_message_response(message: str, stream: bool) -> Response:
    id = generate_id()
    created = str(int(time()))

    chunk = build_chunk(
        id,
        "stop",
        {"role": "assistant", "content": message},
        created,
        stream,
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


async def prepend_to_stream(
    value: T, iterator: AsyncIterator[T]
) -> AsyncIterator[T]:
    yield value
    async for item in iterator:
        yield item


async def map_stream(
    func: Callable[[T], Optional[V]], iterator: AsyncIterator[T]
) -> AsyncIterator[V]:
    async for item in iterator:
        new_item = func(item)
        if new_item is not None:
            yield new_item
