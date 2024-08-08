import logging
from time import time
from typing import Any, AsyncIterator, Callable, Optional, TypeVar
from uuid import uuid4

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.utils.errors import json_error
from aidial_sdk.utils.merge_chunks import merge
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import APIError
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.env import get_env_bool
from aidial_adapter_openai.utils.errors import dial_exception_to_json_error
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream

fix_streaming_issues_in_new_api_versions = get_env_bool(
    "FIX_STREAMING_ISSUES_IN_NEW_API_VERSIONS", False
)


def generate_id():
    return "chatcmpl-" + str(uuid4())


def build_chunk(
    id: str,
    finish_reason: Optional[str],
    message: Any,
    created: str,
    is_stream: bool,
    **extra,
):
    message_key = "delta" if is_stream else "message"
    object_name = "chat.completion.chunk" if is_stream else "chat.completion"

    return {
        "id": id,
        "object": object_name,
        "created": created,
        "choices": [
            {
                "index": 0,
                message_key: message,
                "finish_reason": finish_reason,
            }
        ],
        **extra,
    }


async def generate_stream(
    *,
    get_prompt_tokens: Callable[[], int],
    tokenize: Callable[[str], int],
    deployment: str,
    discarded_messages: Optional[list[int]],
    stream: AsyncIterator[dict],
) -> AsyncIterator[dict]:

    last_chunk, temp_chunk = None, None
    stream_finished = False
    total_content = ""

    def finalize_finish_chunk(chunk: dict) -> None:
        """
        Adding additional information to a chunk that has non-null finish_reason field.
        """

        if not chunk.get("usage"):
            completion_tokens = tokenize(total_content)
            prompt_tokens = get_prompt_tokens()
            chunk["usage"] = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        if discarded_messages is not None:
            chunk["statistics"] = {"discarded_messages": discarded_messages}

    try:
        async for chunk in stream:
            if len(chunk["choices"]) > 0:
                if temp_chunk is not None:
                    chunk = merge(temp_chunk, chunk)
                    temp_chunk = None

                choice = chunk["choices"][0]
                total_content += (choice.get("delta") or {}).get(
                    "content"
                ) or ""

                if choice["finish_reason"] is not None:
                    stream_finished = True
                    finalize_finish_chunk(chunk)

                yield chunk
            else:
                if fix_streaming_issues_in_new_api_versions:
                    temp_chunk = chunk
                else:
                    yield chunk

            last_chunk = chunk
    except APIError as e:
        yield json_error(
            message=e.message,
            type=e.type,
            param=e.param,
            code=e.code,
        )
        return

    if not stream_finished:
        if last_chunk is None:
            logger.warning("Received 0 chunks")
        else:
            logger.warning("Didn't receive chunk with the finish reason")

        last_chunk = last_chunk or {}
        id = last_chunk.get("id") or generate_id()
        created = last_chunk.get("created") or str(int(time()))
        model = last_chunk.get("model") or deployment

        finish_chunk = build_chunk(
            id=id,
            created=created,
            model=model,
            is_stream=True,
            message={},
            finish_reason="length",
        )

        finalize_finish_chunk(finish_chunk)

        yield finish_chunk


def create_stage_chunk(name: str, content: str, stream: bool) -> dict:
    id = generate_id()
    created = str(int(time()))

    stage = {
        "index": 0,
        "name": name,
        "content": content,
        "status": "completed",
    }

    custom_content = {"stages": [stage]}

    return build_chunk(
        id,
        "stop",
        {
            "role": "assistant",
            "content": "",
            "custom_content": custom_content,
        },
        created,
        stream,
        usage={
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    )


def create_response_from_chunk(
    chunk: dict, exc: DialException | None, stream: bool
) -> Response:
    if not stream:
        if exc is not None:
            return JSONResponse(
                status_code=exc.status_code,
                content=dial_exception_to_json_error(exc),
            )
        else:
            return JSONResponse(content=chunk)

    async def generator() -> AsyncIterator[dict]:
        yield chunk
        if exc is not None:
            yield dial_exception_to_json_error(exc)

    return StreamingResponse(
        to_openai_sse_stream(generator()),
        media_type="text/event-stream",
    )


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


def debug_print(title: str, chunk: dict) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{title}: {chunk}")


def chunk_to_dict(chunk: ChatCompletionChunk) -> dict:
    dict = chunk.to_dict()
    debug_print("chunk", dict)
    return dict
