import logging
from time import time
from typing import Any, AsyncIterator, Callable, Iterable, Optional, TypeVar
from uuid import uuid4

from aidial_sdk.exceptions import HTTPException as DialException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import APIError, APIStatusError
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.env import get_eliminate_empty_choices
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.merge_chunks import merge_chunks
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream

ELIMINATE_EMPTY_CHOICES = get_eliminate_empty_choices()


def generate_id() -> str:
    return "chatcmpl-" + str(uuid4())


def generate_created() -> int:
    return int(time())


def build_chunk(
    id: str,
    finish_reason: Optional[str],
    message: Any,
    created: int,
    is_stream: bool,
    **extra,
) -> dict:
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

    noop_chunk = build_chunk(
        id=generate_id(),
        created=generate_created(),
        model=deployment,
        is_stream=True,
        message={},
        finish_reason=None,
    )

    def set_usage(chunk: dict | None, completions: Iterable[str]) -> dict:
        chunk = chunk or noop_chunk
        completion_tokens = sum(map(tokenize, completions))
        prompt_tokens = get_prompt_tokens()
        chunk["usage"] = {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return chunk

    def set_finish_reason(chunk: dict | None, finish_reason: str) -> dict:
        chunk = chunk or noop_chunk
        chunk["choices"] = chunk.get("choices") or [{"index": 0, "delta": {}}]
        chunk["choices"][0]["finish_reason"] = finish_reason
        return chunk

    def set_discarded_messages(chunk: dict | None, indices: list[int]) -> dict:
        chunk = chunk or noop_chunk
        chunk["statistics"] = {"discarded_messages": indices}
        return chunk

    n_chunks = 0
    last_chunk = None
    buffer_chunk = None

    completions: dict[int, str] = {}
    found_finish_reason = False
    found_usage = False
    error = None

    try:
        async for chunk in stream:
            n_chunks += 1

            if buffer_chunk is not None:
                chunk = merge_chunks(buffer_chunk, chunk)
                buffer_chunk = None

            choices = chunk.get("choices") or []

            for choice in choices:
                index = choice["index"]
                content = (choice.get("delta") or {}).get("content") or ""

                completions[index] = completions.get(index, "") + content
                found_finish_reason |= bool(choice.get("finish_reason"))

            found_usage |= bool(chunk.get("usage"))

            # Azure OpenAI returns an empty list of choices as a first chunk
            # when content filtering is enabled for a corresponding deployment.
            # The safety rating of the request is reported in this first chunk.
            # Here we withhold such a chunk and merge it later with a follow-up chunk.
            if len(choices) == 0 and ELIMINATE_EMPTY_CHOICES:
                buffer_chunk = chunk
            else:
                if last_chunk is not None:
                    yield last_chunk
                last_chunk = chunk

    except APIError as e:
        status_code = e.status_code if isinstance(e, APIStatusError) else 500
        error = DialException(
            status_code=status_code,
            message=e.message,
            type=e.type,
            param=e.param,
            code=e.code,
        ).json_error()

    if last_chunk is not None and buffer_chunk is not None:
        last_chunk = merge_chunks(buffer_chunk, last_chunk)

    if discarded_messages is not None:
        last_chunk = set_discarded_messages(last_chunk, discarded_messages)

    if not found_usage and (not error or completions):
        last_chunk = set_usage(last_chunk, completions.values())

    if not error:
        if n_chunks == 0:
            logger.warning("Received 0 chunks")
        elif not found_finish_reason:
            logger.warning("Didn't receive chunk with the finish reason")

        if not found_finish_reason:
            last_chunk = set_finish_reason(last_chunk, "length")

        if not found_usage:
            last_chunk = set_usage(last_chunk, completions.values())

    if last_chunk:
        yield last_chunk

    if error:
        yield error


def create_stage_chunk(name: str, content: str, stream: bool) -> dict:
    id = generate_id()
    created = generate_created()

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
            return exc.to_fastapi_response()
        else:
            return JSONResponse(content=chunk)

    async def generator() -> AsyncIterator[dict]:
        yield chunk
        if exc is not None:
            yield exc.json_error()

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
