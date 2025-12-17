import logging
from time import time
from typing import (
    AsyncIterator,
    Callable,
    Coroutine,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
from uuid import uuid4

from aidial_sdk.utils.merge_chunks import (
    cleanup_indices,
    merge_chat_completion_chunks,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import BaseModel

from aidial_adapter_openai.exception_handlers import to_adapter_exception
from aidial_adapter_openai.utils.adapter_exception import AdapterException
from aidial_adapter_openai.utils.chat_completion_response import (
    ChatCompletionResponse,
    ChatCompletionStreamingChunk,
)
from aidial_adapter_openai.utils.json import remove_nones
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream


def generate_id() -> str:
    return "chatcmpl-" + str(uuid4())


def generate_created() -> int:
    return int(time())


def build_chunk(
    *,
    id: str,
    model: str,
    finish_reason: Optional[str],
    message: dict | List[dict],
    created: int,
    is_stream: bool,
    **extra,
) -> dict:
    message_key = "delta" if is_stream else "message"
    object_name = "chat.completion.chunk" if is_stream else "chat.completion"

    messages = [message] if isinstance(message, dict) else message

    choices = [
        {"index": index, message_key: msg, "finish_reason": finish_reason}
        for (index, msg) in enumerate(messages)
    ]

    return {
        "id": id,
        "model": model,
        "object": object_name,
        "created": created,
        "choices": choices,
        **remove_nones(extra),
    }


async def generate_stream(
    *,
    n: int,
    stream: AsyncIterator[dict],
    get_prompt_tokens: Callable[[], Coroutine[None, None, int]],
    tokenize_response: Callable[
        [ChatCompletionResponse], Coroutine[None, None, int]
    ],
    model: str,
    discarded_messages: Optional[list[int]],
    eliminate_empty_choices: bool,
) -> AsyncIterator[dict]:

    empty_chunk = build_chunk(
        id=generate_id(),
        created=generate_created(),
        model=model,
        is_stream=True,
        message={},
        finish_reason=None,
    )

    async def set_usage(
        chunk: dict | None, resp: ChatCompletionResponse
    ) -> dict:
        chunk = chunk or empty_chunk

        # Do not fail the whole response if tokenization has failed
        try:
            completion_tokens = await tokenize_response(resp)
            prompt_tokens = await get_prompt_tokens()
        except Exception as e:
            logger.exception(
                f"caught exception while tokenization: {type(e).__module__}.{type(e).__name__}. "
                "The tokenization has failed, therefore, the usage won't be reported."
            )
        else:
            chunk["usage"] = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return chunk

    def set_default_finish_reasons(
        chunk: dict | None,
        missing_indices: Set[int],
        default_finish_reason: str,
    ) -> dict:
        def _set_reason(choice: dict) -> dict:
            if choice.get("finish_reason") is None:
                choice["finish_reason"] = default_finish_reason
            return choice

        chunk = chunk or empty_chunk
        choices = chunk.setdefault("choices", [])

        for choice in choices:
            index = choice.get("index")
            if index in missing_indices:
                missing_indices.discard(index)
                _set_reason(choice)

        for index in sorted(missing_indices):
            choice = {"index": index, "delta": {}}
            choices.append(_set_reason(choice))

        return chunk

    def set_discarded_messages(chunk: dict | None, indices: list[int]) -> dict:
        chunk = chunk or empty_chunk
        chunk["statistics"] = {"discarded_messages": indices}
        return chunk

    last_chunk = None
    buffer_chunk = None
    response_snapshot = ChatCompletionStreamingChunk(response={})

    error: Exception | None = None

    try:
        async for chunk in stream:
            response_snapshot.merge(chunk)

            if buffer_chunk is not None:
                chunk = merge_chat_completion_chunks(buffer_chunk, chunk)
                buffer_chunk = None

            choices = chunk.get("choices") or []

            # Azure OpenAI returns an empty list of choices as a first chunk
            # when content filtering is enabled for a corresponding deployment.
            # The safety rating of the request is reported in this first chunk.
            # Here we withhold such a chunk and merge it later with a follow-up chunk.
            if len(choices) == 0 and eliminate_empty_choices:
                buffer_chunk = chunk
            else:
                if last_chunk is not None:
                    yield last_chunk
                last_chunk = chunk

    except Exception as e:
        logger.exception(
            f"caught exception while streaming: {type(e).__module__}.{type(e).__name__}"
        )
        error = e

    if last_chunk is not None and buffer_chunk is not None:
        last_chunk = merge_chat_completion_chunks(last_chunk, buffer_chunk)

    if discarded_messages is not None:
        last_chunk = set_discarded_messages(last_chunk, discarded_messages)

    if response_snapshot.usage is None and (
        not error or response_snapshot.has_messages
    ):
        last_chunk = await set_usage(last_chunk, response_snapshot)

    if not error:
        missing_finish_reasons = response_snapshot.get_missing_finish_reasons(n)

        if response_snapshot.is_empty:
            logger.warning("Received 0 chunks")
        elif missing_finish_reasons:
            logger.warning(
                "Didn't receive finish reasons for all completion choices"
            )

        if missing_finish_reasons:
            last_chunk = set_default_finish_reasons(
                last_chunk, missing_finish_reasons, "length"
            )

    if last_chunk:
        yield last_chunk

    if error:
        raise error


def block_response_to_streaming_chunk(response: dict) -> dict:
    response["object"] = "chat.completion.chunk"
    for choice in response.get("choices") or []:
        if message := choice.get("message"):
            choice["delta"] = message
            del choice["message"]
    return response


def streaming_chunks_to_block_response(chunks: List[dict]) -> dict:
    response = merge_chat_completion_chunks(*chunks)
    response["object"] = "chat.completion"

    for choice in response.get("choices") or []:
        if delta := choice.get("delta"):
            choice["message"] = cleanup_indices(delta)
            del choice["delta"]
    return response


_Body = TypeVar("_Body")


class ResponseWithHeaders(Generic[_Body], BaseModel):
    headers: dict[str, str] | None = None
    body: _Body


_BaseResponse = AsyncIterator[dict] | dict | BaseModel
AppResponse = ResponseWithHeaders[_BaseResponse] | _BaseResponse | Response


async def create_server_response(
    emulate_streaming: bool, response: AppResponse
) -> Response:
    if isinstance(response, ResponseWithHeaders):
        body = response.body
        headers = response.headers or {}
    else:
        body = response
        headers = {}

    def block_to_stream(block: dict) -> AsyncIterator[dict]:
        async def stream():
            yield block_response_to_streaming_chunk(block)

        return stream()

    async def stream_to_response(iterator: AsyncIterator[dict]) -> Response:
        item, stream = await peek_head(reify_exceptions(iterator))

        if isinstance(item, Exception):
            return item.to_fastapi_response()
        else:
            content = to_openai_sse_stream(prepend(item, stream))
            return StreamingResponse(
                content=content,
                media_type="text/event-stream",
                headers=headers,
            )

    async def block_to_response(block: dict) -> Response:
        if emulate_streaming:
            return await stream_to_response(block_to_stream(block))
        else:
            return JSONResponse(block, headers=headers)

    if isinstance(body, AsyncIterator):
        return await stream_to_response(body)

    if isinstance(body, dict):
        return await block_to_response(body)

    if isinstance(body, BaseModel):
        return await block_to_response(body.dict())

    return body


T = TypeVar("T")
V = TypeVar("V")


async def map_stream_generator(
    func: Callable[[T], Generator[V, None, None]], iterator: AsyncIterator[T]
) -> AsyncIterator[V]:
    async for item in iterator:
        for new_item in func(item):
            yield new_item


async def map_stream(
    func: Callable[[T], Optional[V]], iterator: AsyncIterator[T]
) -> AsyncIterator[V]:
    async for item in iterator:
        new_item = func(item)
        if new_item is not None:
            yield new_item


async def prepend(
    value: T | None, iterator: AsyncIterator[T]
) -> AsyncIterator[T]:
    if value is not None:
        yield value
    async for item in iterator:
        yield item


async def peek_head(
    iterator: AsyncIterator[T],
) -> Tuple[T | None, AsyncIterator[T]]:
    try:
        val = await iterator.__anext__()
        return val, iterator
    except StopAsyncIteration:
        return None, iterator


async def reify_exceptions(
    stream: AsyncIterator[dict],
) -> AsyncIterator[dict | AdapterException]:
    try:
        async for chunk in stream:
            yield chunk
    except Exception as e:
        adapter_exception = to_adapter_exception(e)

        logger.exception(
            f"Caught exception while streaming: {type(e).__module__}.{type(e).__name__}. "
            f"Converted to the adapter exception: {adapter_exception!r}"
        )

        yield adapter_exception


def debug_print(title: str, chunk: dict) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{title}: {chunk}")


def chunk_to_dict(chunk: ChatCompletionChunk) -> dict:
    dict = chunk.to_dict()
    debug_print("chunk", dict)
    return dict
