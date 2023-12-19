import json
from time import time
from typing import Any, AsyncIterator, Mapping, Optional
from uuid import uuid4

import tiktoken

from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.exceptions import create_error
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.tokens import calculate_prompt_tokens

END_MARKER = "[DONE]"
CHUNK_PREFIX = "data: "


def chunk_format(data: str | Mapping[str, Any]):
    if type(data) == str:
        return CHUNK_PREFIX + data.strip() + "\n\n"
    else:
        return CHUNK_PREFIX + json.dumps(data, separators=(",", ":")) + "\n\n"


END_CHUNK = chunk_format(END_MARKER)


def generate_id():
    return "chatcmpl-" + str(uuid4())


async def parse_sse_stream(stream: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    async for line in stream:
        try:
            payload = line.decode("utf-8-sig").lstrip()  # type: ignore
        except Exception:
            yield create_error(
                message="Can't decode streaming chunk", type="runtime_error"
            )
            return

        if payload.strip() == "":
            continue

        if not payload.startswith(CHUNK_PREFIX):
            yield create_error(
                message="Invalid chunk format", type="runtime_error"
            )
            return

        payload = payload[len(CHUNK_PREFIX) :]

        if payload.strip() == END_MARKER:
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            yield create_error(
                message="Can't parse JSON chunk", type="runtime_error"
            )
            return

        logger.debug(f"incoming chunk: {chunk}")
        yield chunk


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
                "finish_reason": finish_reason,
                choice_content_key: delta,
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
            yield chunk_format(chunk_dict)
    except OpenAIException as e:
        yield chunk_format(e.body)
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

            yield chunk_format(last_chunk)
        else:
            logger.warning("Received 0 chunks")

            yield chunk_format(
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
