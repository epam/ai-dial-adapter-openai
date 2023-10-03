import json
from time import time
from typing import Any, Mapping
from uuid import uuid4

import tiktoken

from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.tokens import calculate_prompt_tokens


def chunk_format(data: str | Mapping[str, Any]):
    if type(data) == str:
        return "data: " + data.strip() + "\n\n"
    else:
        return "data: " + json.dumps(data, separators=(",", ":")) + "\n\n"


END_CHUNK = chunk_format("[DONE]")


async def generate_stream(
    messages: list[Any], response, model: str, deployment: str
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
            else:
                total_content += chunk_dict["choices"][0]["delta"].get(
                    "content", ""
                )

            last_chunk = chunk_dict
            yield chunk_format(chunk_dict)
    except OpenAIException as e:
        yield chunk_format(e.body)
        yield END_CHUNK

        return

    if not stream_finished:
        completion_tokens = len(encoding.encode(total_content))

        if last_chunk is not None:
            logger.warning("Don't received chunk with the finish reason")

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
                    "id": "chatcmpl-" + str(uuid4()),
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
