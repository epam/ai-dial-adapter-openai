import json
from collections.abc import Callable
from typing import Any

import httpx

from aidial_adapter_openai.utils.streaming import (
    streaming_chunks_to_block_response,
)


class OpenAIStream:
    chunks: list[dict]

    def __init__(self, *chunks: dict):
        self.chunks = list(chunks)

    def to_content(self) -> str:
        ret = ""
        for chunk in self.chunks:
            ret += f"data: {json.dumps(chunk)}\n\n"
        ret += "data: [DONE]\n\n"
        return ret

    def to_block_response(self) -> dict:
        return streaming_chunks_to_block_response(self.chunks)

    def assert_response_content(
        self,
        response: httpx.Response,
        assert_equality: Callable[[Any, Any], None],
        usages: dict[int, dict] = {},
    ):
        for line_idx, line in enumerate(response.iter_lines()):
            chunk_idx = line_idx // 2

            if line_idx % 2 == 1:
                assert_equality(line, "")

            elif chunk_idx < len(self.chunks):
                chunk = self.chunks[chunk_idx]
                if chunk_idx in usages:
                    chunk = chunk | {"usage": usages[chunk_idx]}
                assert_equality(json.loads(line.removeprefix("data: ")), chunk)

            elif chunk_idx == len(self.chunks):
                assert_equality(line, "data: [DONE]")

            else:
                assert False


def chunk(
    *,
    id: str = "chatcmpl-test",
    object_: str = "chat.completion.chunk",
    created: int = 1695940483,
    model: str = "gpt-4",
    choices: list[dict],
    usage: dict | None = None,
    **kwargs,
) -> dict:
    return {
        "id": id,
        "object": object_,
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
        **kwargs,
    }


def single_choice_chunk(
    *,
    id: str = "chatcmpl-test",
    object_: str = "chat.completion.chunk",
    created: int = 1695940483,
    model: str = "gpt-4",
    finish_reason: str | None = None,
    delta: dict | None = None,
    logprobs: dict | None = None,
    usage: dict | None = None,
    choice_index: int = 0,
    **kwargs,
) -> dict:
    return chunk(
        id=id,
        created=created,
        model=model,
        object_=object_,
        choices=[
            create_choice(
                index=choice_index,
                finish_reason=finish_reason,
                delta=delta,
                logprobs=logprobs,
            ),
        ],
        usage=usage,
        **kwargs,
    )


def create_choice(
    *,
    index: int = 0,
    finish_reason: str | None = None,
    delta: dict | None = None,
    logprobs: dict | None = None,
    **kwargs,
) -> dict:
    return {
        "index": index,
        "finish_reason": finish_reason,
        "delta": delta or {},
        **({"logprobs": logprobs} if logprobs else {}),
        **kwargs,
    }


def many_choices_chunk(
    *,
    id: str = "chatcmpl-test",
    object_: str = "chat.completion.chunk",
    created: int = 1695940483,
    model: str = "gpt-4",
    usage: dict | None = None,
    choices: list[dict] | None = None,
    **kwargs,
) -> dict:
    return chunk(
        id=id,
        created=created,
        model=model,
        object_=object_,
        choices=choices or [],
        usage=usage,
        **kwargs,
    )
