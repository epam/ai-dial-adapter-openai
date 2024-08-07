import json
from typing import Any, Callable, List

import httpx


class OpenAIStream:
    chunks: List[dict]

    def __init__(self, *chunks: dict):
        self.chunks = list(chunks)

    def to_content(self) -> str:
        ret = ""
        for chunk in self.chunks:
            ret += f"data: {json.dumps(chunk)}\n\n"
        ret += "data: [DONE]\n\n"
        return ret

    def assert_response_content(
        self,
        response: httpx.Response,
        assert_equality: Callable[[Any, Any], None],
        usages: dict[int, dict] = {},
    ):
        line_idx = 0
        for line in response.iter_lines():
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

            line_idx += 1


def chunk(
    *,
    id: str = "chatcmpl-test",
    created: str = "1695940483",
    model: str = "gpt-4",
    choices: List[dict],
    usage: dict | None = None,
    **kwargs,
) -> dict:
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
        **kwargs,
    }


def single_choice_chunk(
    *,
    id: str = "chatcmpl-test",
    created: str = "1695940483",
    model: str = "gpt-4",
    finish_reason: str | None = None,
    delta: dict = {},
    usage: dict | None = None,
    **kwargs,
) -> dict:
    return chunk(
        id=id,
        created=created,
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": finish_reason,
                "delta": delta,
            }
        ],
        usage=usage,
        **kwargs,
    )
