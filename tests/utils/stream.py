import json
from typing import List

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
        self, response: httpx.Response, usages: dict[int, dict] = {}
    ):
        line_idx = 0
        for line in response.iter_lines():
            chunk_idx = line_idx // 2

            if line_idx % 2 == 1:
                assert line == ""

            elif chunk_idx < len(self.chunks):
                chunk = self.chunks[chunk_idx]
                if chunk_idx in usages:
                    chunk = chunk | {"usage": usages[chunk_idx]}
                assert json.loads(line.removeprefix("data: ")) == chunk

            elif chunk_idx == len(self.chunks):
                assert line == "data: [DONE]"

            else:
                assert False

            line_idx += 1
