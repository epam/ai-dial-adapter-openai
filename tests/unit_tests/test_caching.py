import dataclasses
import json
from typing import List

import httpx
import pytest
import respx

from tests.utils.stream import OpenAIStream, single_choice_chunk


def mock_response(upstream_url: str, stream: bool, chunks: List[dict]):
    mock_stream = OpenAIStream(*chunks)
    if stream:
        respx.post(upstream_url).respond(
            status_code=200,
            content_type="text/event-stream",
            content=mock_stream.to_content(),
        )
    else:
        respx.post(upstream_url).respond(
            status_code=200,
            content_type="application/json",
            content=json.dumps(mock_stream.to_block_response()),
        )


@dataclasses.dataclass
class TestCase:
    __test__ = False

    stream: bool
    request_content: str
    request_usage: dict | None
    caching_enabled: bool

    expected_caching_response: bool

    def get_name(self):
        xs = []
        if self.stream:
            xs.append("stream+")
        else:
            xs.append("stream-")

        if len(self.request_content) > 1024:
            xs.append(">=1024")
        else:
            xs.append("<1024")

        if self.caching_enabled:
            xs.append("caching+")
        else:
            xs.append("caching-")

        if self.request_usage:
            xs.append("usage+")
        else:
            xs.append("usage-")

        return "/".join(xs)


token_threshold = 1024
big_content = "cat " * 1512  # #tokens >= token_threshold
small_content = "cat"  # #tokens < token_threshold

big_usage = {"prompt_tokens": token_threshold}
small_usage = {"prompt_tokens": token_threshold - 1}


@respx.mock
@pytest.mark.parametrize(
    "ts",
    [
        ts
        for stream in [True, False]
        for ts in [
            TestCase(stream, big_content, None, True, True),
            TestCase(stream, big_content, None, False, False),
            TestCase(stream, small_content, None, True, False),
            TestCase(stream, small_content, None, False, False),
            TestCase(stream, big_content, small_usage, True, stream),
            TestCase(stream, big_content, small_usage, False, False),
            TestCase(stream, small_content, big_usage, True, not stream),
            TestCase(stream, small_content, big_usage, False, False),
        ]
    ],
    ids=lambda x: x.get_name(),
)
async def test_auto_caching(test_app: httpx.AsyncClient, ts: TestCase):

    query_part = "api-version=2023-03-15-preview"
    adapter_url = f"/openai/deployments/gpt-4/chat/completions?{query_part}"
    upstream_endpoint = (
        "http://localhost:5001/openai/deployments/gpt-4o/chat/completions"
    )
    upstream_url = f"{upstream_endpoint}?{query_part}"

    mock_response(
        upstream_url=upstream_url,
        stream=ts.stream,
        chunks=[
            single_choice_chunk(
                delta={"role": "assistant", "content": "5"},
                finish_reason="stop",
                usage=ts.request_usage,
            )
        ],
    )

    response = await test_app.post(
        adapter_url,
        json={
            "messages": [
                {"role": "system", "content": "be a helpful assistant"},
                {"role": "user", "content": ts.request_content},
            ],
            "stream": ts.stream,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            **(
                {}
                if not ts.caching_enabled
                else {"X-DIAL-CACHE-BREAKPOINT-PATH": "whatever"}
            ),
        },
    )

    assert response.status_code == 200

    cache_path = response.headers.get("X-DIAL-CACHE-BREAKPOINT-PATH")
    expire_at = response.headers.get("X-DIAL-CACHE-EXPIRE-AT")

    if ts.expected_caching_response:
        assert cache_path == "prefix.body.messages[1]"
        assert expire_at is not None
    else:
        assert cache_path is None
        assert expire_at is None
