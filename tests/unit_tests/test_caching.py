import dataclasses
import json
from typing import List

import httpx
import pytest
from aioresponses import aioresponses

from aidial_adapter_openai.app_config import ApplicationConfig
from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from tests.conftest import create_test_client
from tests.utils.stream import OpenAIStream, single_choice_chunk


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


def mock_response(
    mock: aioresponses, upstream_url: str, stream: bool, chunks: List[dict]
):
    mock_stream = OpenAIStream(*chunks)
    if stream:
        mock.add(
            upstream_url,
            method="POST",
            status=200,
            content_type="text/event-stream",
            body=mock_stream.to_content(),
        )
    else:
        mock.add(
            upstream_url,
            method="POST",
            status=200,
            content_type="application/json",
            body=json.dumps(mock_stream.to_block_response()),
        )


token_threshold = 1024
big_content = "cat " * 1512  # #tokens >= token_threshold
small_content = "cat"  # #tokens < token_threshold

big_usage = {
    "prompt_tokens": token_threshold,
    "completion_tokens": 1,
    "total_tokens": token_threshold + 1,
}
small_usage = {
    "prompt_tokens": big_usage["prompt_tokens"] - 1,
    "completion_tokens": 1,
    "total_tokens": big_usage["total_tokens"] - 1,
}


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

        if len(self.request_content) >= token_threshold:
            xs.append("big-content")
        else:
            xs.append("small-content")

        if self.caching_enabled:
            xs.append("caching+")
        else:
            xs.append("caching-")

        if self.request_usage:
            if self.request_usage["prompt_tokens"] >= token_threshold:
                xs.append("big usage")
            else:
                xs.append("small usage")
        else:
            xs.append("no-usage")

        return "/".join(xs)


@pytest.fixture
async def gpt4o_client():
    app_config = (
        ApplicationConfig()
        .add_deployment("app", ChatCompletionDeploymentType.GPT4O)
        .map_to_tiktoken_model("app", "gpt-4o")
    )

    async with create_test_client(
        app_config=app_config,
        base_url="http://test-app.com/openai/deployments/app",
    ) as client:
        yield client


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
async def test_auto_caching(
    mock_aioresponse: aioresponses,
    gpt4o_client: httpx.AsyncClient,
    ts: TestCase,
):

    query_part = "api-version=2023-03-15-preview"
    adapter_url = f"chat/completions?{query_part}"
    upstream_endpoint = "http://test-upstream/openai/deployments/upstream-deployment/chat/completions"
    upstream_url = f"{upstream_endpoint}?{query_part}"

    mock_response(
        mock=mock_aioresponse,
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

    response = await gpt4o_client.post(
        adapter_url,
        json={
            "messages": [
                {"role": "system", "content": "be a helpful assistant"},
                {"role": "user", "content": ts.request_content},
            ],
            "stream": ts.stream,
        },
        headers={
            "X-UPSTREAM-KEY": "dummy-upstream-api-key",
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
