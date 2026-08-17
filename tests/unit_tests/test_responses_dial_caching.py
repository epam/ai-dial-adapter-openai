import json
from typing import Any

import httpx
import pytest
import respx

from aidial_adapter_openai.utils.caching import (
    get_responses_breakpoint_path,
)
from tests.conftest import OpenAIClientFactory
from tests.utils.mock_server import MockServer

_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"

_UPSTREAM_ENDPOINT = "https://api.openai.com/v1/responses"

CHAT_COMPLETIONS_INPUT = RESPONSES_INPUT = [
    {"role": "system", "content": "be a helpful assistant"},
    {"role": "user", "content": "2+3=?"},
]


@pytest.fixture(params=[False, True], ids=["block", "stream"])
def stream(request) -> bool:
    return request.param


@pytest.fixture(params=[False, True], ids=["caching-off", "caching-on"])
def caching_enabled(request) -> bool:
    return request.param


def _mock_upstream_response():
    MockServer().post(_UPSTREAM_ENDPOINT)(
        MockServer.mock_responses_api_response("text.txt")
    )


def _caching_headers(caching_enabled: bool) -> dict[str, str]:
    # DIAL Core sends the header if the deployment
    # is marked in listing as supporting auto-caching
    return {_CACHE_BREAKPOINT_PATH: "whatever"} if caching_enabled else {}


async def _post_chat_completions(
    test_app: httpx.AsyncClient,
    *,
    messages: list[dict[str, Any]],
    stream: bool,
    caching_enabled: bool = True,
    **extra_body: Any,
) -> httpx.Response:
    return await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "stream": stream,
            "messages": messages,
            **extra_body,
        },
        headers={
            "X-UPSTREAM-KEY": "test-api-key",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
            **_caching_headers(caching_enabled),
        },
    )


def _assert_breakpoint_path(
    response: httpx.Response | Any, expected_path: str | None
):
    if expected_path is None:
        assert _CACHE_BREAKPOINT_PATH not in response.headers
        assert _CACHE_EXPIRE_AT not in response.headers
    else:
        assert response.headers[_CACHE_BREAKPOINT_PATH] == expected_path
        assert int(response.headers[_CACHE_EXPIRE_AT]) > 0


def _discarded_messages(response: httpx.Response) -> list[int] | None:
    if response.headers["content-type"].startswith("text/event-stream"):
        body = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ][-1]
    else:
        body = response.json()

    return (body.get("statistics") or {}).get("discarded_messages")


@pytest.mark.parametrize(
    "request_body,expected_path",
    [
        ({"input": RESPONSES_INPUT}, "prefix.body.input[1]"),
        ({"input": "2+3=?"}, "prefix.body.input[0]"),
        (
            {"input": [], "instructions": "be a helpful assistant"},
            "prefix.body.instructions[0]",
        ),
        (
            {"instructions": "be a helpful assistant"},
            "prefix.body.instructions[0]",
        ),
        ({"input": []}, None),
        ({}, None),
    ],
)
def test_responses_breakpoint_path(
    request_body: Any, expected_path: str | None
):
    assert get_responses_breakpoint_path(request_body) == expected_path


@respx.mock
async def test_passthrough_auto_caching(
    create_openai_client: OpenAIClientFactory,
    stream: bool,
    caching_enabled: bool,
):
    _mock_upstream_response()
    client = create_openai_client(upstream_endpoint=_UPSTREAM_ENDPOINT)

    response = await client.responses.with_raw_response.create(
        model="upstream-model-name",
        input=RESPONSES_INPUT,  # type: ignore[arg-type]
        stream=stream,
        extra_headers=_caching_headers(caching_enabled),
    )

    _assert_breakpoint_path(
        response, "prefix.body.input[1]" if caching_enabled else None
    )


@respx.mock
async def test_adapter_auto_caching(
    test_app: httpx.AsyncClient, stream: bool, caching_enabled: bool
):
    _mock_upstream_response()

    response = await _post_chat_completions(
        test_app,
        messages=CHAT_COMPLETIONS_INPUT,
        stream=stream,
        caching_enabled=caching_enabled,
    )

    assert response.status_code == 200
    # The adapter is called with a Chat Completions API request,
    # so the breakpoint addresses its messages
    _assert_breakpoint_path(
        response, "prefix.body.messages[1]" if caching_enabled else None
    )


@respx.mock
async def test_adapter_auto_caching_with_truncation(
    test_app: httpx.AsyncClient, stream: bool
):
    """The breakpoint addresses the request as DIAL Core has sent it,
    regardless of the messages discarded by the adapter."""

    _mock_upstream_response()
    token_counts = iter([100, 80, 40])

    @MockServer().post(f"{_UPSTREAM_ENDPOINT}/input_tokens")
    def _count_tokens(_request: httpx.Request):
        return {
            "object": "response.input_tokens",
            "input_tokens": next(token_counts),
        }

    response = await _post_chat_completions(
        test_app,
        messages=[
            {"role": "system", "content": "be a helpful assistant"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "2+3=?"},
        ],
        stream=stream,
        max_prompt_tokens=50,
    )

    assert response.status_code == 200
    assert _discarded_messages(response) == [1, 2]
    _assert_breakpoint_path(response, "prefix.body.messages[3]")
