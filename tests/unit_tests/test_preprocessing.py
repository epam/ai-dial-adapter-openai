import json

import httpx
import pytest
import respx

from tests.utils.stream import OpenAIStream, single_choice_chunk

_API_VERSION = "api-version=2023-03-15-preview"
_UPSTREAM_ENDPOINT = (
    "http://localhost:5001/openai/deployments/gpt-4/chat/completions"
)


@respx.mock
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "block"])
async def test_stream_options(test_app: httpx.AsyncClient, stream: bool):
    chat_completion_response = OpenAIStream(
        single_choice_chunk(
            finish_reason="stop", delta={"role": "assistant", "content": "test"}
        ),
    )

    def chat_completion_handler(request: httpx.Request):
        body = json.loads(request.content)
        stream = body.get("stream", False)
        stream_options = "stream_options" in body

        if stream:
            assert stream_options, (
                "stream_options should be preserved for streaming requests"
            )
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "text/event-stream"},
                content=chat_completion_response.to_content(),
            )
        else:
            assert not stream_options, (
                "stream_options should be removed for non-streaming requests"
            )
            return httpx.Response(
                status_code=200,
                json=chat_completion_response.to_block_response(),
            )

    respx.post(f"{_UPSTREAM_ENDPOINT}?{_API_VERSION}").mock(
        side_effect=chat_completion_handler
    )

    response = await test_app.post(
        f"/openai/deployments/gpt-4/chat/completions?{_API_VERSION}",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": stream,
            "stream_options": {"include_usage": True},
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        },
    )

    assert response.status_code == 200


@respx.mock
async def test_chat_completion_proxies_extra_headers(
    test_app: httpx.AsyncClient,
):
    chat_completion_response = OpenAIStream(
        single_choice_chunk(
            finish_reason="stop", delta={"role": "assistant", "content": "ok"}
        )
    )

    def chat_completion_handler(request: httpx.Request):
        assert request.headers.get("x-user-id") == "user-1"
        return httpx.Response(
            status_code=200,
            json=chat_completion_response.to_block_response(),
        )

    respx.post(f"{_UPSTREAM_ENDPOINT}?{_API_VERSION}").mock(
        side_effect=chat_completion_handler
    )

    response = await test_app.post(
        f"/openai/deployments/gpt-4/chat/completions?{_API_VERSION}",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": False,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
            "X-UPSTREAM-EXTRA-DATA": '{"headers_to_proxy": ["x-user-id"]}',
            "x-user-id": "user-1",
        },
    )

    assert response.status_code == 200
