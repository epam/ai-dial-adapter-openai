import json
from typing import Callable

import httpx
import pytest
import respx
from respx.types import SideEffectTypes

from tests.utils.stream import OpenAIStream


def assert_equal(actual, expected):
    assert actual == expected


def mock_response(
    status_code: int,
    headers: dict,
    content: str,
    check_request: Callable[[httpx.Request], None] = lambda _: None,
) -> SideEffectTypes:
    def side_effect(request: httpx.Request):
        check_request(request)
        return httpx.Response(
            status_code=status_code,
            headers=headers,
            content=content,
        )

    return side_effect


@respx.mock
@pytest.mark.asyncio
async def test_top_level_extra_field(test_app: httpx.AsyncClient):
    # The adapter tolerates top-level extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1695940483,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta": {"role": "assistant", "content": "5"},
                }
            ],
        },
    )

    def check_request(request: httpx.Request):
        assert json.loads(request.content)["extra_field"] == 1

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(
        side_effect=mock_response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=mock_stream.to_content(),
            check_request=check_request,
        ),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
            "extra_field": 1,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(
        response,
        assert_equal,
        usages={
            0: {
                "prompt_tokens": 9,
                "completion_tokens": 1,
                "total_tokens": 10,
            }
        },
    )


@respx.mock
@pytest.mark.asyncio
async def test_nested_extra_field(test_app: httpx.AsyncClient):
    # The adapter tolerates nested extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1695940483,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta": {"role": "assistant", "content": "5"},
                }
            ],
        },
    )

    def check_request(request: httpx.Request):
        assert json.loads(request.content)["messages"][0]["extra_field"] == 1

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(
        side_effect=mock_response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=mock_stream.to_content(),
            check_request=check_request,
        ),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [
                {"role": "user", "content": "2+3=?", "extra_field": 1}
            ],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(
        response,
        assert_equal,
        usages={
            0: {
                "prompt_tokens": 11,
                "completion_tokens": 1,
                "total_tokens": 12,
            }
        },
    )


@respx.mock
@pytest.mark.asyncio
async def test_missing_api_version(test_app: httpx.AsyncClient):

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "api-version is a required query parameter",
            "type": "invalid_request_error",
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_error_during_streaming(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1695940483,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta": {"role": "assistant"},
                }
            ],
            "usage": None,
        },
        {
            "error": {
                "message": "Error test",
                "type": "runtime_error",
            }
        },
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(
        response,
        assert_equal,
        usages={
            0: {
                "prompt_tokens": 9,
                "completion_tokens": 0,
                "total_tokens": 9,
            }
        },
    )


@respx.mock
@pytest.mark.asyncio
async def test_incorrect_upstream_url(test_app: httpx.AsyncClient):
    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            # upstream endpoint should contain the full path
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Invalid upstream endpoint format",
            "type": "invalid_request_error",
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_correct_upstream_url(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(status_code=400, content=b"Any response")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    assert response.content == b"Any response"


@respx.mock
@pytest.mark.asyncio
async def test_incorrect_streaming_request(test_app: httpx.AsyncClient):
    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
            "max_prompt_tokens": 0,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    expected_response = {
        "error": {
            "message": "'0' is less than the minimum of 1 - 'max_prompt_tokens'",
            "type": "invalid_request_error",
        }
    }

    assert response.status_code == 400
    assert response.json() == expected_response
