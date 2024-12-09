import json
from typing import Any, Callable

import httpx
import pytest
import respx
from respx.types import SideEffectTypes

from tests.utils.dictionary import exclude_keys
from tests.utils.stream import OpenAIStream, single_choice_chunk


def assert_equal(actual: Any, expected: Any):
    assert actual == expected


def assert_equal_no_dynamic_fields(actual: Any, expected: Any):
    if isinstance(actual, dict) and isinstance(expected, dict):
        keys = {"id", "created"}
        assert exclude_keys(actual, keys) == exclude_keys(expected, keys)
    else:
        assert actual == expected


def mock_response(
    status_code: int,
    content_type: str,
    content: str,
    check_request: Callable[[httpx.Request], None] = lambda _: None,
) -> SideEffectTypes:
    def side_effect(request: httpx.Request):
        check_request(request)
        return httpx.Response(
            status_code=status_code,
            headers={"content-type": content_type},
            content=content,
        )

    return side_effect


@respx.mock
@pytest.mark.asyncio
async def test_single_chunk_token_counting(test_app: httpx.AsyncClient):
    # The adapter tolerates top-level extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        single_choice_chunk(
            delta={"role": "assistant", "content": "5"}, finish_reason="stop"
        ),
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
                "completion_tokens": 1,
                "total_tokens": 10,
            }
        },
    )


@respx.mock
@pytest.mark.asyncio
async def test_top_level_extra_field(test_app: httpx.AsyncClient):
    # The adapter tolerates top-level extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        {"error": {"message": "whatever", "code": "500"}}
    )

    def check_request(request: httpx.Request):
        assert json.loads(request.content)["extra_field"] == 1

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(
        side_effect=mock_response(
            status_code=200,
            content_type="text/event-stream",
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
    mock_stream.assert_response_content(response, assert_equal)


@respx.mock
@pytest.mark.asyncio
async def test_nested_extra_field(test_app: httpx.AsyncClient):
    # The adapter tolerates nested extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        {"error": {"message": "whatever", "code": "500"}}
    )

    def check_request(request: httpx.Request):
        assert json.loads(request.content)["messages"][0]["extra_field"] == 1

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(
        side_effect=mock_response(
            status_code=200,
            content_type="text/event-stream",
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
    mock_stream.assert_response_content(response, assert_equal)


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
            "code": "400",
            "message": "api-version is a required query parameter",
            "type": "invalid_request_error",
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_error_during_streaming_stopped(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(finish_reason="stop", delta={"role": "assistant"}),
        {
            "error": {
                "message": "Error test",
                "type": "runtime_error",
                "code": "500",
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
async def test_error_during_streaming_unfinished(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant", "content": "hello "}),
        {
            "error": {
                "message": "Error test",
                "type": "runtime_error",
                "code": "500",
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
                "completion_tokens": 2,
                "prompt_tokens": 9,
                "total_tokens": 11,
            }
        },
    )


@respx.mock
@pytest.mark.asyncio
async def test_interrupted_stream(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant", "content": "hello"}),
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

    expected_stream = OpenAIStream(
        single_choice_chunk(
            delta={"role": "assistant", "content": "hello"},
            finish_reason="length",
            usage={
                "completion_tokens": 1,
                "prompt_tokens": 9,
                "total_tokens": 10,
            },
        )
    )
    expected_stream.assert_response_content(response, assert_equal)


@respx.mock
@pytest.mark.asyncio
async def test_zero_chunk_stream(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream()

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

    expected_final_chunk = single_choice_chunk(
        delta={},
        finish_reason="length",
        usage={"prompt_tokens": 9, "completion_tokens": 0, "total_tokens": 9},
    )

    expected_stream = OpenAIStream(expected_final_chunk)
    expected_stream.assert_response_content(
        response, assert_equal_no_dynamic_fields
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
            "code": "400",
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_no_request_response_validation(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200, json={"messages": "string", "extra_response": "string"}
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Test content",
                    "extra_mesage": "string",
                }
            ],
            "extra_request": "string",
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
            "Content-Type": "application/pdf",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "messages": "string",
        "extra_response": "string",
    }


@respx.mock
@pytest.mark.asyncio
async def test_status_error_from_upstream(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(status_code=400, content="Bad request")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    assert response.content == b"Bad request"


@respx.mock
@pytest.mark.asyncio
async def test_timeout_error_from_upstream(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(side_effect=httpx.ReadTimeout("Timeout error"))

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 504
    assert response.json() == {
        "error": {
            "message": "Request timed out",
            "type": "timeout",
            "code": "504",
            "display_message": "Request timed out. Please try again later.",
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_connection_error_from_upstream(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(side_effect=httpx.ConnectError("Connection error"))

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "Error communicating with OpenAI",
            "type": "connection",
            "code": "502",
            "display_message": "OpenAI server is not responsive. Please try again later.",
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_unexpected_multi_modal_input_streaming(
    test_app: httpx.AsyncClient,
):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test response"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content=mock_stream.to_content(),
        content_type="text/event-stream",
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "http://example.com/image.png"
                            },
                        }
                    ],
                }
            ],
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(response, assert_equal)


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
            "code": "400",
            "message": "'0' is less than the minimum of 1 - 'max_prompt_tokens'",
            "type": "invalid_request_error",
        }
    }

    assert response.status_code == 400
    assert response.json() == expected_response
