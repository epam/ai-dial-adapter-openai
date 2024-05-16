import json

import httpx
import pytest
import respx

from tests.utils.stream import OpenAIStream


def assert_equal(actual, expected):
    assert actual == expected


@respx.mock
@pytest.mark.asyncio
async def test_incorrect_top_level_field(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        {
            "error": {
                "message": "Unrecognized argument: extra_field.",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=400,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
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

    assert response.status_code == 400
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
            "code": None,
            "message": "Api version is a required query parameter",
            "param": None,
            "type": "invalid_request_error",
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_incorrect_nested_field(test_app: httpx.AsyncClient):

    mock_stream = OpenAIStream(
        {
            "error": {
                "message": "Unrecognized argument: extra_field.",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=400,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [
                {"role": "user", "content": "Test content", "extra_field": 1}
            ],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    mock_stream.assert_response_content(response, assert_equal)


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
                    "message": {"role": "assistant"},
                }
            ],
            "usage": None,
        },
        {
            "error": {
                "message": "Error test",
                "type": "runtime_error",
                "param": None,
                "code": None,
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
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(status_code=200)

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
            "param": None,
            "code": None,
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_incorrect_format(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(status_code=400, content=b"Incorrect format")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    assert response.content == b"Incorrect format"


@respx.mock
@pytest.mark.asyncio
async def test_incorrect_streaming_request(test_app: httpx.AsyncClient):
    expected_response = {
        "error": {
            "message": "0 is less than the minimum of 1 - 'n'",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=400,
        content=json.dumps(expected_response),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
            "n": 0,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    assert response.json() == expected_response
