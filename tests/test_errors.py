import json

import httpx
import pytest
import respx

from aidial_adapter_openai.app import app


@respx.mock
@pytest.mark.asyncio
async def test_error_during_streaming():
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content="data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1695940483,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                        },
                    }
                ],
                "usage": None,
            }
        )
        + "\n\n"
        + "data: "
        + json.dumps(
            {
                "error": {
                    "message": "Error test",
                    "type": "runtime_error",
                    "param": None,
                    "code": None,
                }
            }
        )
        + "\n\n"
        + "data: [DONE]\n\n",
    )

    test_app = httpx.AsyncClient(app=app, base_url="http://test.com")

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

    for index, line in enumerate(response.iter_lines()):
        if index % 2 == 1:
            assert line == ""
            continue

        if index == 0:
            assert json.loads(line.removeprefix("data: ")) == {
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
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 9,
                    "total_tokens": 9,
                },
            }

        elif index == 2:
            assert json.loads(line.removeprefix("data: ")) == {
                "error": {
                    "message": "Error test",
                    "type": "runtime_error",
                    "param": None,
                    "code": None,
                }
            }
        elif index == 4:
            assert line == "data: [DONE]"
        else:
            assert False


@respx.mock
@pytest.mark.asyncio
async def test_incorrect_upstream_url():
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(status_code=200)

    test_app = httpx.AsyncClient(app=app, base_url="http://test.com")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001",  # upstream endpoint should contain the full path
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
async def test_incorrect_format():
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(status_code=400, content=b"Incorrect format")

    test_app = httpx.AsyncClient(app=app, base_url="http://test.com")

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
async def test_incorrect_streaming_request():
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

    test_app = httpx.AsyncClient(app=app, base_url="http://test.com")

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
