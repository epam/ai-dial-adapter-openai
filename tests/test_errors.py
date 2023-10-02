import json

import pytest
from aioresponses import aioresponses
from httpx import AsyncClient

from aidial_adapter_openai.app import app


@pytest.mark.asyncio
async def test_error_during_streaming(aioresponses: aioresponses):
    aioresponses.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        status=200,
        body="data: "
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
        content_type="text/event-stream",
    )
    test_app = AsyncClient(app=app, base_url="http://test.com")

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

    assert response.status_code == 200

    for index, line in enumerate(response.iter_lines()):
        if index % 2 == 1:
            assert line == ""
            continue

        if index == 0:
            assert (
                line
                == 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1695940483,"model":"gpt-4","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant"}}],"usage":{"completion_tokens":0,"prompt_tokens":9,"total_tokens":9}}'
            )
        elif index == 2:
            assert (
                line
                == 'data: {"error": {"message": "Error test", "type": "runtime_error", "param": null, "code": null}}'
            )
        elif index == 4:
            assert line == "data: [DONE]"
        else:
            assert False


@pytest.mark.asyncio
async def test_incorrect_upsteram_url(aioresponses: aioresponses):
    aioresponses.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        status=200,
        body={},
    )
    test_app = AsyncClient(app=app, base_url="http://test.com")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions",
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


@pytest.mark.asyncio
async def test_asdasd(aioresponses: aioresponses):
    aioresponses.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        status=400,
        body="Incorrect format",
    )
    test_app = AsyncClient(app=app, base_url="http://test.com")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400

    assert response.content == b"Incorrect format"
