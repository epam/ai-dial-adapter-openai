import json

import httpx
import pytest
import respx

from aidial_adapter_openai.app import app


@respx.mock
@pytest.mark.asyncio
async def test_streaming():
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15"
    ).respond(
        status_code=200,
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
                        "finish_reason": None,
                        "delta": {
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
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1695940483,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {
                            "content": "Test content",
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
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1696245654,
                "model": "gpt-4",
                "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
                "usage": None,
            }
        )
        + "\n\n"
        + "data: [DONE]\n\n",
        content_type="text/event-stream",
    )

    test_app = httpx.AsyncClient(app=app, base_url="http://test.com")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15",
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
                        "finish_reason": None,
                        "delta": {"role": "assistant"},
                    }
                ],
                "usage": None,
            }
        elif index == 2:
            assert json.loads(line.removeprefix("data: ")) == {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1695940483,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {"content": "Test content"},
                    }
                ],
                "usage": None,
            }

        elif index == 4:
            assert json.loads(line.removeprefix("data: ")) == {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1696245654,
                "model": "gpt-4",
                "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 9,
                    "total_tokens": 11,
                },
            }

        elif index == 6:
            assert line == "data: [DONE]"
        else:
            assert False
