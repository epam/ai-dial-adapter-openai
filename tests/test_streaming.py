import json

import pytest
from aioresponses import aioresponses
from httpx import AsyncClient

from aidial_adapter_openai.app import app


@pytest.mark.asyncio
async def test_streaming(aioresponses: aioresponses):
    aioresponses.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15",
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
    test_app = AsyncClient(app=app, base_url="http://test.com")

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

        print("!", index, line, "!")  # TODO: change to asserts
