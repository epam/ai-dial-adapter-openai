import httpx
import pytest
import respx

from aidial_adapter_openai.app import app
from tests.utils.stream import OpenAIStream


@respx.mock
@pytest.mark.asyncio
async def test_streaming():
    mock_stream = OpenAIStream(
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
        },
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
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1696245654,
            "model": "gpt-4",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": None,
        },
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15"
    ).respond(
        status_code=200,
        content=mock_stream.to_content(),
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
    mock_stream.assert_response_content(
        response,
        usages={
            2: {
                "completion_tokens": 2,
                "prompt_tokens": 9,
                "total_tokens": 11,
            }
        },
    )
