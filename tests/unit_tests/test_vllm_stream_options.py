import json

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.conftest import create_test_client
from tests.utils.stream import OpenAIStream, single_choice_chunk

_UPSTREAM_ENDPOINT = "http://localhost:5001/v1/chat/completions"
_API_VERSION = "api-version=2024-12-01-preview"


@pytest.fixture
async def test_app():
    """Configure vllm-test as a vLLM deployment."""
    config = ApplicationConfig().add_deployment(
        "vllm-test", ChatCompletionDeploymentType.VLLM_CHAT_COMPLETIONS_API
    )
    async with create_test_client(config) as client:
        yield client


@respx.mock
@pytest.mark.asyncio
async def test_vllm_stream_options_include_usage_injected(
    test_app: httpx.AsyncClient,
):
    """For vLLM streaming calls, the adapter must force stream_options.include_usage=True."""

    # Mock the upstream chat completion response
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant", "content": "hi"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
    )

    def chat_completion_handler(request: httpx.Request):
        body = json.loads(request.content)
        # Verify that stream_options.include_usage was injected
        assert body.get("stream") is True
        assert body.get("stream_options", {}).get("include_usage") is True

        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=mock_stream.to_content(),
        )

    respx.post(_UPSTREAM_ENDPOINT).mock(side_effect=chat_completion_handler)

    # Mock the tokenize endpoint (vLLM tokenizer will call it for truncation check)
    respx.post("http://localhost:5001/tokenize").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "count": 10,
                "tokens": list(range(10)),
            },
        )
    )

    response = await test_app.post(
        f"/openai/deployments/vllm-test/chat/completions?{_API_VERSION}",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        },
    )

    assert response.status_code == 200


@respx.mock
@pytest.mark.asyncio
async def test_vllm_stream_options_include_usage_merged(
    test_app: httpx.AsyncClient,
):
    """If stream_options already exists, include_usage must be set/overridden but other fields kept."""

    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant", "content": "hi"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
    )

    def chat_completion_handler(request: httpx.Request):
        body = json.loads(request.content)
        stream_options = body.get("stream_options", {})

        # Verify that include_usage was set to True
        assert stream_options.get("include_usage") is True
        # Verify that other fields in stream_options are preserved
        assert stream_options.get("foo") == "bar"

        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=mock_stream.to_content(),
        )

    respx.post(_UPSTREAM_ENDPOINT).mock(side_effect=chat_completion_handler)

    # Mock the tokenize endpoint
    respx.post("http://localhost:5001/tokenize").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "count": 10,
                "tokens": list(range(10)),
            },
        )
    )

    response = await test_app.post(
        f"/openai/deployments/vllm-test/chat/completions?{_API_VERSION}",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"foo": "bar", "include_usage": False},
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        },
    )

    assert response.status_code == 200


@respx.mock
@pytest.mark.asyncio
async def test_vllm_non_stream_does_not_inject_stream_options(
    test_app: httpx.AsyncClient,
):
    """For non-stream calls, adapter shouldn't force stream_options."""

    def chat_completion_handler(request: httpx.Request):
        body = json.loads(request.content)

        # Verify that stream_options is not injected for non-streaming requests
        assert "stream_options" not in body

        return httpx.Response(
            status_code=200,
            json={
                "id": "chat-123",
                "object": "chat.completion",
                "model": "vllm-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    respx.post(_UPSTREAM_ENDPOINT).mock(side_effect=chat_completion_handler)

    # Mock the tokenize endpoint
    respx.post("http://localhost:5001/tokenize").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "count": 10,
                "tokens": list(range(10)),
            },
        )
    )

    response = await test_app.post(
        f"/openai/deployments/vllm-test/chat/completions?{_API_VERSION}",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "hi"
