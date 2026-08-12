import json

import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.conftest import create_test_client

_FOUNDRY_BASE = "https://test-foundry.services.ai.azure.com/anthropic"
_ANTHROPIC_BASE = "https://api.anthropic.com"
_FIREWORKS_BASE = "https://api.fireworks.ai/inference"
_OPENROUTER_BASE = "https://openrouter.ai/api"
_UPSTREAM_KEY = "test-upstream-key"

_MESSAGES_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "model": "claude-opus-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3},
}


@pytest.mark.parametrize(
    ("upstream_base", "auth_header"),
    [
        # Azure AI Foundry authenticates with the "api-key" header,
        # while the native Anthropic API and the third-party providers
        # of the Messages API follow the "x-api-key" convention.
        (_FOUNDRY_BASE, "api-key"),
        (_ANTHROPIC_BASE, "x-api-key"),
        (_FIREWORKS_BASE, "x-api-key"),
        (_OPENROUTER_BASE, "x-api-key"),
    ],
)
@respx.mock
async def test_dial_chat_completion_via_messages_api(
    upstream_base: str, auth_header: str
):
    route = respx.post(f"{upstream_base}/v1/messages").respond(
        status_code=200,
        content_type="application/json",
        content=json.dumps(_MESSAGES_RESPONSE),
    )

    async with create_test_client(ApplicationConfig()) as client:
        response = await client.post(
            "/openai/deployments/claude-opus-4-5/chat/completions"
            "?api-version=2024-12-01-preview",
            json={"messages": [{"role": "user", "content": "Say hello."}]},
            headers={
                "X-UPSTREAM-ENDPOINT": f"{upstream_base}/v1/messages",
                "X-UPSTREAM-KEY": _UPSTREAM_KEY,
                "Api-Key": "test-adapter-key",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["usage"]["prompt_tokens"] == 10
    assert body["usage"]["completion_tokens"] == 3

    assert route.called
    upstream_request = route.calls.last.request
    assert upstream_request.headers[auth_header] == _UPSTREAM_KEY
