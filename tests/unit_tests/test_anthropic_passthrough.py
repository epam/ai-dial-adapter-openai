import json

import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.conftest import create_test_client

_UPSTREAM_BASE = "https://test-foundry.services.ai.azure.com/anthropic"
_UPSTREAM_ENDPOINT = f"{_UPSTREAM_BASE}/v1/messages"
_UPSTREAM_KEY = "test-upstream-key"

_MESSAGES_REQUEST = {
    "model": "claude-opus-4-5",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 1024,
}

_MESSAGES_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello! How can I assist you today?"}],
    "model": "claude-opus-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 9},
}


def _headers(**extra: str) -> dict[str, str]:
    return {
        "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        "X-UPSTREAM-KEY": _UPSTREAM_KEY,
        **extra,
    }


@respx.mock
async def test_messages_passthrough_relays_upstream_response():
    route = respx.post(url__regex=rf"{_UPSTREAM_BASE}/v1/messages.*").respond(
        status_code=200,
        content_type="application/json",
        content=json.dumps(_MESSAGES_RESPONSE),
    )

    async with create_test_client(ApplicationConfig()) as client:
        response = await client.post(
            "/anthropic/v1/messages",
            json=_MESSAGES_REQUEST,
            headers=_headers(),
        )

    assert response.status_code == 200
    assert response.json() == _MESSAGES_RESPONSE
    assert route.called

    # The upstream must receive the request body unchanged.
    upstream_request = route.calls.last.request
    assert json.loads(upstream_request.content) == _MESSAGES_REQUEST
