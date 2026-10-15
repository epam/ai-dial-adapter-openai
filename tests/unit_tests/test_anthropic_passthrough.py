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


def _headers(upstream_base: str, **extra: str) -> dict[str, str]:
    return {
        "X-UPSTREAM-ENDPOINT": f"{upstream_base}/v1/messages",
        "X-UPSTREAM-KEY": _UPSTREAM_KEY,
        **extra,
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
async def test_messages_passthrough_relays_upstream_response(
    upstream_base: str, auth_header: str
):
    route = respx.post(url__regex=rf"{upstream_base}/v1/messages.*").respond(
        status_code=200,
        content_type="application/json",
        content=json.dumps(_MESSAGES_RESPONSE),
    )

    async with create_test_client(ApplicationConfig()) as client:
        response = await client.post(
            "/anthropic/v1/messages",
            json=_MESSAGES_REQUEST,
            headers=_headers(upstream_base),
        )

    assert response.status_code == 200
    assert response.json() == _MESSAGES_RESPONSE
    assert route.called

    # The upstream must receive the request body unchanged.
    upstream_request = route.calls.last.request
    assert json.loads(upstream_request.content) == _MESSAGES_REQUEST
    assert upstream_request.headers[auth_header] == _UPSTREAM_KEY


@respx.mock
async def test_messages_passthrough_forwards_prompt_fields_as_received():
    # The prompt fields ("messages", "system", "tools") are large, so the
    # passthrough sends them in their source text instead of re-encoding them.
    # Whitespace, key order and \u escapes would all change in a re-encode.
    messages = (
        '[ {"text": "caf\u00e9", "role" : "user",\n   "content":"Say hello."} ]'
    )
    system = '[{"type":"text",  "text":"Be brief."}]'
    body = (
        '{"max_tokens": 1024, "messages": '
        + messages
        + ', "model" : "claude-opus-4-5", "system":'
        + system
        + "}"
    ).encode()

    route = respx.post(url__regex=rf"{_ANTHROPIC_BASE}/v1/messages.*").respond(
        status_code=200,
        content_type="application/json",
        content=json.dumps(_MESSAGES_RESPONSE),
    )

    async with create_test_client(ApplicationConfig()) as client:
        response = await client.post(
            "/anthropic/v1/messages",
            content=body,
            headers={
                **_headers(_ANTHROPIC_BASE),
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 200

    upstream_content = route.calls.last.request.content
    assert messages.encode() in upstream_content
    assert system.encode() in upstream_content
    assert json.loads(upstream_content) == json.loads(body)
