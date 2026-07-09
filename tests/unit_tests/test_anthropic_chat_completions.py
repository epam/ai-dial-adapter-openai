import json

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.conftest import create_test_client
from tests.utils.openai import user

_DEPLOYMENT_ID = "prefix.claude-opus-4-6"
_UPSTREAM_BASE = "https://test-foundry.services.ai.azure.com/anthropic"
_UPSTREAM_ENDPOINT = f"{_UPSTREAM_BASE}/v1/messages"

_MESSAGES_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "model": "claude-opus-4-6",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


@pytest.fixture
async def anthropic_client():
    async with create_test_client(
        app_config=ApplicationConfig(),
        base_url=f"http://test-app.com/openai/deployments/{_DEPLOYMENT_ID}",
    ) as client:
        yield client


def _headers(**extra: str) -> dict[str, str]:
    return {
        "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        "X-UPSTREAM-KEY": "dummy-upstream-api-key",
        "api-key": "dummy-api-key",
        **extra,
    }


@respx.mock
@pytest.mark.parametrize(
    ("override_name", "expected_model"),
    [
        ("claude-opus-4-6", "claude-opus-4-6"),
        (None, _DEPLOYMENT_ID),
    ],
)
async def test_upstream_model_name(
    anthropic_client: httpx.AsyncClient,
    override_name: str | None,
    expected_model: str,
):
    route = respx.post(url__regex=rf"{_UPSTREAM_BASE}/v1/messages.*").respond(
        status_code=200,
        content_type="application/json",
        content=json.dumps(_MESSAGES_RESPONSE),
    )

    extra = (
        {} if override_name is None else {"X-DIAL-OVERRIDE-NAME": override_name}
    )

    response = await anthropic_client.post(
        "chat/completions?api-version=2024-12-01-preview",
        json={"messages": [user("Say hello.")], "stream": False},
        headers=_headers(**extra),
    )

    assert response.status_code == 200
    assert route.called

    upstream_body = json.loads(route.calls.last.request.content)
    assert upstream_body["model"] == expected_model

    assert response.json()["model"] == expected_model
