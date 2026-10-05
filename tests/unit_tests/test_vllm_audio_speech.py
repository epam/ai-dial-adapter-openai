import base64
import json

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.conftest import create_test_client

_DEPLOYMENT = "vllm-tts"
_UPSTREAM_ENDPOINT = "http://localhost:5001/v1/audio/speech"
_AUDIO = b"audio-bytes"


@pytest.fixture
async def test_app():
    config = ApplicationConfig(VLLM_DEPLOYMENTS=[_DEPLOYMENT])
    async with create_test_client(
        config,
        base_url=f"http://test-app.com/openai/deployments/{_DEPLOYMENT}",
    ) as client:
        yield client


def _request(
    test_app: httpx.AsyncClient,
    configuration: dict,
    *,
    upstream_key: str | None = "key",
):
    headers = {
        "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        "Api-Key": "dial-key",
    }
    if upstream_key is not None:
        headers["X-UPSTREAM-KEY"] = upstream_key

    return test_app.post(
        "chat/completions",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "custom_fields": {"configuration": configuration},
        },
    )


@respx.mock
async def test_extra_configuration_fields_reach_the_upstream(
    test_app: httpx.AsyncClient,
):
    """
    vLLM-hosted TTS models take provider-specific parameters that the OpenAI
    client doesn't know about: they must be forwarded verbatim.
    """
    route = respx.post(_UPSTREAM_ENDPOINT).mock(
        return_value=httpx.Response(
            200, headers={"Content-Type": "audio/wav"}, content=_AUDIO
        )
    )

    response = await _request(
        test_app,
        {
            "voice": "nova",
            "speed": 1.5,
            "foobar1": "https://example.com/sample.wav",
            "foobar2": 24000,
        },
    )

    assert response.status_code == 200
    assert json.loads(route.calls.last.request.content) == {
        "input": "Hello",
        "model": _DEPLOYMENT,
        "voice": "nova",
        "speed": 1.5,
        "foobar1": "https://example.com/sample.wav",
        "foobar2": 24000,
    }

    attachment = response.json()["choices"][0]["message"]["custom_content"][
        "attachments"
    ][0]
    assert attachment["type"] == "audio/wav"
    assert base64.b64decode(attachment["data"]) == _AUDIO


@respx.mock
async def test_missing_upstream_key_is_not_an_azure_login(
    test_app: httpx.AsyncClient,
):
    """
    A vLLM deployment has no Azure identity behind it, so the request must
    fail right away instead of reaching out to Entra ID for a token.
    """
    response = await _request(test_app, {}, upstream_key=None)

    assert response.status_code == 401
    assert (
        response.json()["error"]["message"]
        == "X-UPSTREAM-KEY header is missing"
    )
