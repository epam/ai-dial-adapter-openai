import json

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.conftest import create_test_client

_UPSTREAM_ENDPOINT = "http://localhost:5001/v1/chat/completions"
_TOKENIZE_URL = "http://localhost:5001/tokenize"


@pytest.fixture
async def vllm_client():
    config = ApplicationConfig().add_deployment(
        "vllm-test", ChatCompletionDeploymentType.VLLM_CHAT_COMPLETIONS_API
    )
    async with create_test_client(
        app_config=config,
        base_url="http://test-app.com/openai/deployments/vllm-test",
    ) as client:
        yield client


@pytest.fixture
async def gpt_client():
    config = ApplicationConfig(
        TIKTOKEN_MODEL_MAPPING={"gpt-test": "gpt-4o"}
    ).add_deployment("gpt-test", ChatCompletionDeploymentType.GPT_GENERIC)
    async with create_test_client(
        app_config=config,
        base_url="http://test-app.com/openai/deployments/gpt-test",
    ) as client:
        yield client


@respx.mock
@pytest.mark.asyncio
async def test_tokenize_vllm_request_input(vllm_client: httpx.AsyncClient):
    captured: dict = {}

    def tokenize_handler(request: httpx.Request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            status_code=200,
            json={"count": 42, "tokens": list(range(42))},
        )

    respx.post(_TOKENIZE_URL).mock(side_effect=tokenize_handler)

    response = await vllm_client.post(
        "tokenize",
        json={
            "inputs": [
                {
                    "type": "request",
                    "value": {
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                }
            ]
        },
        headers={"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT},
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": 42}],
    }
    assert captured["body"]["model"] == "vllm-test"
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "hello"},
    ]


@respx.mock
@pytest.mark.asyncio
async def test_tokenize_vllm_uses_override_name_header(
    vllm_client: httpx.AsyncClient,
):
    captured: dict = {}

    def tokenize_handler(request: httpx.Request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            status_code=200, json={"count": 2, "tokens": [1, 2]}
        )

    respx.post(_TOKENIZE_URL).mock(side_effect=tokenize_handler)

    response = await vllm_client.post(
        "tokenize",
        json={
            "inputs": [
                {"type": "string", "value": "abc"},
                {
                    "type": "request",
                    "value": {
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                },
            ]
        },
        headers={
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
            "X-DIAL-OVERRIDE-NAME": "upstream-model-name",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [
            {"status": "success", "token_count": 2},
            {"status": "success", "token_count": 2},
        ],
    }
    assert captured["body"]["model"] == "upstream-model-name"


@respx.mock
@pytest.mark.asyncio
async def test_tokenize_vllm_string_input(vllm_client: httpx.AsyncClient):
    captured: dict = {}

    def tokenize_handler(request: httpx.Request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            status_code=200, json={"count": 3, "tokens": [1, 2, 3]}
        )

    respx.post(_TOKENIZE_URL).mock(side_effect=tokenize_handler)

    response = await vllm_client.post(
        "tokenize",
        json={"inputs": [{"type": "string", "value": "abc"}]},
        headers={"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT},
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": 3}],
    }
    assert captured["body"] == {
        "model": "vllm-test",
        "prompt": "abc",
        "add_special_tokens": False,
    }


@pytest.mark.asyncio
async def test_tokenize_tiktoken_request_input(gpt_client: httpx.AsyncClient):
    response = await gpt_client.post(
        "tokenize",
        json={
            "inputs": [
                {
                    "type": "request",
                    "value": {
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                }
            ]
        },
        headers={
            "X-UPSTREAM-ENDPOINT": "https://example.com/openai/deployments/gpt-test/chat/completions"
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["outputs"][0]["status"] == "success"
    assert body["outputs"][0]["token_count"] > 0


@pytest.mark.asyncio
async def test_tokenize_tiktoken_string_input(gpt_client: httpx.AsyncClient):
    response = await gpt_client.post(
        "tokenize",
        json={"inputs": [{"type": "string", "value": "hello"}]},
        headers={
            "X-UPSTREAM-ENDPOINT": "https://example.com/openai/deployments/gpt-test/chat/completions"
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": 1}],
    }


@respx.mock
@pytest.mark.asyncio
async def test_tokenize_per_input_error_isolation(
    vllm_client: httpx.AsyncClient,
):
    respx.post(_TOKENIZE_URL).mock(
        return_value=httpx.Response(
            status_code=200, json={"count": 5, "tokens": [1] * 5}
        )
    )

    response = await vllm_client.post(
        "tokenize",
        json={
            "inputs": [
                {"type": "string", "value": "ok"},
                {"type": "request", "value": {}},
                {"type": "unknown", "value": "bad"},
            ]
        },
        headers={"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT},
    )

    assert response.status_code == 200
    outputs = response.json()["outputs"]
    assert outputs[0] == {"status": "success", "token_count": 5}
    assert outputs[1]["status"] == "error"
    assert outputs[2]["status"] == "error"


@respx.mock
@pytest.mark.asyncio
async def test_tokenize_forwards_proxied_headers(
    vllm_client: httpx.AsyncClient,
):
    captured: dict = {}

    def tokenize_handler(request: httpx.Request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(status_code=200, json={"count": 1, "tokens": [1]})

    respx.post(_TOKENIZE_URL).mock(side_effect=tokenize_handler)

    response = await vllm_client.post(
        "tokenize",
        json={"inputs": [{"type": "string", "value": "x"}]},
        headers={
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
            "X-UPSTREAM-EXTRA-DATA": '{"headers_to_proxy": ["x-user-id"]}',
            "x-user-id": "user-42",
        },
    )

    assert response.status_code == 200
    assert captured["headers"]["x-user-id"] == "user-42"
    assert "authorization" not in {k.lower() for k in captured["headers"]}


@pytest.mark.asyncio
async def test_tokenize_invalid_inputs_raises(vllm_client: httpx.AsyncClient):
    response = await vllm_client.post(
        "tokenize",
        json={"inputs": "not-a-list"},
        headers={"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT},
    )

    assert response.status_code == 400
