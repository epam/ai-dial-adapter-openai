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
_RESPONSES_UPSTREAM_ENDPOINT = "https://api.openai.com/v1/responses"
_RESPONSES_INPUT_TOKENS_URL = f"{_RESPONSES_UPSTREAM_ENDPOINT}/input_tokens"
_UNSUPPORTED_RESPONSES_UPSTREAM_ENDPOINTS = [
    "https://test.openai.azure.com/openai/v1/responses",
    "https://test.services.ai.azure.com/openai/v1/responses",
    "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
]
_UNSUPPORTED_RESPONSES_VENDOR_ERROR_MESSAGE = (
    "The tokenize and truncate_prompt endpoints are not implemented for "
    "Responses API deployments backed by Azure OpenAI or Amazon Bedrock."
)
_API_KEY = "test-adapter-api-key"
_RESPONSES_TOKENIZE_HEADERS = {
    "X-UPSTREAM-ENDPOINT": _RESPONSES_UPSTREAM_ENDPOINT,
    "X-UPSTREAM-KEY": "dummy",
}
_RESPONSES_TOKENIZE_PARAMS = {"api-version": "2025-01-01"}

_CHAT_COMPLETIONS_REQUEST = {
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "How is weather in LA?"}],
        }
    ]
}


def _tokenize_headers(**extra: str) -> dict[str, str]:
    return {"Api-Key": _API_KEY, **extra}


async def _post_tokenize_to_responses(
    client: httpx.AsyncClient, tokenize_input: dict
) -> httpx.Response:
    return await client.post(
        "tokenize",
        json={"inputs": [tokenize_input]},
        headers=_tokenize_headers(**_RESPONSES_TOKENIZE_HEADERS),
        params=_RESPONSES_TOKENIZE_PARAMS,
    )


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


@pytest.fixture
async def responses_client():
    async with create_test_client(
        app_config=ApplicationConfig(),
        base_url="http://test-app.com/openai/deployments/responses-test",
    ) as client:
        yield client


@pytest.fixture
async def anthropic_client():
    async with create_test_client(
        app_config=ApplicationConfig(),
        base_url="http://test-app.com/openai/deployments/claude-test",
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
        headers=_tokenize_headers(
            **{"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT}
        ),
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
                        "model": "user-model-in-body",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                },
            ]
        },
        headers=_tokenize_headers(
            **{
                "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
                "X-DIAL-OVERRIDE-NAME": "upstream-model-name",
            }
        ),
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
        headers=_tokenize_headers(
            **{"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT}
        ),
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
        headers=_tokenize_headers(
            **{
                "X-UPSTREAM-ENDPOINT": "https://example.com/openai/deployments/gpt-test/chat/completions"
            }
        ),
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
        headers=_tokenize_headers(
            **{
                "X-UPSTREAM-ENDPOINT": "https://example.com/openai/deployments/gpt-test/chat/completions"
            }
        ),
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": 1}],
    }


@pytest.mark.asyncio
async def test_tokenize_anthropic_string_input(
    anthropic_client: httpx.AsyncClient,
):
    response = await anthropic_client.post(
        "tokenize",
        params={"api-version": "2024-02-01"},
        json={"inputs": [{"type": "string", "value": "hello"}]},
        headers=_tokenize_headers(
            **{
                "X-UPSTREAM-ENDPOINT": "https://example.com/anthropic/v1/messages",
                "X-UPSTREAM-KEY": "upstream-key",
            }
        ),
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": 5}],
    }


@pytest.mark.asyncio
async def test_tokenize_anthropic_request_input(
    anthropic_client: httpx.AsyncClient,
):
    response = await anthropic_client.post(
        "tokenize",
        params={"api-version": "2024-02-01"},
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
        headers=_tokenize_headers(
            **{
                "X-UPSTREAM-ENDPOINT": "https://example.com/anthropic/v1/messages",
                "X-UPSTREAM-KEY": "upstream-key",
            }
        ),
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": 10}],
    }


@respx.mock
@pytest.mark.asyncio
async def test_tokenize_per_input_error_isolation(
    vllm_client: httpx.AsyncClient,
):
    call_count = 0

    def tokenize_handler(_request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                status_code=200, json={"count": 5, "tokens": [1] * 5}
            )
        return httpx.Response(
            status_code=500, json={"error": "upstream failed"}
        )

    respx.post(_TOKENIZE_URL).mock(side_effect=tokenize_handler)

    response = await vllm_client.post(
        "tokenize",
        json={
            "inputs": [
                {"type": "string", "value": "ok"},
                {"type": "string", "value": "fail"},
            ]
        },
        headers=_tokenize_headers(
            **{"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT}
        ),
    )

    assert response.status_code == 200
    outputs = response.json()["outputs"]
    assert outputs[0] == {"status": "success", "token_count": 5}
    assert outputs[1]["status"] == "error"


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
        headers=_tokenize_headers(
            **{
                "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
                "X-UPSTREAM-EXTRA-DATA": '{"headers_to_proxy": ["x-user-id"]}',
                "x-user-id": "user-42",
            }
        ),
    )

    assert response.status_code == 200
    assert captured["headers"]["x-user-id"] == "user-42"
    assert "authorization" not in {k.lower() for k in captured["headers"]}


@pytest.mark.asyncio
async def test_tokenize_invalid_inputs_raises_422(
    vllm_client: httpx.AsyncClient,
):
    response = await vllm_client.post(
        "tokenize",
        json={"inputs": "not-a-list"},
        headers=_tokenize_headers(
            **{"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT}
        ),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_tokenize_invalid_input_type_returns_422(
    vllm_client: httpx.AsyncClient,
):
    response = await vllm_client.post(
        "tokenize",
        json={"inputs": [{"type": "unknown", "value": "bad"}]},
        headers=_tokenize_headers(
            **{"X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT}
        ),
    )

    assert response.status_code == 422


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tokenize_input", "input_tokens"),
    [
        ({"type": "string", "value": "Hello World!"}, 123),
        ({"type": "string", "value": "Hello World!"}, 456),
    ],
)
async def test_tokenize_to_responses_string_input(
    responses_client: httpx.AsyncClient,
    tokenize_input: dict,
    input_tokens: int,
):
    respx.post(_RESPONSES_INPUT_TOKENS_URL).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "object": "response.input_tokens",
                "input_tokens": input_tokens,
            },
        )
    )

    response = await _post_tokenize_to_responses(
        responses_client, tokenize_input
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": input_tokens}],
    }


@respx.mock
@pytest.mark.asyncio
async def test_tokenize_to_responses_request_input(
    responses_client: httpx.AsyncClient,
):
    input_tokens = 123

    def input_tokens_handler(request: httpx.Request):
        return httpx.Response(
            status_code=200,
            json={
                "object": "response.input_tokens",
                "input_tokens": input_tokens,
            },
        )

    respx.post(_RESPONSES_INPUT_TOKENS_URL).mock(
        side_effect=input_tokens_handler
    )

    response = await _post_tokenize_to_responses(
        responses_client,
        {"type": "request", "value": _CHAT_COMPLETIONS_REQUEST},
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": input_tokens}],
    }


@pytest.mark.parametrize(
    "upstream_endpoint", _UNSUPPORTED_RESPONSES_UPSTREAM_ENDPOINTS
)
@pytest.mark.asyncio
async def test_tokenize_to_unsupported_responses_vendor_returns_404(
    responses_client: httpx.AsyncClient,
    upstream_endpoint: str,
):
    response = await responses_client.post(
        "tokenize",
        json={"inputs": [{"type": "string", "value": "Hello World!"}]},
        headers=_tokenize_headers(
            **{
                **_RESPONSES_TOKENIZE_HEADERS,
                "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            }
        ),
        params=_RESPONSES_TOKENIZE_PARAMS,
    )

    assert response.status_code == 404
    assert (
        response.json()["error"]["message"]
        == _UNSUPPORTED_RESPONSES_VENDOR_ERROR_MESSAGE
    )
