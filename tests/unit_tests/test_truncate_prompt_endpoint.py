import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.conftest import create_test_client

_API_KEY = "test-adapter-api-key"
_GPT_UPSTREAM_ENDPOINT = (
    "https://example.com/openai/deployments/gpt-test/chat/completions"
)
_VLLM_UPSTREAM_ENDPOINT = "http://localhost:5001/v1/chat/completions"
_VLLM_TOKENIZE_URL = "http://localhost:5001/tokenize"
_DALLE_UPSTREAM_ENDPOINT = (
    "https://example.com/openai/deployments/dalle-test/images/generations"
)

# "this is four tokens" -> 4 content tokens + 3 per-message + 1 role = 8 tokens.
# The empty request base costs 3 tokens (TOKENS_PER_REQUEST).
_FOUR_TOKENS = "this is four tokens"


def _headers(upstream_endpoint: str) -> dict[str, str]:
    return {"Api-Key": _API_KEY, "X-UPSTREAM-ENDPOINT": upstream_endpoint}


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
async def dalle_client():
    config = ApplicationConfig().add_deployment(
        "dalle-test", ChatCompletionDeploymentType.DALLE3
    )
    async with create_test_client(
        app_config=config,
        base_url="http://test-app.com/openai/deployments/dalle-test",
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_truncate_prompt_gpt_fits(gpt_client: httpx.AsyncClient):
    response = await gpt_client.post(
        "truncate_prompt",
        json={
            "inputs": [
                {
                    "max_prompt_tokens": 100,
                    "messages": [
                        {"role": "system", "content": _FOUR_TOKENS},
                        {"role": "user", "content": _FOUR_TOKENS},
                    ],
                }
            ]
        },
        headers=_headers(_GPT_UPSTREAM_ENDPOINT),
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "discarded_messages": []}],
    }


@pytest.mark.asyncio
async def test_truncate_prompt_gpt_over_budget(gpt_client: httpx.AsyncClient):
    response = await gpt_client.post(
        "truncate_prompt",
        json={
            "inputs": [
                {
                    "max_prompt_tokens": 25,
                    "messages": [
                        {"role": "system", "content": _FOUR_TOKENS},
                        {"role": "user", "content": _FOUR_TOKENS},
                        {"role": "user", "content": _FOUR_TOKENS},
                        {"role": "user", "content": _FOUR_TOKENS},
                    ],
                }
            ]
        },
        headers=_headers(_GPT_UPSTREAM_ENDPOINT),
    )

    assert response.status_code == 200
    output = response.json()["outputs"][0]
    assert output["status"] == "success"
    # System (index 0) and last user message (index 3) are always retained.
    assert output["discarded_messages"] == [1, 2]


@pytest.mark.asyncio
async def test_truncate_prompt_missing_max_prompt_tokens_is_isolated(
    gpt_client: httpx.AsyncClient,
):
    response = await gpt_client.post(
        "truncate_prompt",
        json={
            "inputs": [
                {
                    "max_prompt_tokens": 100,
                    "messages": [{"role": "user", "content": _FOUR_TOKENS}],
                },
                {
                    "messages": [{"role": "user", "content": _FOUR_TOKENS}],
                },
            ]
        },
        headers=_headers(_GPT_UPSTREAM_ENDPOINT),
    )

    assert response.status_code == 200
    outputs = response.json()["outputs"]
    assert outputs[0] == {"status": "success", "discarded_messages": []}
    assert outputs[1]["status"] == "error"
    assert "max_prompt_tokens" in outputs[1]["error"]


@pytest.mark.asyncio
async def test_truncate_prompt_unsupported_deployment_returns_404(
    dalle_client: httpx.AsyncClient,
):
    response = await dalle_client.post(
        "truncate_prompt",
        json={
            "inputs": [
                {
                    "max_prompt_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ]
        },
        headers=_headers(_DALLE_UPSTREAM_ENDPOINT),
    )

    assert response.status_code == 404


@respx.mock
@pytest.mark.asyncio
async def test_truncate_prompt_vllm_over_budget(
    vllm_client: httpx.AsyncClient,
):
    counts = iter([100, 80, 40])

    def tokenize_handler(_request: httpx.Request):
        return httpx.Response(status_code=200, json={"count": next(counts)})

    respx.post(_VLLM_TOKENIZE_URL).mock(side_effect=tokenize_handler)

    response = await vllm_client.post(
        "truncate_prompt",
        json={
            "inputs": [
                {
                    "max_prompt_tokens": 50,
                    "messages": [
                        {"role": "user", "content": "one"},
                        {"role": "user", "content": "two"},
                        {"role": "user", "content": "three"},
                        {"role": "user", "content": "four"},
                    ],
                }
            ]
        },
        headers=_headers(_VLLM_UPSTREAM_ENDPOINT),
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "discarded_messages": [0, 1]}],
    }


@pytest.mark.asyncio
async def test_truncate_prompt_invalid_inputs_returns_422(
    gpt_client: httpx.AsyncClient,
):
    response = await gpt_client.post(
        "truncate_prompt",
        json={"inputs": "not-a-list"},
        headers=_headers(_GPT_UPSTREAM_ENDPOINT),
    )

    assert response.status_code == 422
