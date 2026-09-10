"""
The Azure OpenAI v1 API endpoints carry no deployment id in the path,
so the upstream deployment id comes from the X-DIAL-DEPLOYMENT-ID header.
"""

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

_API_VERSION = "2024-02-01"
_GPT_DEPLOYMENT = "gpt-test"
_DALLE_DEPLOYMENT = "dalle-test"

_CHAT_COMPLETIONS_UPSTREAM = "http://test-upstream/v1/chat/completions"
_EMBEDDINGS_UPSTREAM = "http://test-upstream/v1/embeddings"
_DALLE_UPSTREAM = (
    "http://test-upstream/openai/deployments/dalle-test/images/generations"
)


# (X-DIAL-DEPLOYMENT-ID, X-DIAL-OVERRIDE-NAME, deployment id sent upstream)
_DEPLOYMENT_ID_MATRIX = [
    ("dial-deployment-id", None, "dial-deployment-id"),
    ("dial-deployment-id", "override-name", "override-name"),
]

# The `model` field of the request sent to the adapter. It's untrusted:
# it must not affect the deployment id sent upstream.
_REQUEST_MODELS = [None, "foobar"]


def _headers(
    deployment_id: str,
    upstream_endpoint: str,
    override_name: str | None = None,
) -> dict[str, str]:
    headers = {
        "Api-Key": "test-adapter-api-key",
        "X-DIAL-DEPLOYMENT-ID": deployment_id,
        "X-UPSTREAM-KEY": "test-upstream-api-key",
        "X-UPSTREAM-ENDPOINT": upstream_endpoint,
    }
    if override_name is not None:
        headers["X-DIAL-OVERRIDE-NAME"] = override_name
    return headers


def _upstream_model() -> str:
    return json.loads(respx.calls.last.request.content)["model"]


@pytest.fixture
async def client():
    app_config = (
        ApplicationConfig(TIKTOKEN_MODEL_MAPPING={_GPT_DEPLOYMENT: "gpt-4o"})
        .add_deployment(
            _GPT_DEPLOYMENT, ChatCompletionDeploymentType.GPT_GENERIC
        )
        .add_deployment(_DALLE_DEPLOYMENT, ChatCompletionDeploymentType.DALLE3)
    )
    async with create_test_client(
        app_config=app_config, base_url="http://test-app.com/openai/v1"
    ) as client:
        yield client


@respx.mock
@pytest.mark.parametrize("request_model", _REQUEST_MODELS)
@pytest.mark.parametrize(
    "dial_deployment_id,override_name,expected_upstream_id",
    _DEPLOYMENT_ID_MATRIX,
)
async def test_chat_completions(
    client: httpx.AsyncClient,
    dial_deployment_id: str,
    override_name: str | None,
    expected_upstream_id: str,
    request_model: str | None,
):
    respx.post(_CHAT_COMPLETIONS_UPSTREAM).respond(
        json=OpenAIStream(
            single_choice_chunk(
                delta={"role": "assistant", "content": "5"},
                finish_reason="stop",
            )
        ).to_block_response()
    )

    response = await client.post(
        "chat/completions",
        params={"api-version": _API_VERSION},
        json={
            "model": request_model,
            "messages": [{"role": "user", "content": "2+3=?"}],
        },
        headers=_headers(
            dial_deployment_id, _CHAT_COMPLETIONS_UPSTREAM, override_name
        ),
    )

    assert response.status_code == 200
    assert _upstream_model() == expected_upstream_id


@respx.mock
@pytest.mark.parametrize("request_model", _REQUEST_MODELS)
@pytest.mark.parametrize(
    "dial_deployment_id,override_name,expected_upstream_id",
    _DEPLOYMENT_ID_MATRIX,
)
async def test_embeddings(
    client: httpx.AsyncClient,
    dial_deployment_id: str,
    override_name: str | None,
    expected_upstream_id: str,
    request_model: str | None,
):
    respx.post(_EMBEDDINGS_UPSTREAM).respond(
        json={
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
            ],
            "model": expected_upstream_id,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )

    response = await client.post(
        "embeddings",
        params={"api-version": _API_VERSION},
        json={"model": request_model, "input": "hello"},
        headers=_headers(
            dial_deployment_id, _EMBEDDINGS_UPSTREAM, override_name
        ),
    )

    assert response.status_code == 200
    assert _upstream_model() == expected_upstream_id


async def test_tokenize(client: httpx.AsyncClient):
    response = await client.post(
        "tokenize",
        json={"inputs": [{"type": "string", "value": "hello"}]},
        headers=_headers(_GPT_DEPLOYMENT, _CHAT_COMPLETIONS_UPSTREAM),
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "token_count": 1}],
    }


async def test_truncate_prompt(client: httpx.AsyncClient):
    response = await client.post(
        "truncate_prompt",
        json={
            "inputs": [
                {
                    "max_prompt_tokens": 100,
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ]
        },
        headers=_headers(_GPT_DEPLOYMENT, _CHAT_COMPLETIONS_UPSTREAM),
    )

    assert response.status_code == 200
    assert response.json() == {
        "outputs": [{"status": "success", "discarded_messages": []}],
    }


async def test_configuration(client: httpx.AsyncClient):
    response = await client.get(
        "configuration",
        headers=_headers(_DALLE_DEPLOYMENT, _DALLE_UPSTREAM),
    )

    assert response.status_code == 200
    assert set(response.json()["properties"]) == {"quality", "size", "style"}
