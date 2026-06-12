import base64
import json

import httpx
import pytest
import respx
from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.embeddings.vllm.api_type import (
    EmbeddingAPIType,
    select_api_type,
)
from aidial_adapter_openai.embeddings.vllm.openai_api import (
    OpenAIEmbeddingsAdapter,
)
from aidial_adapter_openai.embeddings.vllm.pooling_api import (
    PoolingEmbeddingsAdapter,
    VllmPoolingDataItem,
    VllmPoolingResponse,
)
from aidial_adapter_openai.embeddings.vllm.qwen3_vl_api import (
    Qwen3VLEmbeddingsAdapter,
)
from tests.conftest import create_test_client
from tests.integration_tests.constants import IMAGE_RESOURCE

_UPSTREAM_EMBEDDINGS = "http://localhost:5001/v1/embeddings"
_UPSTREAM_POOLING = "http://localhost:5001/pooling"
_API_VERSION = "2023-03-15-preview"


@pytest.fixture
def vllm_app_config() -> ApplicationConfig:
    return ApplicationConfig(
        VLLM_DEPLOYMENTS=[
            "embeddinggemma",
            "qwen3-vl-embed",
            "nemotron-colembed-4b",
        ]
    )


def test_select_api_type_openai_embeddings():
    assert (
        select_api_type("google/embeddinggemma-300m", _UPSTREAM_EMBEDDINGS)
        == EmbeddingAPIType.OPENAI_EMBEDDINGS
    )


def test_select_api_type_pooling():
    assert (
        select_api_type("any-model", _UPSTREAM_POOLING)
        == EmbeddingAPIType.POOLING
    )


def test_select_api_type_qwen3_vl():
    assert (
        select_api_type("Qwen3-VL-Embedding-2B", _UPSTREAM_EMBEDDINGS)
        == EmbeddingAPIType.QWEN3_VL_EMBEDDINGS
    )


def test_select_api_type_qwen3_embedding_text():
    assert (
        select_api_type("Qwen/Qwen3-Embedding-8B", _UPSTREAM_EMBEDDINGS)
        == EmbeddingAPIType.OPENAI_EMBEDDINGS
    )


async def test_openai_adapter_text_body():
    request = EmbeddingsRequest.model_validate(
        {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": "hello",
            "encoding_format": "float",
            "dimensions": 1024,
        }
    )
    adapter = OpenAIEmbeddingsAdapter(
        request=request,
        model="Qwen/Qwen3-Embedding-8B",
        endpoint=_UPSTREAM_EMBEDDINGS,
        creds={},
        headers=None,
    )
    body = adapter.build_body("hello")
    dumped = body.model_dump(exclude_none=True)
    assert dumped["input"] == "hello"
    assert dumped["dimensions"] == 1024
    assert "messages" not in dumped


async def test_qwen3_vl_adapter_text_body():
    request = EmbeddingsRequest.model_validate(
        {
            "model": "Qwen3-VL-Embedding-2B",
            "input": "hello",
            "encoding_format": "float",
        }
    )
    adapter = Qwen3VLEmbeddingsAdapter(
        request=request,
        model="Qwen3-VL-Embedding-2B",
        endpoint=_UPSTREAM_EMBEDDINGS,
        creds={},
        headers=None,
    )
    body = await adapter.build_body("hello")
    dumped = body.model_dump(exclude_none=True)
    assert dumped["continue_final_message"] is True
    assert dumped["add_special_tokens"] is True
    assert dumped["messages"][1]["content"] == [
        {"type": "text", "text": "hello"}
    ]


async def test_pooling_adapter_image_body():
    adapter = PoolingEmbeddingsAdapter(
        model="nvidia/nemotron-colembed-vl-4b-v2",
        endpoint=_UPSTREAM_POOLING,
        creds={},
        headers=None,
    )
    body = await adapter.build_body(IMAGE_RESOURCE)
    dumped = body.model_dump(exclude_none=True)
    assert dumped["task"] == "token_embed"
    assert dumped["messages"][0]["content"][0]["type"] == "image_url"
    assert dumped["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_pooling_adapter_response_mean_pool():
    embedding = VllmPoolingResponse(
        data=[VllmPoolingDataItem(data=[[1.0, 0.0], [3.0, 2.0]])]
    ).to_embedding(index=0)
    assert embedding.embedding == [2.0, 1.0]


@respx.mock
async def test_vllm_embeddinggemma_text_batch(
    vllm_app_config: ApplicationConfig,
):
    def handler(request: httpx.Request):
        assert request.url == _UPSTREAM_EMBEDDINGS
        payload = json.loads(request.content)
        assert payload["input"] == ["cat", "fish"]
        return httpx.Response(
            status_code=200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1], "index": 0},
                    {"object": "embedding", "embedding": [0.2], "index": 1},
                ],
                "model": "google/embeddinggemma-300m",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    respx.post(_UPSTREAM_EMBEDDINGS).mock(side_effect=handler)

    async with create_test_client(vllm_app_config) as client:
        response = await client.post(
            f"/openai/deployments/embeddinggemma/embeddings?api-version={_API_VERSION}",
            json={
                "model": "google/embeddinggemma-300m",
                "input": ["cat", "fish"],
            },
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": _UPSTREAM_EMBEDDINGS,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["embedding"] == [0.1]
    assert body["data"][1]["embedding"] == [0.2]


@respx.mock
async def test_vllm_qwen3_vl_custom_input_image(
    vllm_app_config: ApplicationConfig,
):
    captured: dict = {}

    def handler(request: httpx.Request):
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            status_code=200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.5, 0.6],
                        "index": 0,
                    }
                ],
                "model": "Qwen3-VL-Embedding-2B",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    respx.post(_UPSTREAM_EMBEDDINGS).mock(side_effect=handler)

    image_attachment = {
        "type": IMAGE_RESOURCE.type,
        "data": IMAGE_RESOURCE.data_base64,
    }

    async with create_test_client(vllm_app_config) as client:
        response = await client.post(
            f"/openai/deployments/qwen3-vl-embed/embeddings?api-version={_API_VERSION}",
            json={
                "model": "Qwen3-VL-Embedding-2B",
                "input": [],
                "custom_input": [image_attachment],
            },
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": _UPSTREAM_EMBEDDINGS,
            },
        )

    assert response.status_code == 200, response.text
    messages = captured["json"]["messages"]
    image_url = messages[1]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]) == IMAGE_RESOURCE.data


@respx.mock
async def test_vllm_pooling_text(vllm_app_config: ApplicationConfig):
    captured: dict = {}

    def handler(request: httpx.Request):
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {
                        "data": [
                            [1.0, 0.0],
                            [1.0, 2.0],
                        ]
                    }
                ]
            },
        )

    respx.post(_UPSTREAM_POOLING).mock(side_effect=handler)

    async with create_test_client(vllm_app_config) as client:
        response = await client.post(
            f"/openai/deployments/nemotron-colembed-4b/embeddings?api-version={_API_VERSION}",
            json={
                "model": "nvidia/nemotron-colembed-vl-4b-v2",
                "input": "invoice total",
            },
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": _UPSTREAM_POOLING,
            },
        )

    assert response.status_code == 200, response.text
    assert captured["json"]["task"] == "token_embed"
    assert captured["json"]["input"] == "invoice total"
    assert response.json()["data"][0]["embedding"] == [1.0, 1.0]


@respx.mock
async def test_vllm_embeddinggemma_single_text(
    vllm_app_config: ApplicationConfig,
):
    respx.post(_UPSTREAM_EMBEDDINGS).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1], "index": 0}
                ],
                "model": "google/embeddinggemma-300m",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    )

    async with create_test_client(vllm_app_config) as client:
        response = await client.post(
            f"/openai/deployments/embeddinggemma/embeddings?api-version={_API_VERSION}",
            json={"model": "google/embeddinggemma-300m", "input": "hello"},
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": _UPSTREAM_EMBEDDINGS,
            },
        )

    assert response.status_code == 200, response.text


@respx.mock
async def test_vllm_proxy_headers(vllm_app_config: ApplicationConfig):
    def handler(request: httpx.Request):
        assert request.headers.get("x-user-id") == "user-1"
        return httpx.Response(
            status_code=200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1], "index": 0}
                ],
                "model": "google/embeddinggemma-300m",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    respx.post(_UPSTREAM_EMBEDDINGS).mock(side_effect=handler)

    async with create_test_client(vllm_app_config) as client:
        response = await client.post(
            f"/openai/deployments/embeddinggemma/embeddings?api-version={_API_VERSION}",
            json={"model": "google/embeddinggemma-300m", "input": "hello"},
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": _UPSTREAM_EMBEDDINGS,
                "X-UPSTREAM-EXTRA-DATA": '{"headers_to_proxy": ["x-user-id"]}',
                "x-user-id": "user-1",
            },
        )

    assert response.status_code == 200
