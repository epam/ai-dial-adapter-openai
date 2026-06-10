import base64
import json

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.embeddings.vllm.builders import (
    BuilderKind,
    build_upstream_body,
    select_builder,
)
from aidial_adapter_openai.embeddings.vllm.mode import (
    VllmEmbeddingMode,
    detect_mode,
)
from aidial_adapter_openai.embeddings.vllm.response import to_embedding_response
from tests.conftest import create_test_client
from tests.integration_tests.constants import IMAGE_RESOURCE

_UPSTREAM_EMBEDDINGS = "http://localhost:5001/v1/embeddings"
_UPSTREAM_POOLING = "http://localhost:5001/pooling"
_API_VERSION = "2023-03-15-preview"


@pytest.fixture
def vllm_app_config() -> ApplicationConfig:
    return ApplicationConfig(
        VLLM_EMBEDDINGS_DEPLOYMENTS=[
            "embeddinggemma",
            "qwen3-vl-embed",
            "nemotron-colembed-4b",
        ]
    )


def test_detect_mode_sequence():
    assert detect_mode(_UPSTREAM_EMBEDDINGS) == VllmEmbeddingMode.SEQUENCE


def test_detect_mode_token_embed():
    assert detect_mode(_UPSTREAM_POOLING) == VllmEmbeddingMode.POOLING


def test_select_builder_embeddinggemma():
    assert (
        select_builder("google/embeddinggemma-300m", VllmEmbeddingMode.SEQUENCE)
        == BuilderKind.TEXT_INPUT
    )


def test_select_builder_qwen3_vl():
    assert (
        select_builder("Qwen3-VL-Embedding-2B", VllmEmbeddingMode.SEQUENCE)
        == BuilderKind.QWEN3_VL
    )


def test_select_builder_qwen3_embedding_text():
    assert (
        select_builder("Qwen/Qwen3-Embedding-8B", VllmEmbeddingMode.SEQUENCE)
        == BuilderKind.TEXT_INPUT
    )


def test_select_builder_colembed_from_mode():
    assert (
        select_builder("any-model", VllmEmbeddingMode.POOLING)
        == BuilderKind.COLEMBED
    )


async def test_build_qwen3_embedding_text_body():
    body = await build_upstream_body(
        request={"encoding_format": "float", "dimensions": 1024},
        model="Qwen/Qwen3-Embedding-8B",
        input_item="hello",
        builder=BuilderKind.TEXT_INPUT,
    )
    assert body["input"] == "hello"
    assert body["dimensions"] == 1024
    assert "messages" not in body


async def test_build_qwen3_vl_text_body():
    body = await build_upstream_body(
        request={"encoding_format": "float"},
        model="Qwen3-VL-Embedding-2B",
        input_item="hello",
        builder=BuilderKind.QWEN3_VL,
    )
    assert body["continue_final_message"] is True
    assert body["add_special_tokens"] is True
    assert body["messages"][1]["content"] == [{"type": "text", "text": "hello"}]


async def test_build_colembed_image_body():
    body = await build_upstream_body(
        request={},
        model="nvidia/nemotron-colembed-vl-4b-v2",
        input_item=IMAGE_RESOURCE,
        builder=BuilderKind.COLEMBED,
    )
    assert body["task"] == "token_embed"
    assert body["messages"][0]["content"][0]["type"] == "image_url"
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_token_embed_response_mean_pool():
    response = to_embedding_response(
        model="nemotron-colembed-vl-4b-v2",
        responses=[
            {
                "data": [
                    {
                        "data": [
                            [1.0, 0.0],
                            [3.0, 2.0],
                        ]
                    }
                ]
            }
        ],
        mode=VllmEmbeddingMode.POOLING,
    )
    assert response.data[0].embedding == [2.0, 1.0]


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
async def test_vllm_colembed_pooling_text(vllm_app_config: ApplicationConfig):
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
    captured: dict = {}

    def handler(request: httpx.Request):
        captured["x-user-id"] = request.headers.get("x-user-id")
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
