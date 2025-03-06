import json

import httpx
import pytest
import respx

from aidial_adapter_openai.app_config import ApplicationConfig
from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from tests.conftest import create_test_client
from tests.utils.json import match_objects


async def test_dalle3_configuration_endpoint():
    app_config = ApplicationConfig().add_deployment(
        "app", ChatCompletionDeploymentType.DALLE3
    )

    async with create_test_client(app_config=app_config) as http_client:

        response = await http_client.get(
            "/openai/deployments/app/configuration"
        )

        assert response.status_code == 200
        assert response.json()["properties"].keys() == {
            "quality",
            "size",
            "style",
        }


@respx.mock
@pytest.mark.parametrize(
    "conf",
    [
        None,
        {},
        {"quality": "hd"},
        {
            "quality": "hd",
            "size": "1792x1024",
            "style": "vivid",
        },
        {
            "quality": "standard",
            "size": None,
            "style": None,
        },
        {"quality": "extra-hi-fi"},
        {"negativePrompt": "negative prompt"},
    ],
)
async def test_dalle3_chat_success(conf: dict | None):
    extra_request = {}
    if conf is not None:
        extra_request["custom_fields"] = {"configuration": conf}

    expected_extra_request = {}
    if conf is not None:
        expected_extra_request = {
            k: v for k, v in conf.items() if v is not None
        }

    upstream_api_version = "dalle-3-api-version"
    upstream_endpoint = "http://test-upstream/openai/deployments/upstream-deployment/images/generations"

    app_config = ApplicationConfig(
        DALLE3_AZURE_API_VERSION=upstream_api_version
    ).add_deployment("dalle3-app", ChatCompletionDeploymentType.DALLE3)

    def _mock_response(request: httpx.Request):
        assert json.loads(request.content) == {
            "prompt": "test",
            "response_format": "b64_json",
            **expected_extra_request,
        }
        return httpx.Response(
            status_code=200,
            json={
                "created": 43,
                "data": [
                    {
                        "b64_json": "base64_image",
                        "revised_prompt": "revised prompt",
                    }
                ],
            },
        )

    respx.post(
        f"{upstream_endpoint}?api-version={upstream_api_version}",
    ).mock(side_effect=_mock_response)

    async with create_test_client(app_config=app_config) as http_client:

        response = await http_client.post(
            "/openai/deployments/dalle3-app/chat/completions?api-version=2023-03-15-preview",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
                **extra_request,
            },
            headers={
                "X-UPSTREAM-KEY": "dummy-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            },
        )

        expected_response = {
            "created": 43,
            "id": lambda x: isinstance(x, str),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "custom_content": {
                            "attachments": [
                                {
                                    "title": "Revised prompt",
                                    "data": "revised prompt",
                                },
                                {
                                    "title": "Image",
                                    "type": "image/png",
                                    "data": "base64_image",
                                },
                            ]
                        },
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 1,
                "total_tokens": 1,
            },
        }

        assert response.status_code == 200
        assert match_objects(expected_response, response.json())


@pytest.mark.parametrize("conf", [{"quality": 23}])
async def test_dalle3_configuration_fail(conf: dict | None):
    extra_request = {}
    if conf is not None:
        extra_request["custom_fields"] = {"configuration": conf}

    app_config = ApplicationConfig().add_deployment(
        "app", ChatCompletionDeploymentType.DALLE3
    )

    async with create_test_client(app_config=app_config) as http_client:

        response = await http_client.post(
            "/openai/deployments/app/chat/completions?api-version=2023-03-15-preview",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
                **extra_request,
            },
            headers={
                "X-UPSTREAM-KEY": "dummy-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": "http://test-upstream/openai/deployments/upstream-deployment/images/generations",
            },
        )

        assert response.status_code == 422
        assert response.json() == {
            "error": {
                "code": "422",
                "message": "Invalid request. Path: 'custom_field.configuration.quality', error: unexpected value; permitted: 'standard', 'hd'",
                "type": "invalid_request_error",
            }
        }


@respx.mock
async def test_dalle3_chat_fail():
    upstream_api_version = "dalle-3-api-version"
    upstream_endpoint = "http://test-upstream/openai/deployments/upstream-deployment/images/generations"

    app_config = ApplicationConfig(
        DALLE3_AZURE_API_VERSION=upstream_api_version
    ).add_deployment("dalle3-app", ChatCompletionDeploymentType.DALLE3)

    def _mock_response(request: httpx.Request):
        return httpx.Response(
            status_code=457,
            json={
                "error": {
                    "code": "error.code",
                    "message": "error.message",
                    "param": "error.param",
                    "type": "error.type",
                }
            },
        )

    respx.post(
        f"{upstream_endpoint}?api-version={upstream_api_version}",
    ).mock(side_effect=_mock_response)

    async with create_test_client(app_config=app_config) as http_client:

        response = await http_client.post(
            "/openai/deployments/dalle3-app/chat/completions?api-version=2023-03-15-preview",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
            headers={
                "X-UPSTREAM-KEY": "dummy-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            },
        )

        assert response.status_code == 457
        assert response.json() == {
            "error": {
                "message": "error.message",
                "type": "error.type",
                "param": "error.param",
                "code": "error.code",
            }
        }


@respx.mock
async def test_dalle3_content_filter():
    upstream_api_version = "dalle-3-api-version"
    upstream_endpoint = "http://test-upstream/openai/deployments/upstream-deployment/images/generations"

    app_config = ApplicationConfig(
        DALLE3_AZURE_API_VERSION=upstream_api_version
    ).add_deployment("dalle3-app", ChatCompletionDeploymentType.DALLE3)

    def _mock_response(request: httpx.Request):
        return httpx.Response(
            status_code=457,
            json={
                "error": {
                    "type": "error.type",
                    "code": "content_policy_violation",
                    "message": "error.message",
                }
            },
        )

    respx.post(
        f"{upstream_endpoint}?api-version={upstream_api_version}",
    ).mock(side_effect=_mock_response)

    async with create_test_client(app_config=app_config) as http_client:

        response = await http_client.post(
            "/openai/deployments/dalle3-app/chat/completions?api-version=2023-03-15-preview",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
            headers={
                "X-UPSTREAM-KEY": "dummy-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            },
        )

        assert response.status_code == 457
        assert response.json() == {
            "error": {
                "message": "error.message",
                "type": "error.type",
                "code": "content_filter",
            }
        }
