import pytest
from aioresponses import CallbackResult, aioresponses

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

    expected_extra_response = {}
    if conf is not None:
        expected_extra_response = {
            k: v for k, v in conf.items() if v is not None
        }

    upstream_api_version = "dalle-3-api-version"
    app_config = ApplicationConfig(
        DALLE3_AZURE_API_VERSION=upstream_api_version
    ).add_deployment("app", ChatCompletionDeploymentType.DALLE3)

    with aioresponses() as aio_mock:

        def callback(url, json, **kwargs):
            assert json == {
                "prompt": "test",
                "response_format": "b64_json",
                **expected_extra_response,
            }
            return CallbackResult(
                status=200,
                payload={
                    "created": 43,
                    "data": [
                        {
                            "b64_json": "base64_image",
                            "revised_prompt": "revised prompt",
                        }
                    ],
                },
            )

        aio_mock.add(
            method="POST",
            url=f"http://test-upstream/?api-version={upstream_api_version}",
            callback=callback,
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
                    "X-UPSTREAM-ENDPOINT": "http://test-upstream",
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
                    "completion_tokens": 1,
                    "prompt_tokens": 0,
                    "total_tokens": 1,
                },
            }

            assert response.status_code == 200
            assert match_objects(expected_response, response.json())


@pytest.mark.parametrize("conf", [{"quality": 23}])
async def test_dalle3_chat_fail(conf: dict | None):
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
                "X-UPSTREAM-ENDPOINT": "http://test-upstream",
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
