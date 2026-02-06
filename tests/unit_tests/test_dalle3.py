import json

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from aidial_adapter_openai.utils.json import remove_nones
from tests.conftest import create_test_client
from tests.utils.json import match_objects
from tests.utils.openai import user

_DALLE_API_VERSION = "dalle-3-api-version"


@pytest.fixture
async def dalle3_client():
    app_config = ApplicationConfig(
        DALLE3_AZURE_API_VERSION=_DALLE_API_VERSION
    ).add_deployment("test-dall-e-3", ChatCompletionDeploymentType.DALLE3)

    async with create_test_client(
        app_config=app_config,
        base_url="http://test-app.com/openai/deployments/test-dall-e-3",
    ) as client:
        yield client


@pytest.mark.parametrize(
    "upstream_endpoint",
    [
        "https://example.com/openai/deployments/test-dalle/images/generations",
        "https://example.com/my/path/openai/deployments/test-dalle/images/generations",
        "https://api.openai.com/v1/images/generations",
    ],
)
async def test_dalle3_configuration_endpoint(
    dalle3_client: httpx.AsyncClient, upstream_endpoint: str
):
    response = await dalle3_client.get(
        "configuration",
        headers={"X-UPSTREAM-ENDPOINT": upstream_endpoint},
    )

    assert response.status_code == 200
    schema = response.json()
    assert set(schema["properties"].keys()) == {"quality", "size", "style"}


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
async def test_dalle3_configuration_success(
    dalle3_client: httpx.AsyncClient, conf: dict | None
):
    def _mock_dalle3_response(request: httpx.Request):
        assert json.loads(request.content) == {
            "model": "test-dall-e-3",
            "prompt": "test",
            "response_format": "b64_json",
            "n": 1,
            **remove_nones(conf or {}),
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

    upstream_endpoint = "http://test-upstream/openai/deployments/upstream-deployment/images/generations"

    respx.post(
        f"{upstream_endpoint}?api-version={_DALLE_API_VERSION}",
    ).mock(side_effect=_mock_dalle3_response)

    extra_body = (
        {} if conf is None else {"custom_fields": {"configuration": conf}}
    )

    response = await dalle3_client.post(
        "chat/completions?api-version=incoming-api-version",
        json={"messages": [user("test")], "stream": False, **extra_body},
        headers={
            "X-UPSTREAM-KEY": "dummy-upstream-api-key",
            "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            "api-key": "dummy-api-key",
        },
    )

    expected_response = {
        "created": 43,
        "model": "test-dall-e-3",
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
async def test_dalle3_invalid_configuration(
    dalle3_client: httpx.AsyncClient, conf: dict
):
    response = await dalle3_client.post(
        "chat/completions?api-version=incoming-api-version",
        json={
            "messages": [user("test")],
            "stream": False,
            "custom_fields": {"configuration": conf},
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
            "message": "Invalid request. Path: 'custom_field.configuration.quality.literal['standard','hd']', error: Input should be 'standard' or 'hd'",
            "type": "invalid_request_error",
        }
    }


@respx.mock
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "status_code, dalle3_response, adapter_response",
    [
        (
            457,
            {
                "error": {
                    "code": "error.code",
                    "message": "error.message",
                    "param": "error.param",
                    "type": "error.type",
                }
            },
            {
                "error": {
                    "code": "error.code",
                    "message": "error.message",
                    "param": "error.param",
                    "type": "error.type",
                }
            },
        ),
        (
            457,
            {
                "error": {
                    "type": "error.type",
                    "code": "content_policy_violation",
                    "message": "error.message",
                }
            },
            {
                "error": {
                    "type": "error.type",
                    "code": "content_filter",
                    "message": "error.message",
                }
            },
        ),
    ],
)
async def test_dalle3_fail(
    dalle3_client: httpx.AsyncClient,
    stream: bool,
    status_code: int,
    dalle3_response: dict,
    adapter_response: dict,
):
    upstream_endpoint = "http://test-upstream/openai/deployments/upstream-deployment/images/generations"

    respx.post(
        f"{upstream_endpoint}?api-version={_DALLE_API_VERSION}",
    ).respond(status_code=status_code, json=dalle3_response)

    response = await dalle3_client.post(
        "chat/completions?api-version=incoming-api-version",
        json={"messages": [user("test")], "stream": stream},
        headers={
            "X-UPSTREAM-KEY": "dummy-upstream-api-key",
            "X-UPSTREAM-ENDPOINT": upstream_endpoint,
        },
    )

    assert response.status_code == status_code
    assert response.json() == adapter_response
