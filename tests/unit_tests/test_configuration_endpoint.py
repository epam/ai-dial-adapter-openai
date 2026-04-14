import pytest

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.conftest import create_test_client


@pytest.mark.parametrize(
    "deployment_type,endpoint_suffix,expected_properties",
    [
        (
            ChatCompletionDeploymentType.DALLE3,
            "images/generations",
            {"quality", "size", "style"},
        ),
        (
            ChatCompletionDeploymentType.GPT_IMAGE_1,
            "images/generations",
            {
                "background",
                "moderation",
                "output_compression",
                "output_format",
                "quality",
                "size",
            },
        ),
        (
            ChatCompletionDeploymentType.AZURE_VIDEO_API,
            "videos",
            {"seconds", "size", "auto_crop_reference_images"},
        ),
        (
            ChatCompletionDeploymentType.AZURE_VIDEO_API,
            "video/generations",
            {"width", "height", "n_seconds", "n_variants"},
        ),
        (
            ChatCompletionDeploymentType.AUDIO_SPEECH_API,
            "audio/speech",
            {"instructions", "voice", "speed", "response_format"},
        ),
        (
            ChatCompletionDeploymentType.RESPONSES_API,
            "responses",
            {"reasoning"},
        ),
        (
            ChatCompletionDeploymentType.ANTHROPIC_MESSAGES_API,
            "anthropic/v1/messages",
            {"betas", "thinking", "enable_citations"},
        ),
    ],
)
async def test_configuration_endpoint_supported_types(
    deployment_type: ChatCompletionDeploymentType,
    endpoint_suffix: str,
    expected_properties: set[str],
):
    deployment_id = f"test-{deployment_type.value.lower()}"
    app_config = ApplicationConfig().add_deployment(
        deployment_id, deployment_type
    )

    async with create_test_client(
        app_config=app_config,
        base_url=f"http://test-app.com/openai/deployments/{deployment_id}",
    ) as client:
        upstream_endpoint = f"http://test-upstream/openai/deployments/{deployment_id}/{endpoint_suffix}"

        response = await client.get(
            "configuration",
            headers={"X-UPSTREAM-ENDPOINT": upstream_endpoint},
        )

        assert response.status_code == 200
        schema = response.json()
        assert "properties" in schema
        assert set(schema["properties"].keys()) == expected_properties


@pytest.mark.parametrize(
    "deployment_type,endpoint_suffix",
    [
        (ChatCompletionDeploymentType.GPT_GENERIC, "chat/completions"),
        (ChatCompletionDeploymentType.GPT4O, "chat/completions"),
        (ChatCompletionDeploymentType.GPT4O_MINI, "chat/completions"),
        (ChatCompletionDeploymentType.MISTRAL, "chat/completions"),
        (ChatCompletionDeploymentType.DATABRICKS, "chat/completions"),
        (ChatCompletionDeploymentType.COMPLETIONS_API, "completions"),
        (
            ChatCompletionDeploymentType.AUDIO_TRANSCRIPTIONS_API,
            "audio/transcriptions",
        ),
    ],
)
async def test_configuration_endpoint_unsupported_types(
    deployment_type: ChatCompletionDeploymentType,
    endpoint_suffix: str,
):
    deployment_id = f"test-{deployment_type.value.lower()}"
    app_config = ApplicationConfig().add_deployment(
        deployment_id, deployment_type
    )

    async with create_test_client(
        app_config=app_config,
        base_url=f"http://test-app.com/openai/deployments/{deployment_id}",
    ) as client:
        upstream_endpoint = f"http://test-upstream/openai/deployments/{deployment_id}/{endpoint_suffix}"

        response = await client.get(
            "configuration",
            headers={"X-UPSTREAM-ENDPOINT": upstream_endpoint},
        )

        assert response.status_code == 404
        assert (
            "Configuration endpoint isn't implemented"
            in response.json()["error"]["message"]
        )
