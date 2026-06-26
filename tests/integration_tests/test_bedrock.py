from collections.abc import Callable

import openai
import pytest

from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG
from tests.utils.fixtures import maybe_parametrized_fixture
from tests.utils.openai import chat_completion, user

D = DeploymentConfig

_bedrock_text_deployments: list[D] = [
    deployment
    for deployment in TEST_DEPLOYMENTS_CONFIG.chat_deployments
    if "bedrock-mantle." in deployment.upstream_endpoint
    and not deployment.supports_video_generation
    and not deployment.supports_tts
    and not deployment.supports_stt
    and not deployment.model_features.imageGenerationSupported
]


@maybe_parametrized_fixture(
    params=_bedrock_text_deployments,
    ids=lambda d: d.display_config(),
    skip_reason="No Bedrock text deployments were found",
)
def bedrock_deployment(deployment: D) -> D:
    return deployment


async def test_bedrock_text_with_api_key(
    create_azure_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    bedrock_deployment: D,
):
    if bedrock_deployment.upstream_api_key is None:
        pytest.skip("Deployment has no upstream API key configured")

    response = await chat_completion(
        create_azure_openai_client(
            bedrock_deployment.id_,
            upstream_endpoint=bedrock_deployment.upstream_endpoint,
            upstream_key=bedrock_deployment.upstream_api_key,
        ),
        stream=False,
        deployment_id=bedrock_deployment.model_name,
        messages=[
            user("2+2?"),
        ],
    )

    assert "4" in response.content
    assert response.response.choices[0].finish_reason == "stop"


async def test_bedrock_text_with_provider_auth(
    create_azure_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    bedrock_deployment: D,
):
    response = await chat_completion(
        create_azure_openai_client(
            bedrock_deployment.id_,
            upstream_endpoint=bedrock_deployment.upstream_endpoint,
            upstream_key=None,
        ),
        stream=False,
        deployment_id=bedrock_deployment.model_name,
        messages=[
            user("2+2?"),
        ],
    )

    assert "4" in response.content
    assert response.response.choices[0].finish_reason == "stop"
