from typing import List

import openai
import pytest

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.integration_tests.base import (
    DeploymentConfig,
    EmbeddingsDeploymentType,
    sanitize_id_part,
)
from tests.integration_tests.constants import (
    IMAGE_RESOURCE,
    TEST_DEPLOYMENTS_CONFIG,
)

D = DeploymentConfig[EmbeddingsDeploymentType]

_deployments: List[D] = list(TEST_DEPLOYMENTS_CONFIG.embedding_deployments)


@pytest.fixture
def app_config() -> ApplicationConfig:
    return TEST_DEPLOYMENTS_CONFIG.app_config


if _deployments:

    def _display_config(deployment: D) -> str:
        upstream_idx = deployment.upstream_idx
        parts = [
            sanitize_id_part(deployment.id_),
            *([] if upstream_idx is None else [f"upstream:{upstream_idx}"]),
        ]
        return "/".join(parts)

    @pytest.fixture(params=_deployments, ids=_display_config)
    def deployment(request) -> D:
        return request.param

else:

    @pytest.fixture
    def deployment(request) -> D:
        pytest.skip("No embedding deployments were found")


@pytest.fixture
def client(create_openai_client, deployment: D) -> openai.AsyncAzureOpenAI:
    return create_openai_client(deployment)


@pytest.fixture
def model_name(deployment: D) -> str:
    return deployment.model_name


async def test_embeddings_single_text_input(
    model_name: str, client: openai.AsyncAzureOpenAI
):
    response = await client.embeddings.create(model=model_name, input="cat")
    assert len(response.data) == 1


async def test_embeddings_two_text_inputs(
    model_name: str, client: openai.AsyncAzureOpenAI
):
    response = await client.embeddings.create(
        model=model_name, input=["cat", "fish"]
    )
    assert len(response.data) == 2


@pytest.fixture
def image_input_supported(
    model_name: str, app_config: ApplicationConfig
) -> bool:
    if model_name in app_config.AZURE_AI_VISION_DEPLOYMENTS:
        return True
    pytest.skip("Embeddings doesn't support images")


async def test_embeddings_image_input(
    image_input_supported, model_name: str, client: openai.AsyncAzureOpenAI
):
    resource = IMAGE_RESOURCE
    image_attachment = {"type": resource.type, "data": resource.data_base64}
    response = await client.embeddings.create(
        model=model_name,
        input=[],
        extra_body={"custom_input": [image_attachment]},
    )
    assert len(response.data) == 1
