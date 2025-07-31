from typing import List

import openai
import pytest

from tests.integration_tests.base import (
    DeploymentConfig,
    EmbeddingsDeploymentType,
    sanitize_id_part,
)
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG

D = DeploymentConfig[EmbeddingsDeploymentType]

_deployments: List[D] = list(TEST_DEPLOYMENTS_CONFIG.embedding_deployments)


def _display_config(deployment: D) -> str:
    upstream_idx = deployment.upstream_idx
    parts = [
        sanitize_id_part(deployment.id_),
        *([] if upstream_idx is None else [f"upstream:{upstream_idx}"]),
    ]
    return "/".join(parts)


if _deployments:

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
