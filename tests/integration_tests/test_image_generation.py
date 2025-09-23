from pathlib import Path
from typing import List
from unittest.mock import patch

import openai
import pytest

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG
from tests.utils.openai import chat_completion, user
from tests.utils.storage import MockFileStorage


@pytest.fixture(autouse=True)
def mock_storage(request):
    test_name = request.node.name
    root_dir = Path(__file__).parent / "mock-storage" / test_name
    with MockFileStorage.create(root_dir) as storage:
        with patch(
            "aidial_adapter_openai.endpoints.chat_completion.create_file_storage",
            return_value=storage,
        ):
            yield storage


D = DeploymentConfig[ChatCompletionDeploymentType]

_deployments: List[D] = list(
    d
    for d in TEST_DEPLOYMENTS_CONFIG.chat_deployments
    if d.model_features.oneShotImageGenerationSupported
)

if _deployments:

    @pytest.fixture(params=_deployments, ids=lambda d: d.display_config())
    def deployment(request) -> D:
        return request.param

else:

    @pytest.fixture
    def deployment(request) -> D:
        pytest.skip("No image generation deployments were found")


@pytest.fixture
def client(create_openai_client, deployment: D) -> openai.AsyncAzureOpenAI:
    return create_openai_client(deployment)


@pytest.fixture
def deployment_id(deployment: D) -> str:
    return deployment.model_name


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


async def test_text_to_image(
    client: openai.AsyncAzureOpenAI, deployment_id: str, stream: bool
) -> None:
    response = await chat_completion(
        client,
        stream=stream,
        deployment_id=deployment_id,
        messages=[user("Generate an image of a cat")],
    )

    image_attachments = [a for a in response.attachments if a.get("url")]
    assert len(image_attachments) == 1
