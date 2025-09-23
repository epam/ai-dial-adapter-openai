from pathlib import Path
from typing import List
from unittest.mock import patch

import openai
import pytest

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import (
    IMAGE_RESOURCE,
    TEST_DEPLOYMENTS_CONFIG,
)
from tests.utils.openai import chat_completion, user, user_with_attachment_url
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
    if d.model_features.imageGenerationSupported
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
    client: openai.AsyncAzureOpenAI, deployment: D, stream: bool
) -> None:
    if deployment.type_ == ChatCompletionDeploymentType.DALLE3:
        # DALLE-3 doesn't support n>1
        n = 1
    else:
        n = 2

    response = await chat_completion(
        client,
        n=n,
        stream=stream,
        deployment_id=deployment.model_name,
        messages=[user("generate an image of a cat")],
    )

    assert len(response.response.choices) == 2

    for attachments in response.all_attachments:
        image_attachments = [a for a in attachments if a.get("url")]
        assert len(image_attachments) == 1


async def test_image_to_image(
    client: openai.AsyncAzureOpenAI, deployment: D, stream: bool
) -> None:
    if not deployment.model_features.imageEditingSupported:
        pytest.skip("Image editing isn't supported by this model")

    response = await chat_completion(
        client,
        stream=stream,
        deployment_id=deployment.model_name,
        messages=[
            user_with_attachment_url(
                "Replace the background with outer space", IMAGE_RESOURCE
            ),
        ],
    )

    assert len(response.response.choices) == 1

    for attachments in response.all_attachments:
        image_attachments = [a for a in attachments if a.get("url")]
        assert len(image_attachments) == 1
