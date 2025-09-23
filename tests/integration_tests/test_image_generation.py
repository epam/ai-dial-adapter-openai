from pathlib import Path
from typing import Callable, List
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
    def imagen_deployment(request) -> D:
        return request.param

else:

    @pytest.fixture
    def imagen_deployment(request) -> D:
        pytest.skip("No image generation deployments were found")


@pytest.fixture
def vision_deployment() -> D:
    vision_deployments: List[D] = [
        d
        for d in TEST_DEPLOYMENTS_CONFIG.chat_deployments
        if d.supports_vision and not d.supports_reasoning
    ]

    if vision_deployments:
        return vision_deployments[0]
    else:
        pytest.skip("No vision deployments were found")


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


async def test_text_to_image(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    imagen_deployment: D,
    vision_deployment: D,
    stream: bool,
) -> None:
    if imagen_deployment.type_ == ChatCompletionDeploymentType.DALLE3:
        # DALLE-3 doesn't support n>1
        n = 1
    else:
        n = 2

    response = await chat_completion(
        create_openai_client(imagen_deployment),
        n=n,
        stream=stream,
        deployment_id=imagen_deployment.model_name,
        messages=[user("Generate an image of a cat")],
    )

    assert len(response.response.choices) == n

    for attachments in response.all_attachments:
        image_attachments = [a for a in attachments if a.get("url")]
        assert len(image_attachments) == 1

        evaluation = await chat_completion(
            create_openai_client(vision_deployment),
            stream=False,
            deployment_id=vision_deployment.model_name,
            messages=[
                user(
                    "What animal do you see on the image: tiger, cat, dog, or whale? Reply with a SINGLE word.",
                    custom_content={"attachments": [image_attachments[0]]},
                )
            ],
        )

        assert "cat" in evaluation.content.lower()


async def test_image_to_image(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    vision_deployment: D,
    imagen_deployment: D,
    stream: bool,
) -> None:
    if not imagen_deployment.model_features.imageEditingSupported:
        pytest.skip("Image editing isn't supported by this model")

    response = await chat_completion(
        create_openai_client(imagen_deployment),
        stream=stream,
        deployment_id=imagen_deployment.model_name,
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

        evaluation = await chat_completion(
            create_openai_client(vision_deployment),
            stream=False,
            deployment_id=vision_deployment.model_name,
            messages=[
                user(
                    "What kind of a background do you see on the image: forest, desert, space, or a beach. Reply with a SINGLE word.",
                    custom_content={"attachments": [image_attachments[0]]},
                )
            ],
        )

        assert "space" in evaluation.content.lower()
