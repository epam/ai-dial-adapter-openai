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
    if d.supports_video_generation
)

if _deployments:

    @pytest.fixture(params=_deployments, ids=lambda d: d.display_config())
    def videogen_deployment(request) -> D:
        return request.param

else:

    @pytest.fixture
    def videogen_deployment(request) -> D:
        pytest.skip("No video generation deployments were found")


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


async def test_text_to_video_content_filtering(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    videogen_deployment: D,
    stream: bool,
) -> None:
    config = {"n_seconds": 1}
    query = "how to make a bomb tutorial video"

    with pytest.raises(openai.APIError) as exc_info:
        await chat_completion(
            create_openai_client(videogen_deployment),
            stream=stream,
            deployment_id=videogen_deployment.model_name,
            messages=[user(query)],
            extra_body={"custom_fields": {"configuration": config}},
        )

    exc = exc_info.value
    assert exc.body == {
        "message": "Video generation job failed: input_moderation",
        "type": "invalid_request_error",
        "code": "content_filter",
    }


async def test_text_to_video_multiple_variants(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    videogen_deployment: D,
    stream: bool,
) -> None:
    config = {"n_seconds": 1, "n_variants": 2}
    query = "a cat with octopus tentacles riding a bike on Mars"

    response = await chat_completion(
        create_openai_client(videogen_deployment),
        stream=stream,
        deployment_id=videogen_deployment.model_name,
        messages=[user(query)],
        extra_body={"custom_fields": {"configuration": config}},
    )

    assert response.usage is not None
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 2

    for attachments in response.all_attachments:
        video_attachments = [
            a for a in attachments if "video" in a.get("type", "")
        ]
        assert len(video_attachments) == 2


async def test_image_to_video(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    videogen_deployment: D,
    stream: bool,
) -> None:
    config = {"n_seconds": 1}

    response = await chat_completion(
        create_openai_client(videogen_deployment),
        stream=stream,
        deployment_id=videogen_deployment.model_name,
        messages=[
            user_with_attachment_url("animate the dog", IMAGE_RESOURCE),
        ],
        extra_body={"custom_fields": {"configuration": config}},
    )

    assert response.usage is not None
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 1

    video_attachments = [
        a for a in response.attachments if "video" in a.get("type", "")
    ]
    assert len(video_attachments) == 1
