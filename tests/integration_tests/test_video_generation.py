from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Generic,
    List,
    Literal,
    TypeGuard,
    TypeVar,
    assert_never,
)
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
from tests.utils.openai import (
    ChatCompletionResult,
    chat_completion,
    user,
    user_with_attachment_url,
)
from tests.utils.storage import MockFileStorage


@pytest.fixture(autouse=True)
def mock_storage(request):
    test_name = request.node.name
    root_dir = Path(__file__).parent / "mock-storage" / test_name
    with (
        MockFileStorage.create(root_dir) as storage,
        patch(
            "aidial_adapter_openai.endpoints.chat_completion.create_file_storage",
            return_value=storage,
        ),
    ):
        yield storage


VideoGenType = Literal[
    ChatCompletionDeploymentType.AZURE_VIDEO_API,
    ChatCompletionDeploymentType.OPENAI_VIDEO_API,
]


D = DeploymentConfig[VideoGenType]


def _is_video_gen_type(
    d: DeploymentConfig[ChatCompletionDeploymentType],
) -> TypeGuard[D]:
    return d.type_ in (
        ChatCompletionDeploymentType.AZURE_VIDEO_API,
        ChatCompletionDeploymentType.OPENAI_VIDEO_API,
    )


_deployments: List[D] = [
    d for d in TEST_DEPLOYMENTS_CONFIG.chat_deployments if _is_video_gen_type(d)
]

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


_T = TypeVar("_T")


@dataclass
class _ConfigParam(Generic[_T]):
    name: str
    value: _T

    def to_dict(self) -> dict:
        return {self.name: self.value}


@pytest.fixture
def seconds_param(videogen_deployment: D) -> _ConfigParam[int]:
    ty = videogen_deployment.type_
    if ty == ChatCompletionDeploymentType.OPENAI_VIDEO_API:
        return _ConfigParam("seconds", 4)
    if ty == ChatCompletionDeploymentType.AZURE_VIDEO_API:
        return _ConfigParam("n_seconds", 1)
    assert_never(ty)


@pytest.fixture
def variants_param(videogen_deployment: D) -> _ConfigParam[int]:
    ty = videogen_deployment.type_
    if ty == ChatCompletionDeploymentType.OPENAI_VIDEO_API:
        pytest.skip("OpenAI Video API doesn't support variant parameter")
    if ty == ChatCompletionDeploymentType.AZURE_VIDEO_API:
        return _ConfigParam("n_variants", 1)
    assert_never(ty)


@pytest.fixture
def extra_params(videogen_deployment: D) -> dict:
    ty = videogen_deployment.type_
    if ty == ChatCompletionDeploymentType.OPENAI_VIDEO_API:
        return {"auto_crop_reference_images": True}
    return {}


async def test_text_to_video_content_filtering(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    videogen_deployment: D,
    seconds_param: _ConfigParam[int],
    stream: bool,
) -> None:
    query = "how to make a bomb tutorial video"

    with pytest.raises(openai.APIError) as exc_info:
        await chat_completion(
            create_openai_client(videogen_deployment),
            stream=stream,
            deployment_id=videogen_deployment.model_name,
            messages=[user(query)],
            extra_body={
                "custom_fields": {"configuration": seconds_param.to_dict()}
            },
        )

    err = exc_info.value.body
    assert err is not None
    assert err["message"].startswith("Video generation job failed")  # type:ignore
    assert err["type"] == "invalid_request_error"  # type:ignore
    assert err["code"] == "content_filter"  # type:ignore


async def test_text_to_video_single_variant(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    videogen_deployment: D,
    seconds_param: _ConfigParam[int],
    stream: bool,
) -> None:
    query = "a cat with octopus tentacles riding a bike on Mars"

    response = await chat_completion(
        create_openai_client(videogen_deployment),
        stream=stream,
        deployment_id=videogen_deployment.model_name,
        messages=[user(query)],
        extra_body={
            "custom_fields": {"configuration": seconds_param.to_dict()}
        },
    )

    assert response.usage is not None
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == seconds_param.value

    _check_video_attachments(response, 1)


async def test_text_to_video_multiple_variants(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    videogen_deployment: D,
    seconds_param: _ConfigParam[int],
    variants_param: _ConfigParam[int],
    stream: bool,
) -> None:
    config = seconds_param.to_dict() | variants_param.to_dict()
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
    assert (
        response.usage.completion_tokens
        == seconds_param.value * variants_param.value
    )

    _check_video_attachments(response, variants_param.value)


async def test_image_to_video(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    videogen_deployment: D,
    seconds_param: _ConfigParam[int],
    stream: bool,
    extra_params: dict,
) -> None:
    config = seconds_param.to_dict() | extra_params

    response = await chat_completion(
        create_openai_client(videogen_deployment),
        stream=stream,
        deployment_id=videogen_deployment.model_name,
        messages=[
            user_with_attachment_url(
                "make the dog kick the bowl and jump into the camera; no slow-mo; sharp focus throughout",
                IMAGE_RESOURCE,
            ),
        ],
        extra_body={"custom_fields": {"configuration": config}},
    )

    assert response.usage is not None
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == seconds_param.value

    _check_video_attachments(response, 1)


def _check_video_attachments(
    response: ChatCompletionResult, num_attachments: int
) -> None:
    all_attachments = response.all_attachments
    assert len(all_attachments) == 1
    attachments = all_attachments[0]
    assert len(attachments) == num_attachments
    for a in attachments:
        assert "video" in a.get("type", "")
