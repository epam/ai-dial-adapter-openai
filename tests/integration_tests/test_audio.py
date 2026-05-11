from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from aidial_adapter_openai.utils.resource.base import Resource
from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import (
    TEST_DEPLOYMENTS_CONFIG,
    AUDIO_11s_RESOURCE,
    AUDIO_39s_RESOURCE,
)
from tests.utils.fixtures import maybe_parametrized_fixture
from tests.utils.openai import (
    chat_completion,
    user,
    user_with_attachment_data,
    user_with_attachment_url,
)
from tests.utils.storage import MockFileStorage
from tests.utils.string import is_close_enough


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


D = DeploymentConfig[ChatCompletionDeploymentType]

_tts_deployments: list[D] = [
    d for d in TEST_DEPLOYMENTS_CONFIG.chat_deployments if d.supports_tts
]

if _tts_deployments:

    @pytest.fixture(params=_tts_deployments, ids=lambda d: d.display_config())
    def tts_deployment(request) -> D:
        return request.param

else:

    @pytest.fixture
    def tts_deployment(request) -> D:
        pytest.skip("No TTS deployments were found")


_stt_deployments: list[D] = [
    d for d in TEST_DEPLOYMENTS_CONFIG.chat_deployments if d.supports_stt
]


def _is_diarization_deployment(deployment: D) -> bool:
    return "diarize" in deployment.model_name.lower()


_stt_diarize_deployments: list[D] = [
    d for d in _stt_deployments if _is_diarization_deployment(d)
]


@maybe_parametrized_fixture(
    params=_stt_deployments,
    ids=lambda d: d.display_config(),
    skip_reason="No STT deployments were found",
)
def stt_deployment(deployment: D) -> D:
    return deployment


@maybe_parametrized_fixture(
    params=_stt_diarize_deployments,
    ids=lambda d: d.display_config(),
    skip_reason="No diarization STT deployments were found",
)
def diarize_stt_deployment(deployment: D) -> D:
    return deployment


@pytest.fixture
def any_stt_deployment() -> D:
    if _stt_deployments:
        return _stt_deployments[0]
    else:
        pytest.skip("No STT deployments were found")


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


@pytest.fixture(
    params=[user_with_attachment_url, user_with_attachment_data],
    ids=["attachment-data-url", "attachment-data"],
)
def message_with_attachment(request):
    return request.param


@pytest.fixture()
def text_query() -> str:
    return "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world."


@pytest.fixture()
def stt_expected_transcription(stt_audio_resource: Resource) -> str:
    if stt_audio_resource is AUDIO_11s_RESOURCE:
        return "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world."

    if stt_audio_resource is AUDIO_39s_RESOURCE:
        return "Call me Ishmael. Some years ago ... never mind how long precisely ... having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.\nWhenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet."

    pytest.fail(f"Unknown STT audio resource: {stt_audio_resource}")


@pytest.fixture(
    params=[AUDIO_11s_RESOURCE, AUDIO_39s_RESOURCE],
    ids=["audio-11s", "audio-39s"],
)
def stt_audio_resource(request) -> Resource:
    return request.param


async def test_text_to_speech_and_back(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    tts_deployment: DeploymentConfig,
    any_stt_deployment: D,
    stream: bool,
    text_query: str,
):
    response = await chat_completion(
        create_openai_client(tts_deployment),
        stream=stream,
        deployment_id=tts_deployment.model_name,
        messages=[user(text_query)],
    )

    for attachments in response.all_attachments:
        audio_attachments = [
            a for a in attachments if "audio" in a.get("type", "")
        ]
        assert len(audio_attachments) == 1

        evaluation = await chat_completion(
            create_openai_client(any_stt_deployment),
            stream=False,
            deployment_id=any_stt_deployment.model_name,
            messages=[
                user("", custom_content={"attachments": [audio_attachments[0]]})
            ],
        )

        assert is_close_enough(text_query, evaluation.content)


async def test_speech_to_text(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    stt_deployment: DeploymentConfig,
    stt_expected_transcription: str,
    stream: bool,
    stt_audio_resource: Resource,
    message_with_attachment: Callable[
        [str, Resource], ChatCompletionMessageParam
    ],
):
    response = await chat_completion(
        create_openai_client(stt_deployment),
        stream=stream,
        deployment_id=stt_deployment.model_name,
        messages=[message_with_attachment(" ", stt_audio_resource)],
    )
    assert is_close_enough(stt_expected_transcription, response.content)


async def test_diarize_long_audio_without_chunking_fails(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    diarize_stt_deployment: D,
    stream: bool,
):
    with pytest.raises(openai.APIError) as exc_info:
        await chat_completion(
            create_openai_client(diarize_stt_deployment),
            stream=stream,
            deployment_id=diarize_stt_deployment.model_name,
            messages=[user_with_attachment_data(" ", AUDIO_39s_RESOURCE)],
            extra_body={
                "custom_fields": {"configuration": {"chunking_strategy": None}}
            },
        )

    err = exc_info.value.body or {}
    assert getattr(exc_info.value, "status_code", None) == 400
    assert "chunking_strategy is required" in str(err)
