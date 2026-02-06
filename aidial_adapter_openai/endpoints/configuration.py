from typing import Mapping, Type, assert_never

from aidial_sdk.exceptions import ResourceNotFoundError
from fastapi import Request
from pydantic import BaseModel

from aidial_adapter_openai.audio_api.speech.configuration import (
    Configuration as SpeechConfiguration,
)
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.responses.adapter import ResponsesConfig
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.video_generation.azure.configuration import (
    VideoGenerationConfig,
)


def _get_upstream_endpoint(request_headers: Mapping[str, str]) -> str | None:
    name = "X-UPSTREAM-ENDPOINT"
    return request_headers.get(name)


def _get_deployment_configuration(
    deployment_type: D,
) -> Type[BaseModel] | None:
    match deployment_type:
        case D.DALLE3:
            model = ImageGenerationModel.create(D.DALLE3)
            return model.get_configuration()

        case D.GPT_IMAGE_1:
            model = ImageGenerationModel.create(D.GPT_IMAGE_1)
            return model.get_configuration()

        case D.AZURE_VIDEO_API:
            return VideoGenerationConfig

        case D.AUDIO_SPEECH_API:
            return SpeechConfiguration

        case D.RESPONSES_API:
            return ResponsesConfig

        case (
            D.COMPLETIONS_API
            | D.AUDIO_TRANSCRIPTIONS_API
            | D.GPT4O
            | D.GPT4O_MINI
            | D.MISTRAL
            | D.DATABRICKS
            | D.GPT_GENERIC
        ):
            return None

        case _:
            assert_never(deployment_type)


async def configuration(deployment_id: str, request: Request):
    app_config = get_request_app_config(request)
    upstream_endpoint = _get_upstream_endpoint(request.headers)

    if upstream_endpoint is None:
        raise ResourceNotFoundError(
            "Configuration endpoint requires X-UPSTREAM-ENDPOINT header. "
            "Please upgrade to a newer version of DIAL Core."
        )

    deployment = app_config.get_chat_completion_deployment_type(
        deployment_id, upstream_endpoint
    )
    deployment_type = deployment.deployment_type

    config_model = _get_deployment_configuration(deployment_type)
    if config_model is None:
        raise ResourceNotFoundError(
            "Configuration endpoint isn't implemented for this deployment"
        )

    return config_model.model_json_schema()
