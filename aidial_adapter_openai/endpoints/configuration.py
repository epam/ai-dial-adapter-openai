from typing import assert_never

from aidial_adapter_anthropic.adapter._claude.config import (
    ClaudeConfigurationWithThinking,
)
from aidial_sdk.exceptions import ResourceNotFoundError
from fastapi import Request
from pydantic import BaseModel

from aidial_adapter_openai.audio_api.speech.configuration import (
    Configuration as SpeechConfiguration,
)
from aidial_adapter_openai.audio_api.transcribe.configuration import (
    Configuration as TranscribeConfiguration,
)
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.dial_api.request import (
    DeploymentId,
    get_upstream_endpoint,
)
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.responses.configuration import ResponsesConfig
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.video_generation.azure.configuration import (
    VideoGenerationConfig as AzureVideoGenerationConfig,
)
from aidial_adapter_openai.video_generation.openai.configuration import (
    VideoGenerationConfig as OpenAIVideoGenerationConfig,
)


def _get_deployment_configuration(deployment_type: D) -> type[BaseModel] | None:
    match deployment_type:
        case D.DALLE3:
            model = ImageGenerationModel.create(D.DALLE3)
            return model.get_configuration()

        case D.GPT_IMAGE_1:
            model = ImageGenerationModel.create(D.GPT_IMAGE_1)
            return model.get_configuration()

        case D.OPENAI_VIDEO_API:
            return OpenAIVideoGenerationConfig

        case D.AZURE_VIDEO_API:
            return AzureVideoGenerationConfig

        case D.AUDIO_SPEECH_API:
            return SpeechConfiguration

        case D.RESPONSES_API:
            return ResponsesConfig

        case D.ANTHROPIC_MESSAGES_API:
            return ClaudeConfigurationWithThinking

        case D.AUDIO_TRANSCRIPTIONS_API:
            return TranscribeConfiguration

        case (
            D.COMPLETIONS_API
            | D.GPT4O
            | D.GPT4O_MINI
            | D.MISTRAL
            | D.DATABRICKS
            | D.GPT_GENERIC
            | D.VLLM_CHAT_COMPLETIONS_API
            | D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API
        ):
            return None

        case _:
            assert_never(deployment_type)


async def configuration(deployment_id: DeploymentId, request: Request) -> dict:
    app_config = get_request_app_config(request)
    upstream_endpoint = get_upstream_endpoint(request.headers)
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
