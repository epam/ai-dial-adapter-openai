from aidial_sdk.exceptions import ResourceNotFoundError
from fastapi import Request

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.utils.request import get_request_app_config


async def configuration(deployment_id: str, request: Request):
    app_config = get_request_app_config(request)
    deployment_type = app_config.get_chat_completion_deployment_type(
        deployment_id
    )

    match deployment_type:
        case (
            ChatCompletionDeploymentType.DALLE3
            | ChatCompletionDeploymentType.GPT_IMAGE_1
        ):
            model = ImageGenerationModel.create(deployment_type)
            return model.get_configuration().schema()
        case _:
            raise ResourceNotFoundError(
                "Configuration endpoint isn't implemented for this deployment"
            )
