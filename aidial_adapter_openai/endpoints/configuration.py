from aidial_sdk.exceptions import ResourceNotFoundError
from fastapi import Request

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.utils.request import get_request_app_config


async def configuration(deployment_id: str, request: Request):
    app_config = get_request_app_config(request)

    if deployment_id in app_config.DALLE3_DEPLOYMENTS:
        model = ImageGenerationModel.create(D.DALLE3)
    elif deployment_id in app_config.GPT_IMAGE_1_DEPLOYMENTS:
        model = ImageGenerationModel.create(D.GPT_IMAGE_1)
    else:
        raise ResourceNotFoundError(
            "Configuration endpoint isn't implemented for this deployment"
        )

    return model.get_configuration().model_json_schema()
