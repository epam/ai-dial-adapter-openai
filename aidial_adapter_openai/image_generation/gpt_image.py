from typing import Literal, Optional

from aidial_sdk.pydantic_v1 import Field, StrictStr
from openai import NOT_GIVEN, NotGiven
from pydantic import BaseModel

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel


class GptImage1Config(ExtraAllowedModel):
    quality: Optional[Literal["standard", "hd"] | StrictStr] = Field(
        default=None,
        description="The quality of the image that will be generated.",
    )

    size: Optional[
        Literal["1024x1024", "1792x1024", "1024x1792"] | StrictStr
    ] = Field(default=None, description="The size of the generated images.")

    style: Optional[Literal["vivid", "natural"] | StrictStr] = Field(
        default=None, description="The style of the generated images."
    )


class GptImage1Model(ImageGenerationModel):
    def get_azure_api_version(self, config: ApplicationConfig) -> str:
        return config.GPT_IMAGE_1_AZURE_API_VERSION

    def get_configuration(self) -> type[BaseModel]:
        return GptImage1Config

    def get_response_format(self) -> NotGiven:
        return NOT_GIVEN
