from typing import Literal

from aidial_sdk.pydantic_v1 import Field
from aidial_sdk.pydantic_v1 import StrictStr as Str

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel


class Dalle3Config(ExtraAllowedModel):
    quality: Literal["standard", "hd"] | Str | None = Field(
        default=None,
        description="The quality of the image that will be generated.",
    )

    size: Literal["1024x1024", "1792x1024", "1024x1792"] | Str | None = Field(
        default=None, description="The size of the generated images."
    )

    style: Literal["vivid", "natural"] | Str | None = Field(
        default=None, description="The style of the generated images."
    )


class Dalle3Model(ImageGenerationModel[Dalle3Config]):
    def get_azure_api_version(self, config: ApplicationConfig) -> str:
        return config.DALLE3_AZURE_API_VERSION

    def get_configuration(self) -> type[Dalle3Config]:
        return Dalle3Config

    def get_response_format(self) -> Literal["b64_json"]:
        return "b64_json"

    def get_image_content_type(self, config: Dalle3Config) -> str:
        return "image/png"
