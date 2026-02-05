from typing import Literal

from openai import Omit, omit
from pydantic import Field
from pydantic import StrictInt as Int
from pydantic import StrictStr as Str

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel


class GptImage1Config(ExtraAllowedModel):
    background: Literal["transparent", "opaque", "auto"] | Str | None = Field(
        default=None,
        description="""
Allows to set transparency for the background of the generated image(s).
Must be one of `transparent`, `opaque` or `auto` (default value).
When `auto` is used, the model will automatically determine the best
background for the image.

If `transparent`, the output format needs to support transparency, so it should
be set to either `png` (default value) or `webp`.
        """.strip(),
    )

    moderation: Literal["low", "auto"] | Str | None = Field(
        default=None,
        description="""
Control the content-moderation level for generated images.
Must be either `low` for less restrictive filtering or `auto` (default value).
""".strip(),
    )

    output_compression: Int | None = Field(
        default=None,
        description="""
The compression level (0-100%) for the generated images. This parameter is only
supported with the `webp` or `jpeg` output formats, and defaults to 100.
""".strip(),
    )

    output_format: Literal["png", "jpeg", "webp"] | str | None = Field(
        default=None,
        description="""
The format in which the generated images are returned.
Must be one of `png`, `jpeg`, or `webp`.
""".strip(),
    )

    quality: Literal["high", "medium", "low"] | Str | None = Field(
        default=None,
        description="The quality of the image that will be generated.",
    )

    size: (
        Literal["1024x1024", "1536x1024", "1024x1536", "auto"] | Str | None
    ) = Field(
        default=None,
        description="""
Must be one of `1024x1024`, `1536x1024` (landscape), `1024x1536` (portrait), or `auto` (default value)
""".strip(),
    )


class GptImage1Model(ImageGenerationModel[GptImage1Config]):
    def get_azure_api_version(self, config: ApplicationConfig) -> str:
        return config.GPT_IMAGE_1_AZURE_API_VERSION

    def get_configuration(self) -> type[GptImage1Config]:
        return GptImage1Config

    def get_response_format(self) -> Omit:
        return omit

    def get_image_content_type(self, config: GptImage1Config) -> str:
        ty = config.output_format or "png"
        return f"image/{ty}"
