from pydantic import Field

from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel

_supported_dimensions = "The following dimensions are supported: 480x480, 854x480, 720x720, 1280x720, 1080x1080 and 1920x1080 in both landscape and portrait orientations."


class VideoGenerationConfig(ExtraAllowedModel):
    """Modelled following the official spec:
    https://github.com/Azure/azure-rest-api-specs/blob/main/specification/ai/data-plane/OpenAI.v1/azure-v1-preview-generated.yaml#L6730
    """

    width: int = Field(
        default=480,
        description=f"The height of the video. {_supported_dimensions}",
    )
    height: int = Field(
        default=480,
        description=f"The width of the video. {_supported_dimensions}",
    )
    n_seconds: int | None = Field(
        default=None,
        description="The duration of the video generation job. Must be between 1 and 20 seconds.",
    )
    n_variants: int | None = Field(
        default=None,
        description="The number of videos to create as variants for this job. Must be between 1 and 5. Smaller dimensions allow more variants.",
    )
