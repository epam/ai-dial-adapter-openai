from openai.types import VideoSeconds, VideoSize
from pydantic import Field

from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel


class VideoGenerationConfig(ExtraAllowedModel):
    seconds: VideoSeconds | int | str | None = Field(
        default=None,
        description="Clip duration in seconds (allowed values: 4, 8, 12). Defaults to 4 seconds.",
    )
    size: VideoSize | str | None = Field(
        default=None,
        description=(
            "Output resolution formatted as width x height "
            "(allowed values: 720x1280, 1280x720, 1024x1792, 1792x1024). "
            "Defaults to 720x1280."
        ),
    )
    auto_crop_reference_images: bool | None = Field(
        default=None,
        description=(
            "Enable auto-cropping of the input reference images to the size of the output video. "
            "Defaults to False."
        ),
    )
