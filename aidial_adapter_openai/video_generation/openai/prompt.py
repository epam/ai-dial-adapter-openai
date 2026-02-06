from __future__ import annotations

import re
from typing import Any, List, Tuple

from aidial_sdk.exceptions import InvalidRequestError
from openai._types import FileTypes
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.request import collect_message_text_content
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.image import crop_image_file
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.image import ImageResource
from aidial_adapter_openai.video_generation.openai.configuration import (
    VideoGenerationConfig,
)


class VideoGenPrompt(BaseModel):
    prompt: str
    images: List[ImageResource]

    @classmethod
    async def from_request(
        cls, data: Any, file_storage: FileStorage | None
    ) -> VideoGenPrompt:
        last_message = data["messages"][-1]

        prompt = collect_message_text_content(last_message).strip()
        if not prompt:
            message = "Text prompt must be provided."
            raise InvalidRequestError(message=message, display_message=message)

        multi_modal_message = (
            await ResourceProcessor(
                file_storage=file_storage
            ).transform_messages([last_message])
        )[0]

        if multi_modal_message.files:
            message = "The deployment only accepts image attachments"
            raise InvalidRequestError(message=message, display_message=message)

        return cls(prompt=prompt, images=multi_modal_message.images)

    def get_last_file(self, config: VideoGenerationConfig) -> FileTypes | None:
        for image in reversed(self.images):
            resource = _crop_image_file(config, image) or image.image
            return ("file", resource.data, resource.type)

        return None


_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


def _parse_video_size(config: VideoGenerationConfig) -> Tuple[int, int] | None:
    size = config.size or "720x1280"
    if m := _SIZE_RE.match(size):
        return int(m.group(1)), int(m.group(2))
    return None


def _crop_image_file(
    config: VideoGenerationConfig, image: ImageResource
) -> Resource | None:
    if not config.auto_crop_reference_images:
        return None

    if (size := _parse_video_size(config)) is None:
        return None

    width, height = size
    if width == image.width and height == image.height:
        return None

    return crop_image_file(resource=image.image, width=width, height=height)
