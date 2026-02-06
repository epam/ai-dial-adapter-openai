import re
from typing import Tuple

from openai._types import FileTypes

from aidial_adapter_openai.utils.image import crop_image_file
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.video_generation.openai.configuration import (
    VideoGenerationConfig,
)
from aidial_adapter_openai.video_generation.prompt import VideoGenPrompt


def get_last_file(
    self: VideoGenPrompt, config: VideoGenerationConfig
) -> FileTypes | None:
    for resource in reversed(self.resources):
        if config.auto_crop_reference_images:
            resource = _crop_image_file(config, resource) or resource

        return ("file", resource.data, resource.type)

    return None


_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


def _parse_video_size(config: VideoGenerationConfig) -> Tuple[int, int] | None:
    size = config.size or "720x1280"
    if m := _SIZE_RE.match(size):
        return int(m.group(1)), int(m.group(2))
    return None


def _crop_image_file(
    config: VideoGenerationConfig, resource: Resource
) -> Resource | None:
    if size := _parse_video_size(config):
        return crop_image_file(resource=resource, width=size[0], height=size[1])
    return None
