from __future__ import annotations

import mimetypes
from typing import List, Tuple

from aidial_sdk.exceptions import InvalidRequestError
from httpx._types import RequestFiles

from aidial_adapter_openai.video_generation.azure.types import (
    InpaintItem,
    MediaItemType,
)
from aidial_adapter_openai.video_generation.prompt import VideoGenPrompt


def get_files(prompt: VideoGenPrompt) -> Tuple[List[InpaintItem], RequestFiles]:
    items, files = [], []

    image_idx = 1
    video_idx = 1

    for resource in prompt.resources:
        mime_type = resource.type
        ext = mimetypes.guess_extension(mime_type)

        if "image" in mime_type:
            idx, ty, ext = image_idx, MediaItemType.image, ext or ".png"
            image_idx += 1
        elif "video" in mime_type:
            idx, ty, ext = video_idx, MediaItemType.video, ext or ".mp4"
            video_idx += 1
        else:
            message = f"Unexpected content type of an attachment: {resource.type}. Supported only image and video attachments."
            raise InvalidRequestError(message=message, display_message=message)

        file_name = f"{idx}{ext}"
        items.append(InpaintItem(type=ty, file_name=file_name))
        files.append(("files", (file_name, resource.data, resource.type)))

    return items, files
