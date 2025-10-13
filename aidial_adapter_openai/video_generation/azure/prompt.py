from __future__ import annotations

import mimetypes
from typing import Any, List, Tuple

from aidial_sdk.exceptions import InvalidRequestError
from httpx._types import RequestFiles
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.request import collect_message_text_content
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.video_generation.azure.types import (
    InpaintItem,
    MediaItemType,
)


class VideoGenPrompt(BaseModel):
    prompt: str
    resources: List[Resource]

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

        resources: List[Resource] = []
        for image in multi_modal_message.images:
            resources.append(image.image)

        for file in multi_modal_message.files:
            resources.append(file.resource)

        return cls(prompt=prompt, resources=resources)

    def get_files(self) -> Tuple[List[InpaintItem], RequestFiles]:
        items, files = [], {}

        image_idx = 1
        video_idx = 1

        for resource in self.resources:
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
                raise InvalidRequestError(
                    message=message, display_message=message
                )

            file_name = f"{idx}{ext}"
            items.append(InpaintItem(type=ty, file_name=file_name))
            files[file_name] = (file_name, resource.data, resource.type)

        return items, files
