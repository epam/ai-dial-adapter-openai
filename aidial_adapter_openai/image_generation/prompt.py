from __future__ import annotations

from typing import Any

from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.request import collect_message_text_content
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.resource.base import Resource


class ImageGenPrompt(BaseModel):
    text_prompt: str
    images: list[Resource]

    @classmethod
    async def from_request(
        cls, data: Any, file_storage: FileStorage | None
    ) -> ImageGenPrompt:
        result = await ResourceProcessor(
            file_storage=file_storage
        ).transform_messages(data["messages"])

        text_prompt = ""
        images: list[Resource] = []

        for message in result:
            text_prompt += collect_message_text_content(message.raw_message)

            for image in message.images:
                images.append(image.image)

        if not text_prompt:
            message = "Text prompt must be provided."
            raise InvalidRequestError(message=message, display_message=message)

        return cls(text_prompt=text_prompt, images=images)
