from __future__ import annotations

from typing import Any, List

from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.resource import Resource


class ImageGenPrompt(BaseModel):
    text_prompt: str
    images: List[Resource]

    @classmethod
    async def from_request(
        cls, data: Any, file_storage: FileStorage | None
    ) -> ImageGenPrompt:
        result = await ResourceProcessor(
            file_storage=file_storage
        ).transform_messages(data["messages"])

        text_prompt = ""
        images: List[Resource] = []

        for message in result:
            if content := message.raw_message.get("content"):
                if isinstance(content, str):
                    text_prompt += content
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text_prompt += item["text"]

            for image in message.images:
                images.append(image.image)

        if not text_prompt:
            message = "Text prompt must be provided."
            raise InvalidRequestError(message=message, display_message=message)

        return cls(text_prompt=text_prompt, images=images)
