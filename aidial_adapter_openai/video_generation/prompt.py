from __future__ import annotations

from typing import Any, List

from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.request import collect_message_text_content
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.resource.base import Resource


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
