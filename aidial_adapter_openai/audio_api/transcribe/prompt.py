from __future__ import annotations

import mimetypes
from typing import Any, List

from aidial_sdk.exceptions import RequestValidationError
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.request import collect_message_text_content
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.resource.file import FileResource


def _collect_system_messages(messages: list[dict]) -> str | None:
    ret = ""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            ret += collect_message_text_content(message)
    return ret.strip() or None


class TranscribePrompt(BaseModel):
    system_message: str | None
    audio_data: bytes
    audio_type: str
    audio_filename: str

    @classmethod
    async def from_request(
        cls, request: Any, file_storage: FileStorage | None
    ) -> TranscribePrompt:
        if (n := request.get("n")) not in [None, 1]:
            raise RequestValidationError(
                f"The deployment doesn't support request.n parameter other than 1, but got {n}."
            )

        messages = request["messages"]

        if not messages:
            raise RequestValidationError(
                "The request doesn't contain any messages"
            )

        result = await ResourceProcessor(
            file_storage=file_storage
        ).transform_messages(messages[-1:])

        system_message = _collect_system_messages(messages)

        audios: List[FileResource] = []

        for message in result:
            for file in message.files:
                if file.resource.type.startswith("audio"):
                    audios.append(file)

        if not audios:
            raise RequestValidationError(
                "No audio attachment found in the last message"
            )

        audio = audios[0]
        audio_data, audio_type = (audio.resource.data, audio.resource.type)
        fileext = mimetypes.guess_extension(audio_type) or ".mp3"
        audio_filename = f"file{fileext}"

        return cls(
            system_message=system_message,
            audio_data=audio_data,
            audio_type=audio_type,
            audio_filename=audio_filename,
        )
