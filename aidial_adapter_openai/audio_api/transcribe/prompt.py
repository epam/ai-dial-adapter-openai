from __future__ import annotations

import mimetypes
from typing import Any

from aidial_sdk.exceptions import RequestValidationError
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.request import collect_message_text_content
from aidial_adapter_openai.dial_api.storage import FileStorage


def _normalize_audio_type(audio_type: str) -> str:
    match audio_type.lower().strip():
        case "audio/x-m4a" | "audio/m4a":
            return "audio/mp4"
        case _:
            return audio_type


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

        audios = [audio for message in result for audio in message.audios]

        if not audios:
            msg = "No audio attachment found in the last message"
            raise RequestValidationError(message=msg, display_message=msg)

        if len(audios) > 1:
            msg = "No more than one audio attachment is expected in the last message"
            raise RequestValidationError(message=msg, display_message=msg)

        audio = audios[0]
        audio_data, audio_type = (audio.audio.data, audio.audio.type)
        audio_type = _normalize_audio_type(audio_type)
        fileext = mimetypes.guess_extension(audio_type) or ".mp3"
        audio_filename = f"file{fileext}"

        return cls(
            system_message=system_message,
            audio_data=audio_data,
            audio_type=audio_type,
            audio_filename=audio_filename,
        )
