"""
GPT-4o Audio Completions middleware for handling audio responses.

The module provides helpers that extract audio data and transcripts
from GPT-4o audio completions API responses, transforming them into
DIAL-compatible format with attachments and stages.
"""

from typing import AsyncIterator, TypeVar

from pydantic import BaseModel

from aidial_adapter_openai.utils.streaming import map_stream

_AUDIO_FORMAT_TO_CONTENT_TYPE = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "pcm16": "audio/L16",
}


class _AudioResponseTransformer(BaseModel):
    opened_transcript_stages: set[int] = set()
    opened_transcript_attachments: set[int] = set()

    streaming: bool
    audio_format: str

    @property
    def message_key(self) -> str:
        return "delta" if self.streaming else "message"

    @property
    def content_type(self) -> str:
        return _AUDIO_FORMAT_TO_CONTENT_TYPE.get(
            self.audio_format, f"audio/{self.audio_format}"
        )

    def __call__(self, chunk: dict) -> dict:
        choices = chunk.get("choices") or []
        for choice in choices:
            message = choice.get(self.message_key)
            if not message:
                continue

            audio_obj = message.get("audio")
            if not audio_obj:
                continue

            audio_data = audio_obj.get("data")
            audio_transcript = audio_obj.get("transcript")

            if not audio_data and not audio_transcript:
                continue

            cc = message.setdefault("custom_content", {})

            if audio_data:
                choice_index = choice.get("index")
                is_attachment_opening = (
                    choice_index not in self.opened_transcript_attachments
                )

                if is_attachment_opening:
                    self.opened_transcript_attachments.add(choice_index)

                attachments = cc.setdefault("attachments", [])

                opening_fields = (
                    {"title": "Audio", "type": self.content_type}
                    if is_attachment_opening
                    else {}
                )
                streaming_fields = {"index": 0} if self.streaming else {}
                data_fields = {"data": audio_data}

                attachments.append(
                    {
                        **data_fields,
                        **streaming_fields,
                        **opening_fields,
                    }
                )

            choice_index = choice.get("index")
            has_finish_reason = choice.get("finish_reason") is not None
            is_stage_opened = choice_index in self.opened_transcript_stages

            if audio_transcript or (has_finish_reason and is_stage_opened):
                is_opening = choice_index not in self.opened_transcript_stages

                if is_opening:
                    self.opened_transcript_stages.add(choice_index)

                stages = cc.setdefault("stages", [])

                opening_fields = (
                    {"name": "Audio transcript"} if is_opening else {}
                )
                closing_fields = (
                    {"status": "completed"} if has_finish_reason else {}
                )
                streaming_fields = {"index": 0} if self.streaming else {}
                content_fields = (
                    {"content": audio_transcript} if audio_transcript else {}
                )

                stages.append(
                    {
                        **content_fields,
                        **streaming_fields,
                        **opening_fields,
                        **closing_fields,
                    }
                )

        return chunk


_T = TypeVar("_T", bound=AsyncIterator[dict] | dict)


def extract_audio_content(response: _T, request_body: dict) -> _T:
    """
    Transform OpenAI audio response format into DIAL format by:
    - Creating attachments for audio data
    - Placing audio transcripts in a dedicated "Audio transcript" stage
    - Preserving the main response content
    """
    audio_format = request_body.get("audio", {}).get("format", "mp3")

    if isinstance(response, dict):
        return _AudioResponseTransformer(
            streaming=False, audio_format=audio_format
        )(response)
    else:
        return map_stream(
            _AudioResponseTransformer(
                streaming=True, audio_format=audio_format
            ),
            response,
        )
