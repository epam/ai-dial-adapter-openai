"""
GPT-4o Audio Completions middleware for handling audio responses.

The module provides helpers that extract audio data and transcripts
from GPT-4o audio completions API responses, transforming them into
DIAL-compatible format with attachments and stages.
"""

from typing import AsyncIterator, Set, TypeVar

from pydantic import BaseModel

from aidial_adapter_openai.utils.streaming import map_stream

AUDIO_FORMAT_TO_CONTENT_TYPE = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "pcm16": "audio/L16",
}


class _AudioResponseTransformer(BaseModel):
    opened_transcript_stages: Set[int] = set()
    """Indices of choices where an audio transcript stage was opened."""

    streaming: bool
    audio_format: str = "mp3"

    @property
    def message_key(self) -> str:
        return "delta" if self.streaming else "message"

    @property
    def content_type(self) -> str:
        return AUDIO_FORMAT_TO_CONTENT_TYPE.get(
            self.audio_format, f"audio/{self.audio_format}"
        )

    def __call__(self, chunk: dict) -> dict:
        choices = chunk.get("choices")
        if not choices:
            return chunk

        for choice in choices:
            message = choice.get(self.message_key)
            if not message:
                continue

            audio_obj = message.pop("audio", None)
            if not audio_obj:
                continue

            # Extract audio data and transcript
            audio_data = audio_obj.get("data")
            audio_transcript = audio_obj.get("transcript")

            # Early exit if audio object is empty
            if not audio_data and not audio_transcript:
                continue

            cc = message.setdefault("custom_content", {})

            # Create attachment for audio data
            if audio_data:
                attachments = cc.setdefault("attachments", [])
                attachments.append(
                    {
                        "title": "Audio",
                        "type": self.content_type,
                        "data": audio_data,
                    }
                )

            # Create stage for audio transcript
            if audio_transcript:
                choice_index = choice.get("index")
                is_opening = choice_index not in self.opened_transcript_stages
                is_closing = choice.get("finish_reason") is not None

                if is_opening:
                    self.opened_transcript_stages.add(choice_index)

                stages = cc.setdefault("stages", [])

                opening_fields = (
                    {"name": "Audio transcript"} if is_opening else {}
                )
                closing_fields = {"status": "completed"} if is_closing else {}
                streaming_fields = {"index": 0} if self.streaming else {}
                content_fields = {"content": audio_transcript}

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
    Extract audio data and transcript from GPT-4o audio response.

    This middleware transforms the OpenAI audio response format into DIAL format by:
    - Creating attachments for audio data
    - Placing audio transcripts in a dedicated "Audio transcript" stage
    - Preserving the main response content

    Args:
        response: Either a dict (non-streaming) or AsyncIterator[dict] (streaming)
        request_body: The original request body to extract audio format from

    Returns:
        The transformed response in the same format as input
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
