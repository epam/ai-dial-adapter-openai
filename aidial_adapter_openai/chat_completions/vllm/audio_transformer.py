"""
vLLM-specific audio transformer.

Applied as a second pass after the general ``ResourceProcessor`` transformation.
Injects ``audio_url`` content parts from ``MultiModalMessage.audios`` and
converts ``input_audio`` content parts into the ``audio_url`` format
expected by vLLM.

See https://docs.vllm.ai/en/latest/features/multimodal_inputs/?h=audio#audio-inputs_1
"""

from typing import List

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource.audio import (
    AudioResource,
    AudioUrlContentPart,
)
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.validation import ensure_dict, ensure_str


def _convert_input_audio_to_audio_url(part: dict) -> AudioUrlContentPart:
    """Convert an OpenAI ``input_audio`` content part to ``audio_url``."""
    input_audio = ensure_dict("input_audio", part.get("input_audio"))
    data = ensure_str("input_audio.data", input_audio.get("data"))
    fmt = ensure_str("input_audio.format", input_audio.get("format"))
    mime = f"audio/{fmt}"
    resource = Resource.from_base64(type=mime, data_base64=data)
    return AudioResource(audio=resource).to_content_part()


def transform_audio(
    messages: List[MultiModalMessage],
) -> List[MultiModalMessage]:
    """Inject ``audio_url`` parts from audios and convert ``input_audio`` parts."""
    return [_transform_message(m) for m in messages]


def _transform_message(message: MultiModalMessage) -> MultiModalMessage:
    content = message.raw_message.get("content")

    has_audios = bool(message.audios)
    has_input_audio = isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "input_audio" for p in content
    )

    if not has_audios and not has_input_audio:
        return message

    if not isinstance(content, list):
        content = []

    new_content: list = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_audio":
            new_content.append(_convert_input_audio_to_audio_url(part))
        else:
            new_content.append(part)

    for audio in message.audios:
        new_content.append(audio.to_content_part())

    return MultiModalMessage(
        images=message.images,
        files=message.files,
        audios=message.audios,
        raw_message={**message.raw_message, "content": new_content},
    )
