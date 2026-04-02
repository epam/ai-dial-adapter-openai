"""
vLLM-specific audio transformer.

Applied as a second pass after the general ``ResourceProcessor`` transformation.
Replaces ``input_audio`` content parts with ``audio_url`` content parts
from ``MultiModalMessage.audios`` in the format expected by vLLM.

See https://docs.vllm.ai/en/latest/features/multimodal_inputs/?h=audio#audio-inputs_1
"""

from typing import List

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage


def _audio_url_part(data_url: str) -> dict:
    return {"type": "audio_url", "audio_url": {"url": data_url}}


def transform_audio(
    messages: List[MultiModalMessage],
) -> List[MultiModalMessage]:
    """Replace ``input_audio`` parts with ``audio_url`` from audios."""
    result = []
    for message in messages:
        if not message.audios:
            result.append(message)
            continue

        content = message.raw_message.get("content")
        new_content = [
            p
            for p in (content if isinstance(content, list) else [])
            if not (isinstance(p, dict) and p.get("type") == "input_audio")
        ]
        new_content.extend(
            _audio_url_part(a.audio.to_data_url()) for a in message.audios
        )

        result.append(
            MultiModalMessage(
                images=message.images,
                files=message.files,
                audios=message.audios,
                raw_message={**message.raw_message, "content": new_content},
            )
        )
    return result
