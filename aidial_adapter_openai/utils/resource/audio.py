from typing import TypedDict

from pydantic import BaseModel

from aidial_adapter_openai.utils.resource.base import Resource


class _AudioUrl(TypedDict):
    url: str


class AudioUrlContentPart(TypedDict):
    """
    vLLM-specific content part type for audio, analogous to image_url.
    See https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html
    """

    type: str  # "audio_url"
    audio_url: _AudioUrl


class AudioResource(BaseModel):
    """
    Audio metadata extracted from an attachment with an audio/* content type.
    Produces an ``audio_url`` content part consumable by vLLM.
    """

    audio: Resource

    @classmethod
    def from_resource(cls, resource: Resource) -> "AudioResource":
        return cls(audio=resource)

    def to_content_part(self) -> AudioUrlContentPart:
        return {
            "type": "audio_url",
            "audio_url": {
                "url": self.audio.to_data_url(),
            },
        }

