from openai.types.chat import ChatCompletionContentPartInputAudioParam
from pydantic import BaseModel

from aidial_adapter_openai.utils.resource.base import Resource


class AudioResource(BaseModel):
    """
    Audio metadata extracted from an attachment with an ``audio/*`` content type.
    """

    audio: Resource

    @classmethod
    def from_resource(cls, resource: Resource) -> "AudioResource":
        return cls(audio=resource)

    def to_content_part(self) -> ChatCompletionContentPartInputAudioParam:
        """Return the standard OpenAI ``input_audio`` content part."""
        fmt = (
            self.audio.type.split("/", 1)[1]
            if "/" in self.audio.type
            else self.audio.type
        )
        return {
            "type": "input_audio",
            "input_audio": {
                "data": self.audio.data_base64,
                "format": fmt,  # type: ignore[typeddict-item]
            },
        }
