from typing import Literal

from pydantic import Field

from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel

Voices = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
]
Formats = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class Configuration(ExtraAllowedModel):
    """
    The known fields are passed to the TTS API as such, while the extra
    ones are passed in the request body verbatim, so that the
    provider-specific parameters could reach the upstream.
    """

    instructions: str | None = Field(
        default=None,
        description=(
            "Control the voice of your generated audio with additional instructions. "
            "Does not work with `tts-1` or `tts-1-hd`. "
            "The instruction from the system and developer messages "
            "will be attached to the instructions from the configuration."
        ),
    )
    voice: str | Voices | None = Field(
        default="alloy",
        description="The voice to use when generating the audio.",
    )
    speed: float | None = Field(
        default=None,
        description=(
            "The speed of the generated audio. "
            "Select a value from `0.25` to `4.0`. `1.0` is the default. "
            "Does not work with `gpt-4o-mini-tts`."
        ),
    )
    response_format: str | Formats | None = Field(
        default=None, description="The format of the generated audio."
    )
