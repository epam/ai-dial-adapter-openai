from typing import Literal

from pydantic import BaseModel, Field


class ChunkingStrategyConfig(BaseModel):
    type: Literal["server_vad"] = Field(
        description="Must be set to `server_vad` to enable manual chunking using server side VAD."
    )
    prefix_padding_ms: int = Field(
        description="Amount of audio to include before the VAD detected speech (in milliseconds)."
    )
    silence_duration_ms: int = Field(
        description="Duration of silence to detect speech stop (in milliseconds). "
        "With shorter values the model will respond more quickly, "
        "but may jump in on short pauses from the user."
    )
    threshold: float = Field(
        description="Sensitivity threshold (0.0 to 1.0) for voice activity detection. "
        "A higher threshold will require louder audio to activate the model, "
        "and thus might perform better in noisy environments."
    )


ChunkingStrategy = Literal["auto"] | ChunkingStrategyConfig | None


class Configuration(BaseModel):
    chunking_strategy: ChunkingStrategy = Field(
        default=None,
        description="Controls how the audio is cut into chunks. "
        'When set to "auto", the server first normalizes loudness and then uses '
        "voice activity detection (VAD) to choose boundaries. server_vad object "
        "can be provided to tweak VAD detection parameters manually. If unset, "
        "the audio is transcribed as a single block. Required when using "
        "gpt-4o-transcribe-diarize for inputs longer than 30 seconds.",
    )
