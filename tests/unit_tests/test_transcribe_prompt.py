import pytest

from aidial_adapter_openai.audio_api.transcribe.prompt import (
    _normalize_audio_type,
)


@pytest.mark.parametrize(
    ("audio_type", "expected_audio_type"),
    [
        ("audio/x-m4a", "audio/mp4"),
        ("audio/m4a", "audio/mp4"),
        ("audio/mp4", "audio/mp4"),
    ],
)
def test_normalize_audio_type(
    audio_type: str, expected_audio_type: str
) -> None:
    assert _normalize_audio_type(audio_type) == expected_audio_type
