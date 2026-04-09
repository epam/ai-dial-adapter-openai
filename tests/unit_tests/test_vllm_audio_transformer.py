import pytest

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.chat_completions.vllm.audio_transformer import (
    transform_audio,
)
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource.base import Resource
from tests.utils.storage import DummyFileStorage

_audio_wav = Resource(type="audio/wav", data=b"RIFF\x00\x00\x00\x00WAVEfmt ")
_audio_mp3 = Resource(type="audio/mpeg", data=b"\xff\xfb\x90\x00dummy mp3")


def _attachment(resource: Resource) -> dict:
    return {"type": resource.type, "data": resource.data_base64}


def _audio_url_part(resource: Resource) -> dict:
    return {
        "type": "audio_url",
        "audio_url": {"url": resource.to_data_url()},
    }


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


async def _general_transform(messages: list[dict]) -> list[MultiModalMessage]:
    return await ResourceProcessor(
        file_storage=DummyFileStorage(),
    ).transform_messages(messages)


@pytest.mark.parametrize(
    "audio,text",
    [
        (_audio_wav, "transcribe this"),
        (_audio_mp3, ""),
    ],
)
async def test_audio_attachment_converted_to_audio_url(audio, text):
    message = {
        "role": "user",
        "content": text,
        "custom_content": {"attachments": [_attachment(audio)]},
    }
    general = await _general_transform([message])
    result = transform_audio(general)

    assert len(result) == 1
    assert result[0].raw_message["content"] == [
        _text_part(text),
        _audio_url_part(audio),
    ]


async def test_input_audio_converted_to_audio_url():
    wav_b64 = _audio_wav.data_base64
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": wav_b64, "format": "wav"},
                },
            ],
        },
    ]
    general = await _general_transform(messages)
    result = transform_audio(general)

    assert len(result) == 1
    content = result[0].raw_message["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "transcribe this"}
    assert content[1]["type"] == "audio_url"
    assert content[1]["audio_url"]["url"] == f"data:audio/wav;base64,{wav_b64}"


async def test_no_audio_passthrough():
    messages = [{"role": "user", "content": "Hello"}]
    general = await _general_transform(messages)
    result = transform_audio(general)

    assert result[0] is general[0]
