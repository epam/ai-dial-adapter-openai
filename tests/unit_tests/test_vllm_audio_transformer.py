import base64

import pytest

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.chat_completions.vllm.audio_transformer import (
    transform_audio,
)
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource.audio import AudioResource
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.file import FileResource
from aidial_adapter_openai.utils.resource.image import ImageResource
from tests.utils.images import pic_1_1
from tests.utils.storage import DummyFileStorage

_file_1 = Resource(type="application/pdf", data=b"document content 1")

_audio_wav = Resource(type="audio/wav", data=b"RIFF\x00\x00\x00\x00WAVEfmt ")
_audio_mp3 = Resource(type="audio/mpeg", data=b"\xff\xfb\x90\x00dummy mp3")


def attachment(resource: Resource) -> dict:
    return {"type": resource.type, "data": resource.data_base64}


def image_resource(resource: Resource, w: int, h: int) -> ImageResource:
    return ImageResource(width=w, height=h, detail="low", image=resource)


def file_resource(resource: Resource) -> FileResource:
    return FileResource(name="data attachment", resource=resource)


def audio_resource(resource: Resource) -> AudioResource:
    return AudioResource(audio=resource)


def audio_part(resource: Resource) -> dict:
    return {
        "type": "audio_url",
        "audio_url": {"url": resource.to_data_url()},
    }


def image_part(resource: Resource) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": resource.to_data_url(), "detail": "low"},
    }


def file_part(resource: Resource) -> dict:
    return {
        "type": "file",
        "file": {
            "filename": "data attachment",
            "file_data": resource.to_data_url(),
        },
    }


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


async def _general_transform(messages: list[dict]) -> list[MultiModalMessage]:
    return await ResourceProcessor(
        file_storage=DummyFileStorage(),
    ).transform_messages(messages)


@pytest.mark.parametrize(
    "message,expected_content",
    [
        (
            {
                "role": "user",
                "content": "transcribe this",
                "custom_content": {
                    "attachments": [attachment(_audio_wav)],
                },
            },
            [
                text_part("transcribe this"),
                audio_part(_audio_wav),
            ],
        ),
        (
            {
                "role": "user",
                "content": "",
                "custom_content": {
                    "attachments": [attachment(_audio_mp3)],
                },
            },
            [
                text_part(""),
                audio_part(_audio_mp3),
            ],
        ),
        (
            {
                "role": "user",
                "content": "mixed",
                "custom_content": {
                    "attachments": [
                        attachment(pic_1_1),
                        attachment(_audio_wav),
                    ],
                },
            },
            [
                text_part("mixed"),
                image_part(pic_1_1),
                audio_part(_audio_wav),
            ],
        ),
    ],
)
async def test_audio_attachment_converted_to_audio_url(
    message,
    expected_content,
):
    general = await _general_transform([message])
    result = transform_audio(general)

    assert len(result) == 1
    assert result[0].raw_message.get("custom_content") is None
    assert result[0].raw_message["content"] == expected_content


async def test_audio_not_in_files_list():
    messages = [
        {
            "role": "user",
            "content": "listen",
            "custom_content": {
                "attachments": [
                    attachment(_audio_wav),
                    attachment(_file_1),
                ],
            },
        },
    ]
    general = await _general_transform(messages)
    assert len(general[0].audios) == 1
    assert len(general[0].files) == 1
    assert general[0].files[0].resource.type == "application/pdf"

    result = transform_audio(general)
    assert len(result[0].audios) == 1
    assert len(result[0].files) == 1
    assert result[0].files[0].resource.type == "application/pdf"


async def test_input_audio_converted_to_audio_url():
    wav_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()
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


async def test_full_pipeline_multi_message():
    messages = [
        {
            "role": "user",
            "content": "listen",
            "custom_content": {
                "attachments": [
                    attachment(_audio_wav),
                    attachment(pic_1_1),
                ],
            },
        },
        {
            "role": "user",
            "content": "more",
            "custom_content": {
                "attachments": [
                    attachment(_audio_mp3),
                    attachment(_file_1),
                ],
            },
        },
    ]
    general = await _general_transform(messages)
    result = transform_audio(general)

    assert len(result) == 2

    assert result[0].raw_message["content"] == [
        text_part("listen"),
        image_part(pic_1_1),
        audio_part(_audio_wav),
    ]
    assert result[0].images == [image_resource(pic_1_1, 1, 1)]
    assert result[0].files == []

    assert result[1].raw_message["content"] == [
        text_part("more"),
        file_part(_file_1),
        audio_part(_audio_mp3),
    ]
    assert result[1].files == [file_resource(_file_1)]


async def test_no_audio_passthrough():
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "user",
            "content": "",
            "custom_content": {
                "attachments": [attachment(pic_1_1)],
            },
        },
    ]
    general = await _general_transform(messages)
    result = transform_audio(general)

    assert result[0] is general[0]
    assert result[1] is general[1]
