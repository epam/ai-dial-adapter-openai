import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.chat_completions.transformation import (
    Error,
    MessageTransformer,
    ResourceProcessor,
)
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.file import FileResource
from aidial_adapter_openai.utils.resource.image import ImageResource
from tests.utils.images import data_url, pic_1_1, pic_2_2, pic_3_3
from tests.utils.storage import DummyFileStorage

_file_1 = Resource(type="application/pdf", data=b"document content 1")
_file_2 = Resource(type="application/pdf", data=b"document content 2")

_audio_wav = Resource(type="audio/wav", data=b"RIFF\x00\x00\x00\x00WAVEfmt ")


async def transform_single_message(
    transformer: MessageTransformer, message: dict
) -> MultiModalMessage:
    results = [m async for m in transformer.transform_message(message)]
    assert len(results) == 1
    return results[0]


def attachment(resource: Resource) -> dict:
    return {"type": resource.type, "data": resource.data_base64}


def image_resource(resource: Resource, w: int, h: int) -> ImageResource:
    return ImageResource(width=w, height=h, detail="low", image=resource)


def file_resource(resource: Resource) -> FileResource:
    return FileResource(name="data attachment", resource=resource)


def image_part(resource: Resource) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": data_url(resource), "detail": "low"},
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


@pytest.fixture
def mock_resource_processor():
    return ResourceProcessor(file_storage=DummyFileStorage())


@pytest.fixture
def mock_message_transformer():
    return MessageTransformer(file_storage=DummyFileStorage())


@pytest.mark.parametrize(
    "message,expected_content",
    [
        # Message without attachments
        ({"role": "user", "content": "Hello"}, "Hello"),
        # Message with empty attachments
        (
            {
                "role": "user",
                "content": "Hi",
                "custom_content": {"attachments": []},
            },
            "Hi",
        ),
        # Message with one image
        (
            {
                "role": "user",
                "content": "",
                "custom_content": {"attachments": [attachment(pic_1_1)]},
            },
            [
                text_part(""),
                image_part(pic_1_1),
            ],
        ),
        # Message with one file
        (
            {
                "role": "user",
                "content": "",
                "custom_content": {"attachments": [attachment(_file_1)]},
            },
            [text_part(""), file_part(_file_1)],
        ),
        # Message with multiple images
        (
            {
                "role": "user",
                "content": "test with multiple images",
                "custom_content": {
                    "attachments": [
                        attachment(pic_1_1),
                        attachment(pic_2_2),
                    ]
                },
            },
            [
                text_part("test with multiple images"),
                image_part(pic_1_1),
                image_part(pic_2_2),
            ],
        ),
    ],
)
async def test_transform_to_content_parts(
    mock_message_transformer: MessageTransformer,
    message,
    expected_content,
):
    result = await transform_single_message(mock_message_transformer, message)

    assert isinstance(result, MultiModalMessage)
    assert result.raw_message.get("custom_content") is None
    assert result.raw_message["content"] == expected_content


async def test_transform_messages_not_found(
    mock_resource_processor: ResourceProcessor,
):
    messages = [
        {
            "role": "user",
            "content": "",
            "custom_content": {
                "attachments": [
                    {"url": "not_found1.jpg"},
                    {"url": "not_found2.jpg"},
                ]
            },
        }
    ]

    with pytest.raises(DialException) as exc:
        await mock_resource_processor.transform_messages(messages)

    assert (
        exc.value.message
        == """
The following files failed to process:
1. not_found1.jpg: file not found
2. not_found2.jpg: file not found
""".strip()
    )


async def test_transform_message_not_found(
    mock_message_transformer: MessageTransformer,
):
    message = {
        "role": "user",
        "content": "",
        "custom_content": {"attachments": [{"url": "not_found.jpg"}]},
    }
    await transform_single_message(mock_message_transformer, message)
    assert mock_message_transformer.errors is not None
    assert len(mock_message_transformer.errors) == 1
    image_fail = list(mock_message_transformer.errors)[0]
    assert isinstance(image_fail, Error)
    assert image_fail.name == "not_found.jpg"
    assert image_fail.message == "File not found"


@pytest.mark.parametrize(
    "messages,expected_transformations",
    [
        (
            [{"role": "user", "content": "Hello"}],
            [
                MultiModalMessage(
                    raw_message={"role": "user", "content": "Hello"},
                )
            ],
        ),
        (
            [
                {"role": "system", "content": "Hello"},
                {
                    "role": "user",
                    "content": "",
                    "custom_content": {"attachments": [attachment(pic_1_1)]},
                },
            ],
            [
                MultiModalMessage(
                    raw_message={"role": "system", "content": "Hello"},
                ),
                MultiModalMessage(
                    images=[image_resource(pic_1_1, 1, 1)],
                    raw_message={
                        "role": "user",
                        "content": [text_part(""), image_part(pic_1_1)],
                    },
                ),
            ],
        ),
        # No images, extra message field
        (
            [
                {"role": "system", "content": "Hello"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "User",
                            "extra_field": "extra_value",
                        }
                    ],
                },
            ],
            [
                MultiModalMessage(
                    raw_message={"role": "system", "content": "Hello"},
                ),
                MultiModalMessage(
                    raw_message={
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "User",
                                "extra_field": "extra_value",
                            }
                        ],
                    },
                ),
            ],
        ),
        # Content image
        (
            [
                {
                    "role": "user",
                    "content": [
                        text_part("User"),
                        image_part(pic_1_1),
                    ],
                },
            ],
            [
                MultiModalMessage(
                    images=[image_resource(pic_1_1, 1, 1)],
                    raw_message={
                        "role": "user",
                        "content": [
                            text_part("User"),
                            image_part(pic_1_1),
                        ],
                    },
                )
            ],
        ),
        # Content image + attachment images
        (
            [
                {
                    "role": "user",
                    "content": [
                        image_part(pic_1_1),
                        text_part("User"),
                    ],
                    "custom_content": {
                        "attachments": [
                            attachment(pic_2_2),
                            attachment(pic_3_3),
                        ]
                    },
                },
            ],
            [
                MultiModalMessage(
                    images=[
                        image_resource(pic_1_1, 1, 1),
                        image_resource(pic_2_2, 2, 2),
                        image_resource(pic_3_3, 3, 3),
                    ],
                    raw_message={
                        "role": "user",
                        "content": [
                            image_part(pic_1_1),
                            text_part("User"),
                            image_part(pic_2_2),
                            image_part(pic_3_3),
                        ],
                    },
                )
            ],
        ),
        # Image and files in multiple messages
        (
            [
                {
                    "role": "user",
                    "content": "hello",
                    "custom_content": {
                        "attachments": [
                            attachment(_file_1),
                            attachment(pic_1_1),
                        ]
                    },
                },
                {
                    "role": "user",
                    "content": "world",
                    "custom_content": {
                        "attachments": [
                            attachment(pic_2_2),
                            attachment(_file_2),
                            attachment(pic_3_3),
                        ]
                    },
                },
            ],
            [
                MultiModalMessage(
                    images=[
                        image_resource(pic_1_1, 1, 1),
                    ],
                    files=[file_resource(_file_1)],
                    raw_message={
                        "role": "user",
                        "content": [
                            text_part("hello"),
                            file_part(_file_1),
                            image_part(pic_1_1),
                        ],
                    },
                ),
                MultiModalMessage(
                    images=[
                        image_resource(pic_2_2, 2, 2),
                        image_resource(pic_3_3, 3, 3),
                    ],
                    files=[file_resource(_file_2)],
                    raw_message={
                        "role": "user",
                        "content": [
                            text_part("world"),
                            image_part(pic_2_2),
                            file_part(_file_2),
                            image_part(pic_3_3),
                        ],
                    },
                ),
            ],
        ),
    ],
)
async def test_transform_to_unified_messages(
    mock_resource_processor: ResourceProcessor,
    messages,
    expected_transformations,
):
    result = await mock_resource_processor.transform_messages(messages)
    assert result == expected_transformations


async def test_audio_in_content_part(
    mock_resource_processor: ResourceProcessor,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "AAAA", "format": "wav"},
                },
            ],
        },
    ]
    result = await mock_resource_processor.transform_messages(messages)
    assert len(result) == 1
    content = result[0].raw_message["content"]
    assert len(content) == 2
    assert content[1] == {
        "type": "input_audio",
        "input_audio": {"data": "AAAA", "format": "wav"},
    }
    assert len(result[0].audios) == 1
    assert result[0].audios[0].audio.type == "audio/wav"


async def test_audio_in_attachments(
    mock_message_transformer: MessageTransformer,
):
    message = {
        "role": "user",
        "content": "listen",
        "custom_content": {
            "attachments": [attachment(_audio_wav)],
        },
    }
    result = await transform_single_message(mock_message_transformer, message)

    assert isinstance(result, MultiModalMessage)
    assert len(result.audios) == 1
    assert result.audios[0].audio.type == "audio/wav"
    assert result.files == []
    assert len(result.raw_message["content"]) == 2
    assert result.raw_message["content"][0] == text_part("listen")
    assert result.raw_message["content"][1]["type"] == "input_audio"
