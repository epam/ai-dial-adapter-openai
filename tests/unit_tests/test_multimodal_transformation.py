import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    ResourceProcessor,
    TransformationError,
)
from aidial_adapter_openai.utils.image import ImageMetadata
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource import Resource
from tests.utils.images import data_url, pic_1_1, pic_2_2, pic_3_3
from tests.utils.storage import MockFileStorage

TOKENS_FOR_TEXT = 10
TOKENS_FOR_IMAGE = 20


def attachment(resource: Resource) -> dict:
    return {"type": resource.type, "data": resource.data_base64}


def image_metadata(resource: Resource, w: int, h: int) -> ImageMetadata:
    return ImageMetadata(width=w, height=h, detail="low", image=resource)


def image_url(resource: Resource) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": data_url(resource), "detail": "low"},
    }


def text(text: str) -> dict:
    return {"type": "text", "text": text}


@pytest.fixture
def mock_resource_processor():
    return ResourceProcessor(file_storage=MockFileStorage())


@pytest.mark.parametrize(
    "message,expected_tokens,expected_content",
    [
        # Message without attachments
        ({"role": "user", "content": "Hello"}, TOKENS_FOR_TEXT, "Hello"),
        # Message with empty attachments
        (
            {
                "role": "user",
                "content": "Hi",
                "custom_content": {"attachments": []},
            },
            TOKENS_FOR_TEXT,
            "Hi",
        ),
        # Message with one image
        (
            {
                "role": "user",
                "content": "",
                "custom_content": {"attachments": [attachment(pic_1_1)]},
            },
            TOKENS_FOR_TEXT + TOKENS_FOR_IMAGE,
            [
                text(""),
                image_url(pic_1_1),
            ],
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
            TOKENS_FOR_TEXT + 2 * TOKENS_FOR_IMAGE,
            [
                text("test with multiple images"),
                image_url(pic_1_1),
                image_url(pic_2_2),
            ],
        ),
    ],
)
async def test_transform_message(
    mock_resource_processor,
    message,
    expected_tokens,
    expected_content,
):
    result = await mock_resource_processor.transform_message(message)

    assert isinstance(result, MultiModalMessage)
    assert result.raw_message.get("custom_content") is None
    assert result.raw_message["content"] == expected_content


async def test_transform_messages_with_error(mock_resource_processor):
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

    result = await mock_resource_processor.transform_messages(messages)

    assert isinstance(result, DialException)
    assert (
        result.message
        == """
The following files failed to process:
1. not_found1.jpg: file not found
2. not_found2.jpg: file not found
""".strip()
    )


async def test_transform_message_with_error(mock_resource_processor):
    message = {
        "role": "user",
        "content": "",
        "custom_content": {"attachments": [{"url": "not_found.jpg"}]},
    }
    await mock_resource_processor.transform_message(message)
    assert mock_resource_processor.errors
    assert len(mock_resource_processor.errors) == 1
    image_fail = list(mock_resource_processor.errors)[0]
    assert isinstance(image_fail, TransformationError)
    assert image_fail.name == "not_found.jpg"
    assert image_fail.message == "File not found"


@pytest.mark.parametrize(
    "messages,expected_transformations",
    [
        (
            [{"role": "user", "content": "Hello"}],
            [
                MultiModalMessage(
                    image_metadatas=[],
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
                    image_metadatas=[],
                    raw_message={"role": "system", "content": "Hello"},
                ),
                MultiModalMessage(
                    image_metadatas=[image_metadata(pic_1_1, 1, 1)],
                    raw_message={
                        "role": "user",
                        "content": [text(""), image_url(pic_1_1)],
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
                    image_metadatas=[],
                    raw_message={"role": "system", "content": "Hello"},
                ),
                MultiModalMessage(
                    image_metadatas=[],
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
                        text("User"),
                        image_url(pic_1_1),
                    ],
                },
            ],
            [
                MultiModalMessage(
                    image_metadatas=[image_metadata(pic_1_1, 1, 1)],
                    raw_message={
                        "role": "user",
                        "content": [
                            text("User"),
                            image_url(pic_1_1),
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
                        image_url(pic_1_1),
                        text("User"),
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
                    image_metadatas=[
                        image_metadata(pic_1_1, 1, 1),
                        image_metadata(pic_2_2, 2, 2),
                        image_metadata(pic_3_3, 3, 3),
                    ],
                    raw_message={
                        "role": "user",
                        "content": [
                            image_url(pic_1_1),
                            text("User"),
                            image_url(pic_2_2),
                            image_url(pic_3_3),
                        ],
                    },
                )
            ],
        ),
    ],
)
async def test_transform_messages(
    mock_resource_processor,
    messages,
    expected_transformations,
):
    result = await mock_resource_processor.transform_messages(messages)
    assert result == expected_transformations
