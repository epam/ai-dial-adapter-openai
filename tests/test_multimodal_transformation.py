from unittest.mock import AsyncMock, patch

import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.gpt4_multi_modal.attachment import ImageFail
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    ImageProcessingFails,
    transform_message,
    transform_messages,
)
from aidial_adapter_openai.utils.data_url import DataURL
from aidial_adapter_openai.utils.image import ImageMetadata
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from tests.utils.images import data_url, pic_1_1, pic_2_2, pic_3_3

TOKENS_FOR_TEXT = 10
TOKENS_FOR_IMAGE = 20


def attachment(base64: str) -> dict:
    return {"type": "image/png", "data": base64}


def image_metadata(base64: str, w: int, h: int) -> ImageMetadata:
    image = DataURL.from_data_url(data_url(base64))
    assert image is not None
    return ImageMetadata(width=w, height=h, detail="low", image=image)


def image_url(base64: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": data_url(base64), "detail": "low"},
    }


def text(text: str) -> dict:
    return {"type": "text", "text": text}


@pytest.fixture
def mock_file_storage():
    return AsyncMock()


@pytest.fixture
def mock_download_image():
    with patch(
        "aidial_adapter_openai.gpt4_multi_modal.transformation.download_attachment_image"
    ) as mock:
        yield mock


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
@pytest.mark.asyncio
async def test_transform_message(
    mock_file_storage,
    message,
    expected_tokens,
    expected_content,
):
    result = await transform_message(mock_file_storage, message)

    assert isinstance(result, MultiModalMessage)
    assert result.raw_message.get("custom_content") is None
    assert result.raw_message["content"] == expected_content


@pytest.mark.asyncio
async def test_transform_messages_with_error(
    mock_file_storage, mock_download_image
):
    messages = [
        {
            "role": "user",
            "content": "",
            "custom_content": {"attachments": ["error1.jpg", "error2.jpg"]},
        }
    ]
    mock_download_image.side_effect = lambda _, attachment: ImageFail(
        name=attachment, message="File not found"
    )
    result = await transform_messages(mock_file_storage, messages)
    assert isinstance(result, DialException)
    assert (
        result.message
        == """
The following files failed to process:
1. error1.jpg: file not found
2. error2.jpg: file not found
""".strip()
    )


@pytest.mark.asyncio
async def test_transform_message_with_error(
    mock_file_storage, mock_download_image
):
    message = {
        "role": "user",
        "content": "",
        "custom_content": {"attachments": ["error.jpg"]},
    }
    mock_download_image.return_value = ImageFail(
        name="error.jpg", message="File not found"
    )
    result = await transform_message(mock_file_storage, message)
    assert isinstance(result, ImageProcessingFails)
    assert result.image_fails
    assert len(result.image_fails) == 1
    image_fail = result.image_fails[0]
    assert isinstance(image_fail, ImageFail)
    assert image_fail.name == "error.jpg"
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
@pytest.mark.asyncio
async def test_transform_messages(
    mock_file_storage,
    messages,
    expected_transformations,
):
    result = await transform_messages(mock_file_storage, messages)
    assert result == expected_transformations
