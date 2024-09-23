from unittest.mock import AsyncMock, patch

import pytest

from aidial_adapter_openai.gpt4_multi_modal.attachment import ImageFail
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    TransformationError,
    transform_message,
    transform_messages,
)
from aidial_adapter_openai.utils.image import ImageDataURL, ImageMetadata
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage

TOKENS_FOR_TEXT = 10
TOKENS_FOR_IMAGE = 20


@pytest.fixture
def mock_file_storage():
    return AsyncMock()


@pytest.fixture
def mock_download_image():
    with patch(
        "aidial_adapter_openai.gpt4_multi_modal.transformation.download_image"
    ) as mock:
        yield mock


@pytest.fixture
def mock_image_metadata():
    with patch(
        "aidial_adapter_openai.gpt4_multi_modal.transformation.ImageMetadata.from_image_data_url"
    ) as mock:
        mock.return_value = ImageMetadata(
            width=100,
            height=100,
            detail="high",
            image=ImageDataURL(type="image/jpeg", data="..."),
        )
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
                "custom_content": {"attachments": ["image1.jpg"]},
            },
            TOKENS_FOR_TEXT + TOKENS_FOR_IMAGE,
            [
                {"type": "text", "text": ""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,...",
                        "detail": "high",
                    },
                },
            ],
        ),
        # Message with multiple images
        (
            {
                "role": "user",
                "content": "test with multiple images",
                "custom_content": {"attachments": ["image1.jpg", "image2.jpg"]},
            },
            TOKENS_FOR_TEXT + 2 * TOKENS_FOR_IMAGE,
            [
                {"type": "text", "text": "test with multiple images"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,...",
                        "detail": "high",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,...",
                        "detail": "high",
                    },
                },
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_transform_message(
    mock_file_storage,
    mock_download_image,
    mock_image_metadata,
    message,
    expected_tokens,
    expected_content,
):
    mock_download_image.return_value = ImageDataURL(
        type="image/jpeg", data="..."
    )

    result = await transform_message(mock_file_storage, message)

    assert isinstance(result, MultiModalMessage)
    assert result.raw_message["content"] == expected_content


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
    assert isinstance(result, TransformationError)
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
                    "custom_content": {"attachments": ["image1.jpg"]},
                },
            ],
            [
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={"role": "system", "content": "Hello"},
                ),
                MultiModalMessage(
                    image_metadatas=[
                        ImageMetadata(
                            width=100,
                            height=100,
                            detail="high",
                            image=ImageDataURL(type="image/jpeg", data="..."),
                        )
                    ],
                    raw_message={
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ""},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,...",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ),
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_transform_messages(
    mock_image_metadata,
    mock_file_storage,
    mock_download_image,
    messages,
    expected_transformations,
):
    mock_download_image.return_value = ImageDataURL(
        type="image/jpeg", data="..."
    )
    result = await transform_messages(mock_file_storage, messages)

    assert result == expected_transformations
