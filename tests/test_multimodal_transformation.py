from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aidial_adapter_openai.gpt4_multi_modal.download import ImageFail
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    MessageTransformResult,
    TransformMessagesResult,
    transform_message,
    transform_messages,
)
from aidial_adapter_openai.utils.image_data_url import ImageDataURL
from aidial_adapter_openai.utils.tokens import Tokenizer

TOKENS_FOR_TEXT = 10
TOKENS_FOR_IMAGE = 20


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.calculate_tokens_per_message.return_value = TOKENS_FOR_TEXT
    return tokenizer


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
def mock_tokenize_image():
    with patch(
        "aidial_adapter_openai.gpt4_multi_modal.transformation.tokenize_image"
    ) as mock:
        mock.return_value = (TOKENS_FOR_IMAGE, "high")
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
    mock_tokenizer,
    mock_download_image,
    mock_tokenize_image,
    message,
    expected_tokens,
    expected_content,
):
    mock_download_image.return_value = ImageDataURL(
        type="image/jpeg", data="..."
    )

    result = await transform_message(mock_file_storage, message, mock_tokenizer)

    assert result.total_tokens == expected_tokens
    assert result.message["content"] == expected_content


@pytest.mark.asyncio
async def test_transform_message_with_error(
    mock_file_storage, mock_tokenizer, mock_download_image
):
    message = {
        "role": "user",
        "content": "",
        "custom_content": {"attachments": ["error.jpg"]},
    }
    mock_download_image.return_value = ImageFail(
        name="error.jpg", message="File not found"
    )
    result = await transform_message(mock_file_storage, message, mock_tokenizer)

    assert result.errors is not None
    assert len(result.errors) == 1
    assert result.errors[0].name == "error.jpg"
    assert result.errors[0].message == "File not found"


@pytest.mark.parametrize(
    "messages,expected_result",
    [
        (
            [{"role": "user", "content": "Hello"}],
            TransformMessagesResult(
                transformations=[
                    MessageTransformResult(
                        text_tokens=10,
                        message={"role": "user", "content": "Hello"},
                    )
                ]
            ),
        ),
        (
            [
                {
                    "role": "user",
                    "content": "",
                    "custom_content": {"attachments": ["error.jpg"]},
                }
            ],
            TransformMessagesResult(
                error_message="The following file attachments failed to process:\n1. error.jpg: File not found"
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_transform_messages(
    mock_file_storage,
    mock_tokenizer,
    mock_download_image,
    messages,
    expected_result,
):
    if "error.jpg" in str(messages):
        mock_download_image.return_value = ImageFail(
            name="error.jpg", message="File not found"
        )
    else:
        mock_download_image.return_value = ImageDataURL(
            type="image/jpeg", data="..."
        )

    result = await transform_messages(
        mock_file_storage, messages, mock_tokenizer
    )

    if expected_result.error_message:
        assert result.error_message == expected_result.error_message
    else:
        assert result.transformations == expected_result.transformations
