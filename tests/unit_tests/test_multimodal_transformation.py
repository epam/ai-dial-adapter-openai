import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.chat_completions.transformation import (
    Error,
    ResourceProcessor,
)
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.image import ImageResource
from tests.utils.images import data_url, pic_1_1, pic_2_2, pic_3_3
from tests.utils.storage import DummyFileStorage


def attachment(resource: Resource) -> dict:
    return {"type": resource.type, "data": resource.data_base64}


def image_metadata(resource: Resource, w: int, h: int) -> ImageResource:
    return ImageResource(width=w, height=h, detail="low", image=resource)


def image_url(resource: Resource) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": data_url(resource), "detail": "low"},
    }


def text(text: str) -> dict:
    return {"type": "text", "text": text}


@pytest.fixture
def mock_resource_processor():
    return ResourceProcessor(file_storage=DummyFileStorage())


# @pytest.fixture
# def mock_image_tokenizer():
#     def image_tokenizer(*args):
#         class _Tokenizer:
#             def tokenize(self, *args):
#                 return TOKENS_FOR_IMAGE

#         return _Tokenizer()

#     with patch(
#         "aidial_adapter_openai.endpoints.chat_completion.get_image_tokenizer",
#         return_value=image_tokenizer,
#     ):
#         yield


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
            [
                text("test with multiple images"),
                image_url(pic_1_1),
                image_url(pic_2_2),
            ],
        ),
    ],
)
async def test_transform_to_content_parts(
    mock_resource_processor: ResourceProcessor,
    message,
    expected_content,
):
    result = await mock_resource_processor.transform_message(message)

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
    mock_resource_processor: ResourceProcessor,
):
    message = {
        "role": "user",
        "content": "",
        "custom_content": {"attachments": [{"url": "not_found.jpg"}]},
    }
    await mock_resource_processor.transform_message(message)
    assert mock_resource_processor.errors
    assert len(mock_resource_processor.errors) == 1
    image_fail = list(mock_resource_processor.errors)[0]
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
                    images=[image_metadata(pic_1_1, 1, 1)],
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
                        text("User"),
                        image_url(pic_1_1),
                    ],
                },
            ],
            [
                MultiModalMessage(
                    images=[image_metadata(pic_1_1, 1, 1)],
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
                    images=[
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
        # Image in multiple messages
        (
            [
                {
                    "role": "user",
                    "content": "hello",
                    "custom_content": {
                        "attachments": [
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
                            attachment(pic_3_3),
                        ]
                    },
                },
            ],
            [
                MultiModalMessage(
                    images=[
                        image_metadata(pic_1_1, 1, 1),
                    ],
                    raw_message={
                        "role": "user",
                        "content": [
                            text("hello"),
                            image_url(pic_1_1),
                        ],
                    },
                ),
                MultiModalMessage(
                    images=[
                        image_metadata(pic_2_2, 2, 2),
                        image_metadata(pic_3_3, 3, 3),
                    ],
                    raw_message={
                        "role": "user",
                        "content": [
                            text("world"),
                            image_url(pic_2_2),
                            image_url(pic_3_3),
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
