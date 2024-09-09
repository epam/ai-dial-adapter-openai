import pytest
from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    MessageTransformResult,
    TruncateTransformedMessagesResult,
)
from aidial_adapter_openai.gpt4_multi_modal.truncate import (
    truncate_transformed_messages,
)


def test_truncate_transformed_messages_system_and_last_user_error():
    """
    Only system messages fit
    """
    transformations = [
        MessageTransformResult(
            text_tokens=10,
            message={"role": "system", "content": "System message"},
        ),
        MessageTransformResult(
            image_tokens=40,
            text_tokens=10,
            message={
                "role": "user",
                "content": [
                    {"message": "text"},
                    {"type": "image_url", "image_url": "..."},
                ],
            },
        ),
    ]
    with pytest.raises(TruncatePromptSystemAndLastUserError):
        truncate_transformed_messages(transformations, 15)


def test_truncate_transformed_messages_system_error():
    """
    It's just a system message, but with prompt tokens (+3) it's too long.
    """
    transformations = [
        MessageTransformResult(
            text_tokens=10,
            message={"role": "system", "content": "System message"},
        ),
    ]
    with pytest.raises(TruncatePromptSystemError):
        truncate_transformed_messages(transformations, 10)


@pytest.mark.parametrize(
    "transformations,max_tokens,expected_result",
    [
        # Truncate one text message
        (
            [
                MessageTransformResult(
                    text_tokens=10,
                    message={"role": "system", "content": "System message"},
                ),
                MessageTransformResult(
                    text_tokens=10,
                    message={"role": "user", "content": "User message 1"},
                ),
                MessageTransformResult(
                    text_tokens=10,
                    message={"role": "user", "content": "User message 2"},
                ),
            ],
            30,
            TruncateTransformedMessagesResult(
                messages=[
                    {"role": "system", "content": "System message"},
                    {"role": "user", "content": "User message 2"},
                ],
                discarded_messages=[1],
                overall_token=23,
            ),
        ),
        # No message to truncate
        (
            [
                MessageTransformResult(
                    text_tokens=10,
                    message={"role": "system", "content": "System message"},
                ),
                MessageTransformResult(
                    text_tokens=10,
                    message={"role": "user", "content": "User message"},
                ),
            ],
            30,
            TruncateTransformedMessagesResult(
                messages=[
                    {"role": "system", "content": "System message"},
                    {"role": "user", "content": "User message"},
                ],
                discarded_messages=[],
                overall_token=23,
            ),
        ),
        # Truncate one image message
        (
            [
                MessageTransformResult(
                    text_tokens=10,
                    message={"role": "system", "content": "System message"},
                ),
                MessageTransformResult(
                    text_tokens=10,
                    message={"role": "user", "content": "User message"},
                ),
                MessageTransformResult(
                    text_tokens=10,
                    image_tokens=40,
                    message={
                        "role": "user",
                        "content": [
                            {"message": "image"},
                            {"type": "image_url", "image_url": "..."},
                        ],
                    },
                ),
            ],
            70,
            TruncateTransformedMessagesResult(
                messages=[
                    {"role": "system", "content": "System message"},
                    {
                        "role": "user",
                        "content": [
                            {"message": "image"},
                            {"type": "image_url", "image_url": "..."},
                        ],
                    },
                ],
                discarded_messages=[1],
                overall_token=63,
            ),
        ),
    ],
)
def test_truncate_transformed_messages(
    transformations, max_tokens, expected_result
):
    result = truncate_transformed_messages(transformations, max_tokens)
    assert result == expected_result
