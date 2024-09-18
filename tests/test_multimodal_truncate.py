import pytest
from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.gpt4_multi_modal.chat_completion import (
    multimodal_truncate_prompt,
)
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    MultiModalMessage,
)


def test_multimodal_truncate_with_system_and_last_user_error():
    """
    Only system messages fit
    """
    transformations = [
        MultiModalMessage(
            tokens=10,
            message={"role": "system", "content": "System message"},
        ),
        MultiModalMessage(
            tokens=50,
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
        multimodal_truncate_prompt(transformations, 15, 0)


def test_multimodal_truncate_with_system_error():
    """
    It's just a system message, but with prompt tokens (+3) it's too long.
    """
    transformations = [
        MultiModalMessage(
            tokens=10,
            message={"role": "system", "content": "System message"},
        ),
    ]
    with pytest.raises(TruncatePromptSystemError):
        multimodal_truncate_prompt(transformations, 10, 1)


@pytest.mark.parametrize(
    "transformations,max_prompt_tokens,discarded_messages,used_tokens",
    [
        # Truncate one text message
        (
            [
                MultiModalMessage(
                    tokens=10,
                    message={"role": "system", "content": "System message"},
                ),
                MultiModalMessage(
                    tokens=10,
                    message={"role": "user", "content": "User message 1"},
                ),
                MultiModalMessage(
                    tokens=10,
                    message={"role": "user", "content": "User message 2"},
                ),
            ],
            30,
            [1],
            23,
        ),
        # No message to truncate
        (
            [
                MultiModalMessage(
                    tokens=10,
                    message={"role": "system", "content": "System message"},
                ),
                MultiModalMessage(
                    tokens=10,
                    message={"role": "user", "content": "User message"},
                ),
            ],
            30,
            [],
            23,
        ),
        # Truncate one image message
        (
            [
                MultiModalMessage(
                    tokens=10,
                    message={"role": "system", "content": "System message"},
                ),
                MultiModalMessage(
                    tokens=10,
                    message={"role": "user", "content": "User message"},
                ),
                MultiModalMessage(
                    tokens=50,
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
            [1],
            63,
        ),
    ],
)
def test_multimodal_truncate(
    transformations, max_prompt_tokens, discarded_messages, used_tokens
):
    truncated, actual_discarded_messages, actual_used_tokens = (
        multimodal_truncate_prompt(
            transformations, max_prompt_tokens, initial_prompt_tokens=3
        )
    )
    assert actual_discarded_messages == discarded_messages
    assert actual_used_tokens == used_tokens
    assert truncated == [
        t for i, t in enumerate(transformations) if i not in discarded_messages
    ]
