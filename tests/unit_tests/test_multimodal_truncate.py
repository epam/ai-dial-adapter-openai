import pytest
from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.gpt4_multi_modal.chat_completion import (
    multi_modal_truncate_prompt,
)
from aidial_adapter_openai.utils.image import ImageMetadata
from aidial_adapter_openai.utils.image_tokenizer import GPT4O_IMAGE_TOKENIZER
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource import Resource
from aidial_adapter_openai.utils.tokenizer import MultiModalTokenizer

tokenizer = MultiModalTokenizer("gpt-4o", GPT4O_IMAGE_TOKENIZER)


def test_multimodal_truncate_with_system_and_last_user_error():
    """
    Only system messages fit
    """
    transformations = [
        MultiModalMessage(
            image_metadatas=[],
            raw_message={"role": "system", "content": "this is four tokens"},
        ),
        MultiModalMessage(
            image_metadatas=[
                # Small image for 85 tokens
                ImageMetadata(
                    width=100,
                    height=100,
                    detail="low",
                    image=Resource(type="image/jpeg", data=b"..."),
                )
            ],
            raw_message={
                "role": "user",
                "content": [
                    {"type": "message", "message": "this is four tokens"},
                    {"type": "image_url", "image_url": "..."},
                ],
            },
        ),
    ]
    with pytest.raises(TruncatePromptSystemAndLastUserError):
        multi_modal_truncate_prompt({}, transformations, 15, tokenizer)


def test_multimodal_truncate_with_system_error():
    # 4 tokens for content + 3 tokens for message + 3 tokens for request = 10 tokens
    transformations = [
        MultiModalMessage(
            image_metadatas=[],
            raw_message={"role": "system", "content": "this is four tokens"},
        ),
    ]
    with pytest.raises(TruncatePromptSystemError):
        multi_modal_truncate_prompt({}, transformations, 9, tokenizer)


@pytest.mark.parametrize(
    "transformations,max_prompt_tokens,discarded_messages,used_tokens",
    [
        # Truncate one text message
        (
            # 8 * 3 (messages) + 3 (request) = 27 tokens
            [
                # 4 tokens of content + 3 tokens for message + 1 token of role key = 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "system",
                        "content": "this is four tokens",
                    },
                ),
                # 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "user",
                        "content": "this is four tokens",
                    },
                ),
                # 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "user",
                        "content": "this is four tokens",
                    },
                ),
            ],
            25,
            [1],
            19,
        ),
        # 19 tokens
        (
            [
                # 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "system",
                        "content": "this is four tokens",
                    },
                ),
                # 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "user",
                        "content": "this is four tokens",
                    },
                ),
            ],
            20,
            [],
            19,
        ),
        # 112 tokens
        (
            [
                # 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "system",
                        "content": "this is four tokens",
                    },
                ),
                # 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "user",
                        "content": "this if four tokens",
                    },
                ),
                # 85 (image) + 8 (textual) = 93 tokens
                MultiModalMessage(
                    image_metadatas=[
                        ImageMetadata(
                            width=100,
                            height=100,
                            detail="low",
                            image=Resource(type="image/jpeg", data=b"..."),
                        )
                    ],
                    raw_message={
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "this is four tokens",
                            },
                            {"type": "image_url", "image_url": "..."},
                        ],
                    },
                ),
            ],
            104,
            [1],
            104,
        ),
        # Case with empty content
        # 3 for request + 12 for messages = 15 tokens
        (
            [
                # 8 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={
                        "role": "system",
                        "content": "this is four tokens",
                    },
                ),
                # 4 tokens
                MultiModalMessage(
                    image_metadatas=[],
                    raw_message={"role": "user", "content": None},
                ),
            ],
            15,
            [],
            15,
        ),
    ],
)
def test_multimodal_truncate(
    transformations, max_prompt_tokens, discarded_messages, used_tokens
):
    truncated, actual_discarded_messages, actual_used_tokens = (
        multi_modal_truncate_prompt(
            {},
            transformations,
            max_prompt_tokens,
            tokenizer=tokenizer,
        )
    )
    assert actual_discarded_messages == discarded_messages
    assert actual_used_tokens == used_tokens
    assert truncated == [
        t for i, t in enumerate(transformations) if i not in discarded_messages
    ]
