import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.truncate_messages import truncate_messages
from aidial_adapter_openai.utils.truncation_types import (
    DiscardedMessages,
    TruncatedTokens,
)

PlainTextMessages = list[dict]
MaxPromptTokens = int
TestCase = tuple[
    PlainTextMessages,
    MaxPromptTokens,
    tuple[PlainTextMessages, DiscardedMessages],
]


async def plain_text_truncate_prompt(
    request: dict,
    messages: list[dict],
    max_prompt_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[list[dict], DiscardedMessages, TruncatedTokens]:
    (msgs, disc, tokens) = await truncate_messages(
        messages=[MultiModalMessage(raw_message=m) for m in messages],
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message.raw_message["role"]
        == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=await tokenizer.tokenize_request(request, []),
    )

    msgs = [m.raw_message for m in msgs]
    return (msgs, disc, tokens)


normal_cases: list[TestCase] = [
    (
        [],
        3,
        ([], []),
    ),
    (
        [{"role": "system", "content": "This is four tokens"}],
        11,
        ([{"role": "system", "content": "This is four tokens"}], []),
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
            {"role": "assistant", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        27,
        (
            [
                {"role": "system", "content": "This is four tokens"},
                {"role": "assistant", "content": "This is four tokens"},
                {"role": "user", "content": "This is four tokens"},
            ],
            [1],
        ),
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
            {"role": "assistant", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        34,
        (
            [
                {"role": "system", "content": "This is four tokens"},
                {"role": "assistant", "content": "This is four tokens"},
                {"role": "user", "content": "This is four tokens"},
            ],
            [1],
        ),
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
            {"role": "assistant", "content": "This is four tokens"},
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        27,
        (
            [
                {"role": "system", "content": "This is four tokens"},
                {"role": "system", "content": "This is four tokens"},
                {"role": "user", "content": "This is four tokens"},
            ],
            [1, 2],
        ),
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
            {"role": "assistant", "content": "This is four tokens"},
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        35,
        (
            [
                {"role": "system", "content": "This is four tokens"},
                {"role": "assistant", "content": "This is four tokens"},
                {"role": "system", "content": "This is four tokens"},
                {"role": "user", "content": "This is four tokens"},
            ],
            [1],
        ),
    ),
]

ErrorMessage = str
error_cases: list[
    tuple[
        PlainTextMessages,
        MaxPromptTokens,
        ErrorMessage,
    ]
] = [
    (
        [],
        2,
        "The requested maximum prompt tokens is 2. However, the system messages resulted in 3 tokens. Please reduce the length of the system messages or increase the maximum prompt tokens.",
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "system", "content": "This is four tokens"},
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        11,
        "The requested maximum prompt tokens is 11. However, the system messages resulted in 27 tokens. Please reduce the length of the system messages or increase the maximum prompt tokens.",
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        18,
        "The requested maximum prompt tokens is 18. However, the system messages and the last user message resulted in 19 tokens. Please reduce the length of the messages or increase the maximum prompt tokens.",
    ),
]


@pytest.mark.parametrize("messages, max_prompt_tokens, response", normal_cases)
async def test_discarded_messages_without_error(
    messages: list[dict],
    max_prompt_tokens: int,
    response: tuple[list[dict], DiscardedMessages],
):
    tokenizer = Tokenizer(model="gpt-4")
    (
        truncated_messages,
        discarded_messages,
        _used_tokens,
    ) = await plain_text_truncate_prompt(
        {}, messages, max_prompt_tokens, tokenizer
    )
    assert (truncated_messages, discarded_messages) == response


@pytest.mark.parametrize(
    "messages, max_prompt_tokens, error_message", error_cases
)
async def test_discarded_messages_with_error(
    messages: list[dict],
    max_prompt_tokens: int,
    error_message: str,
):
    tokenizer = Tokenizer(model="gpt-4")

    with pytest.raises(DialException) as e_info:
        await plain_text_truncate_prompt(
            {}, messages, max_prompt_tokens, tokenizer
        )
    assert e_info.value.message == error_message
