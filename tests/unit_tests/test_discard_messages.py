from typing import List, Tuple

import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.gpt import plain_text_truncate_prompt
from aidial_adapter_openai.utils.tokenizer import PlainTextTokenizer
from aidial_adapter_openai.utils.truncate_prompt import DiscardedMessages

PlainTextMessages = List[dict]
MaxPromptTokens = int
TestCase = Tuple[
    PlainTextMessages,
    MaxPromptTokens,
    Tuple[PlainTextMessages, DiscardedMessages],
]

normal_cases: List[TestCase] = [
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
error_cases: List[
    Tuple[
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
def test_discarded_messages_without_error(
    messages: List[dict],
    max_prompt_tokens: int,
    response: Tuple[List[dict], List[int]],
):
    tokenizer = PlainTextTokenizer(model="gpt-4")
    truncated_messages, discarded_messages, _used_tokens = (
        plain_text_truncate_prompt({}, messages, max_prompt_tokens, tokenizer)
    )
    assert (truncated_messages, discarded_messages) == response


@pytest.mark.parametrize(
    "messages, max_prompt_tokens, error_message", error_cases
)
def test_discarded_messages_with_error(
    messages: List[dict],
    max_prompt_tokens: int,
    error_message: str,
):
    tokenizer = PlainTextTokenizer(model="gpt-4")

    with pytest.raises(DialException) as e_info:
        plain_text_truncate_prompt({}, messages, max_prompt_tokens, tokenizer)
    assert e_info.value.message == error_message
