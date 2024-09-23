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
        "The token size of system messages (0) exceeds prompt token limit (2)",
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "system", "content": "This is four tokens"},
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        11,
        "The token size of system messages (27) exceeds prompt token limit (11)",
    ),
    (
        [
            {"role": "system", "content": "This is four tokens"},
            {"role": "user", "content": "This is four tokens"},
        ],
        18,
        "The token size of system messages and the last user message (19) exceeds prompt token limit (18)",
    ),
    ([], 0, ""),
]


@pytest.mark.parametrize("messages, max_prompt_tokens, response", normal_cases)
def test_discarded_messages_without_error(
    messages: List[dict],
    max_prompt_tokens: int,
    response: Tuple[List[dict], List[int]],
):
    tokenizer = PlainTextTokenizer(model="gpt-4")
    truncated_messages, discarded_messages, _used_tokens = (
        plain_text_truncate_prompt(messages, max_prompt_tokens, tokenizer)
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
        plain_text_truncate_prompt(messages, max_prompt_tokens, tokenizer)
        assert e_info.value.message == error_message
