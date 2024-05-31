from typing import List, Tuple

import pytest

from aidial_adapter_openai.errors import UserError
from aidial_adapter_openai.utils.tokens import Tokenizer, discard_messages

TestCase = Tuple[List[dict], int, Tuple[List[dict], List[int]] | str]

normal_cases: List[TestCase] = [
    (
        [],
        0,
        ([], []),
    ),
    (
        [{"role": "system", "message": "This is four tokens"}],
        11,
        ([{"role": "system", "message": "This is four tokens"}], []),
    ),
    (
        [
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
            {"role": "assistant", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
        ],
        27,
        (
            [
                {"role": "system", "message": "This is four tokens"},
                {"role": "assistant", "message": "This is four tokens"},
                {"role": "user", "message": "This is four tokens"},
            ],
            [1],
        ),
    ),
    (
        [
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
            {"role": "assistant", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
        ],
        34,
        (
            [
                {"role": "system", "message": "This is four tokens"},
                {"role": "assistant", "message": "This is four tokens"},
                {"role": "user", "message": "This is four tokens"},
            ],
            [1],
        ),
    ),
    (
        [
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
            {"role": "assistant", "message": "This is four tokens"},
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
        ],
        27,
        (
            [
                {"role": "system", "message": "This is four tokens"},
                {"role": "system", "message": "This is four tokens"},
                {"role": "user", "message": "This is four tokens"},
            ],
            [1, 2],
        ),
    ),
    (
        [
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
            {"role": "assistant", "message": "This is four tokens"},
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
        ],
        35,
        (
            [
                {"role": "system", "message": "This is four tokens"},
                {"role": "assistant", "message": "This is four tokens"},
                {"role": "system", "message": "This is four tokens"},
                {"role": "user", "message": "This is four tokens"},
            ],
            [1],
        ),
    ),
]

error_cases: List[TestCase] = [
    (
        [
            {"role": "system", "message": "This is four tokens"},
            {"role": "system", "message": "This is four tokens"},
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
        ],
        11,
        "The token size of system messages (27) exceeds prompt token limit (11)",
    ),
    (
        [
            {"role": "system", "message": "This is four tokens"},
            {"role": "user", "message": "This is four tokens"},
        ],
        18,
        "The token size of system messages and the last user message (19) exceeds prompt token limit (18)",
    ),
]


@pytest.mark.parametrize("messages, max_prompt_tokens, response", normal_cases)
def test_discarded_messages_without_error(
    messages: List[dict],
    max_prompt_tokens: int,
    response: Tuple[List[dict], List[int]],
):
    tokenizer = Tokenizer(model="gpt-4")
    assert discard_messages(tokenizer, messages, max_prompt_tokens) == response


@pytest.mark.parametrize(
    "messages, max_prompt_tokens, error_message", error_cases
)
def test_discarded_messages_with_error(
    messages: List[dict],
    max_prompt_tokens: int,
    error_message: str,
):
    tokenizer = Tokenizer(model="gpt-4")
    with pytest.raises(UserError) as e_info:
        discard_messages(tokenizer, messages, max_prompt_tokens)
        assert e_info.value.error_message == error_message
