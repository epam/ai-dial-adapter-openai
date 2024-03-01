from typing import List, Tuple

import pytest

from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.tokens import Tokenizer, discard_messages

TestCase = Tuple[List[dict], int, Tuple[List[dict], List[int]] | str]

gpt4_testdata: List[TestCase] = [
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
        ],
        18,
        "The token size of system messages and the last user message (19) exceeds prompt token limit (18)",
    ),
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


@pytest.mark.parametrize("messages, max_prompt_tokens, response", gpt4_testdata)
def test_discarded_messages(
    messages: List[dict],
    max_prompt_tokens: int,
    response: Tuple[List[dict], List[int]] | str,
):
    try:
        tokenizer = Tokenizer(model="gpt-4")
        assert (
            discard_messages(tokenizer, messages, max_prompt_tokens) == response
        )
    except HTTPException as e:
        assert e.status_code == 400
        assert e.type == "invalid_request_error"
        assert e.message == response
