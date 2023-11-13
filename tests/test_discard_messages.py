import pytest

from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.tokens import discard_messages

gpt4_testdata = [
    (
        [],
        0,
        ([], 0),
    ),
    (
        [{"role": "system", "message": "This is four tokens"}],
        11,
        ([{"role": "system", "message": "This is four tokens"}], 0),
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
            1,
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
            1,
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
            2,
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
            1,
        ),
    ),
]


@pytest.mark.parametrize("messages, max_prompt_tokens, response", gpt4_testdata)
def test_discarded_messages(messages, max_prompt_tokens, response):
    try:
        assert (
            discard_messages(messages, "gpt-4", max_prompt_tokens) == response
        )
    except HTTPException as e:
        assert e.message == response
