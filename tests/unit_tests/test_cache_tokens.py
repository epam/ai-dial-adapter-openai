from collections.abc import AsyncIterator

import pytest

from aidial_adapter_openai.chat_completions.cache_tokens import (
    normalize_cache_write_tokens,
)


def _response(prompt_tokens_details: dict | None) -> dict:
    usage: dict = {"prompt_tokens": 10042, "completion_tokens": 36}
    if prompt_tokens_details is not None:
        usage["prompt_tokens_details"] = prompt_tokens_details
    return {"id": "chatcmpl-test", "choices": [], "usage": usage}


@pytest.mark.parametrize(
    "given,expected",
    [
        # Anthropic-style cache-write tokens are normalized
        (
            {"cached_tokens": 0, "cache_creation_input_tokens": 10027},
            {
                "cached_tokens": 0,
                "cache_creation_input_tokens": 10027,
                "cache_write_tokens": 10027,
            },
        ),
        # The greater of the two is reported
        (
            {"cache_creation_input_tokens": 10027, "cache_write_tokens": 3},
            {"cache_creation_input_tokens": 10027, "cache_write_tokens": 10027},
        ),
        (
            {"cache_creation_input_tokens": 3, "cache_write_tokens": 10027},
            {"cache_creation_input_tokens": 3, "cache_write_tokens": 10027},
        ),
        # Untouched when the Anthropic field is missing
        ({"cached_tokens": 5}, {"cached_tokens": 5}),
        ({}, {}),
        (None, None),
    ],
)
def test_block_response(given: dict | None, expected: dict | None):
    response = normalize_cache_write_tokens(_response(given))
    assert response == _response(expected)


@pytest.mark.asyncio
async def test_stream():
    async def stream() -> AsyncIterator[dict]:
        yield {"id": "chatcmpl-test", "choices": [{"index": 0}]}
        yield _response({"cache_creation_input_tokens": 10027})

    chunks = [chunk async for chunk in normalize_cache_write_tokens(stream())]

    assert chunks == [
        {"id": "chatcmpl-test", "choices": [{"index": 0}]},
        _response(
            {
                "cache_creation_input_tokens": 10027,
                "cache_write_tokens": 10027,
            }
        ),
    ]
