"""
Some models report cache-write tokens in
`usage.prompt_tokens_details.cache_creation_input_tokens` - a field borrowed
from the Anthropic Messages API, which isn't a part of the Chat Completions API.

The module normalizes it into `usage.prompt_tokens_details.cache_write_tokens`,
which is what DIAL expects.
"""

from collections.abc import AsyncIterator
from typing import TypeVar

from aidial_adapter_openai.utils.streaming import map_stream

_T = TypeVar("_T", bound=AsyncIterator[dict] | dict)


def _transform(chunk: dict) -> dict:
    details = (chunk.get("usage") or {}).get("prompt_tokens_details") or {}
    anthropic_cache_write = details.pop("cache_creation_input_tokens", None)

    if anthropic_cache_write is not None:
        details["cache_write_tokens"] = max(
            details.get("cache_write_tokens") or 0, anthropic_cache_write
        )

    return chunk


def normalize_cache_write_tokens(response: _T) -> _T:
    if isinstance(response, dict):
        return _transform(response)
    else:
        return map_stream(_transform, response)
