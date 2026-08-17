import time
from collections.abc import Callable, Coroutine, Mapping

from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
)
from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase,
)

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"

# https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/prompt-caching
# > Caches are typically cleared within 5-10 minutes of inactivity and
# are always removed within one hour of the cache's last use.
_DEFAULT_TTL_SEC = 5 * 60  # 5 minutes

# Ibid.
# > For a request to take advantage of prompt caching the request must be both:
# > * A minimum of 1,024 tokens in length.
# > * The first 1,024 tokens in the prompt must be identical.
_PROMPT_TOKENS_THRESHOLD = 1024


def get_chat_completions_breakpoint_path(
    request: CompletionCreateParamsBase,
) -> str | None:
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    return f"prefix.body.messages[{len(messages) - 1}]"


def get_responses_breakpoint_path(
    request: ResponseCreateParamsBase,
) -> str | None:
    input = request.get("input")
    if isinstance(input, list):
        if input:
            return f"prefix.body.input[{len(input) - 1}]"
    elif input is not None:
        # A scalar where an array is expected counts as a one-element array,
        # the way DIAL Core hashes it: `"input": "hi"` is `input[0]`.
        return "prefix.body.input[0]"

    # No input at all: the instructions are the only prefix left to cache.
    if request.get("instructions"):
        return "prefix.body.instructions[0]"

    return None


async def build_cache_headers(
    *,
    request_headers: Mapping[str, str],
    breakpoint_path: str | None,
    get_request_tokens: (
        Callable[[], Coroutine[None, None, int]] | None
    ) = None,
) -> dict[str, str] | None:
    # DIAL Core always sends this header if the deployment
    # is marked in listing as supporting auto-caching
    if request_headers.get(_DIAL_CACHE_BREAKPOINT_PATH) is None:
        return None

    if breakpoint_path is None:
        return None

    if (
        get_request_tokens is not None
        and await get_request_tokens() < _PROMPT_TOKENS_THRESHOLD
    ):
        return None

    return {
        _DIAL_CACHE_BREAKPOINT_PATH: breakpoint_path,
        _DIAL_CACHE_EXPIRE_AT: str(int(time.time()) + _DEFAULT_TTL_SEC),
    }
