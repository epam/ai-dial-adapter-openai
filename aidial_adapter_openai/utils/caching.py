import time
from typing import Any, Callable, Mapping

from aidial_sdk.chat_completion import CacheBreakpointPath

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


def _get_last_message_idx(request_body: Any) -> int | None:
    if not isinstance(request_body, dict):
        return None

    messages = request_body.get("messages") or []
    if not isinstance(messages, list):
        return None

    if not messages:
        return None

    return len(messages) - 1


def get_prompt_tokens_from_response(response_body: Any | None) -> int | None:
    if not isinstance(response_body, dict):
        return None

    usage = response_body.get("usage") or {}
    if not isinstance(usage, dict):
        return None

    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int):
        return None

    return prompt_tokens


def get_response_headers_for_caching(
    *,
    request_headers: Mapping[str, str],
    request_body: Any,
    get_request_tokens: Callable[[], int | None],
) -> dict[str, str] | None:
    # DIAL Core always sends this header if the deployment
    # is marked in listing as supporting auto-caching
    if request_headers.get(_DIAL_CACHE_BREAKPOINT_PATH) is None:
        return None

    if (last_message_idx := _get_last_message_idx(request_body)) is None:
        return None

    path = CacheBreakpointPath.messages(last_message_idx).path
    expire_at = str(int(time.time()) + _DEFAULT_TTL_SEC)

    prompt_tokens = get_request_tokens()
    if prompt_tokens is None or prompt_tokens < _PROMPT_TOKENS_THRESHOLD:
        return None

    return {
        _DIAL_CACHE_BREAKPOINT_PATH: path,
        _DIAL_CACHE_EXPIRE_AT: expire_at,
    }
