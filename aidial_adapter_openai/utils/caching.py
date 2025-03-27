import time
from typing import Any, Mapping

from aidial_sdk.chat_completion import CacheBreakpointPath

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"

# https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/prompt-caching
# > Caches are typically cleared within 5-10 minutes of inactivity and
# are always removed within one hour of the cache's last use.
_DEFAULT_TTL_SEC = 5 * 60  # 5 minutes


def _get_last_message_idx(request_body: Any) -> int | None:
    if not isinstance(request_body, dict):
        return None

    messages = request_body.get("messages") or []
    if not isinstance(messages, list):
        return None

    if not messages:
        return None

    return len(messages) - 1


def get_headers_for_caching(
    request_headers: Mapping[str, str], request_body: Any
) -> dict[str, str]:
    # DIAL Core always sends this header if the deployment is marked in listing
    # as supporting auto-caching
    if request_headers.get(_DIAL_CACHE_BREAKPOINT_PATH) is None:
        return {}

    if (last_message_idx := _get_last_message_idx(request_body)) is None:
        return {}

    path = CacheBreakpointPath.messages(last_message_idx).path
    expire_at = str(int(time.time()) + _DEFAULT_TTL_SEC)

    return {
        _DIAL_CACHE_BREAKPOINT_PATH: path,
        _DIAL_CACHE_EXPIRE_AT: expire_at,
    }
