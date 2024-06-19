import functools

import httpx

from aidial_adapter_openai.constant import (
    DEFAULT_CONNECTION_LIMITS,
    DEFAULT_TIMEOUT,
)


@functools.lru_cache(maxsize=1)
def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True,
    )
