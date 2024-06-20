import functools

import httpx

# connect timeout and total timeout
DEFAULT_TIMEOUT = httpx.Timeout(600, connect=10)

# Borrowed from openai._constants.DEFAULT_CONNECTION_LIMITS
DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000, max_keepalive_connections=100
)


@functools.cache
def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True,
    )
