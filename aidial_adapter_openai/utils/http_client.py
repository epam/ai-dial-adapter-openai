import functools

import aiohttp
import httpx

from aidial_adapter_openai.utils.aiohttp.trace_config import get_trace_config

# connect timeout and total timeout
DEFAULT_HTTPX_TIMEOUT = httpx.Timeout(600, connect=10)
_DEFAULT_AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=600, connect=10)

# Borrowed from openai._constants.DEFAULT_CONNECTION_LIMITS
_DEFAULT_HTTPX_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000, max_keepalive_connections=100
)
_DEFAULT_AIOHTTP_CONNECTION_LIMITS = aiohttp.TCPConnector(
    limit=1000, limit_per_host=0
)


@functools.cache
def get_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=DEFAULT_HTTPX_TIMEOUT,
        limits=_DEFAULT_HTTPX_CONNECTION_LIMITS,
        follow_redirects=True,
    )


@functools.cache
def get_aiohttp_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        trust_env=True,
        trace_configs=[get_trace_config()],
        timeout=_DEFAULT_AIOHTTP_TIMEOUT,
        connector=_DEFAULT_AIOHTTP_CONNECTION_LIMITS,
    )
