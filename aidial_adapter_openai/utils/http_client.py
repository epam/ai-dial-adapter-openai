import httpx

from aidial_adapter_openai.utils.cache import cache
from aidial_adapter_openai.utils.httpx import get_tracing_event_hooks

# connect timeout and total timeout
DEFAULT_TIMEOUT = httpx.Timeout(600, connect=10)

# Borrowed from openai._constants.DEFAULT_CONNECTION_LIMITS
DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000, max_keepalive_connections=100
)


async def _close_httpx_client(client: httpx.AsyncClient) -> None:
    await client.aclose()


@cache(_close_httpx_client)
def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True,
        event_hooks=get_tracing_event_hooks(),
    )
