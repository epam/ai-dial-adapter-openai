import anthropic
import httpx

from aidial_adapter_openai.utils.cache import cache
from aidial_adapter_openai.utils.env import get_env_int
from aidial_adapter_openai.utils.httpx import get_tracing_event_hooks

HTTP_MAX_CONNECTIONS = get_env_int("HTTP_MAX_CONNECTIONS", 1000)
HTTP_MAX_KEEPALIVE_CONNECTIONS = get_env_int(
    "HTTP_MAX_KEEPALIVE_CONNECTIONS", 100
)
HTTP_POOL_TIMEOUT = get_env_int("HTTP_POOL_TIMEOUT", 10)


def _get_default_anthropic_timeout() -> httpx.Timeout:
    # Providing a timeout marginally different from the default Anthropic timeout
    # in order to disable the check that throws an error when
    # stream=False & max_tokens>=128K/6:
    # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176

    timeout = anthropic._constants.DEFAULT_TIMEOUT.as_dict()
    timeout["connect"] *= 1.0001  # type: ignore
    timeout["pool"] = HTTP_POOL_TIMEOUT
    return httpx.Timeout(**timeout)


# connect timeout and total timeout.
# The pool timeout makes requests fail fast when the connection pool
# is exhausted instead of queueing for the whole total timeout.
DEFAULT_TIMEOUT = httpx.Timeout(600, connect=10, pool=HTTP_POOL_TIMEOUT)

DEFAULT_ANTHROPIC_TIMEOUT = _get_default_anthropic_timeout()

# The defaults are borrowed from openai._constants.DEFAULT_CONNECTION_LIMITS
DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=HTTP_MAX_CONNECTIONS,
    max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
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


@cache(_close_httpx_client)
def get_anthropic_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=DEFAULT_ANTHROPIC_TIMEOUT,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True,
        event_hooks=get_tracing_event_hooks(),
    )
