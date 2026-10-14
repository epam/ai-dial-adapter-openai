import anthropic
import httpx
import openai
import pytest

from aidial_adapter_openai.exceptions.handlers import to_adapter_exception
from aidial_adapter_openai.utils.http_client import (
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_POOL_TIMEOUT,
    get_anthropic_httpx_client,
    get_http_client,
)

_REQUEST = httpx.Request("POST", "https://example.com")


def _raise_from(exc: Exception, cause: Exception) -> Exception:
    try:
        raise exc from cause
    except Exception as e:
        return e


@pytest.mark.parametrize(
    "get_client", [get_http_client, get_anthropic_httpx_client]
)
def test_http_client_uses_configured_limits(get_client):
    client: httpx.AsyncClient = get_client()
    pool = client._transport._pool  # type: ignore[attr-defined]
    assert pool._max_connections == HTTP_MAX_CONNECTIONS
    assert pool._max_keepalive_connections == HTTP_MAX_KEEPALIVE_CONNECTIONS
    assert client.timeout.pool == HTTP_POOL_TIMEOUT


@pytest.mark.parametrize(
    "exc",
    [
        httpx.PoolTimeout("pool exhausted", request=_REQUEST),
        _raise_from(
            openai.APITimeoutError(request=_REQUEST),
            httpx.PoolTimeout("pool exhausted", request=_REQUEST),
        ),
        _raise_from(
            anthropic.APITimeoutError(request=_REQUEST),
            httpx.PoolTimeout("pool exhausted", request=_REQUEST),
        ),
    ],
    ids=["httpx", "openai", "anthropic"],
)
def test_pool_timeout_maps_to_503(exc: Exception):
    assert to_adapter_exception(exc).status_code == 503


def test_read_timeout_still_maps_to_504():
    exc = _raise_from(
        openai.APITimeoutError(request=_REQUEST),
        httpx.ReadTimeout("read timeout", request=_REQUEST),
    )
    assert to_adapter_exception(exc).status_code == 504
