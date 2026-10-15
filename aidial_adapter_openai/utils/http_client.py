import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import aiohttp
import anthropic
import httpx
from httpx_aiohttp import AiohttpTransport, HttpxAiohttpClient

from aidial_adapter_openai.utils.cache import cache
from aidial_adapter_openai.utils.env import get_env_int
from aidial_adapter_openai.utils.httpx import get_tracing_event_hooks
from aidial_adapter_openai.utils.log_config import logger

HTTP_MAX_CONNECTIONS = get_env_int("HTTP_MAX_CONNECTIONS", 1000)
HTTP_MAX_KEEPALIVE_CONNECTIONS = get_env_int(
    "HTTP_MAX_KEEPALIVE_CONNECTIONS", 100
)
HTTP_POOL_TIMEOUT = get_env_int("HTTP_POOL_TIMEOUT", 10)
HTTP_TRANSPORT = os.getenv("HTTP_TRANSPORT") or "aiohttp"
if HTTP_TRANSPORT not in {"aiohttp", "httpx"}:
    raise ValueError(
        f"HTTP_TRANSPORT must be 'aiohttp' or 'httpx', got {HTTP_TRANSPORT!r}"
    )


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


@dataclass
class _RequestState:
    # The httpcore-style trace callback of `utils.httpx`
    trace: Any
    waiting_for_connection: bool = False


_request_state: ContextVar[_RequestState] = ContextVar("_request_state")


def _on_signal(*events: str, waiting_for_connection: bool | None = None) -> Any:
    async def _handler(*_) -> None:
        state = _request_state.get(None)
        if state is None:
            return
        if waiting_for_connection is not None:
            state.waiting_for_connection = waiting_for_connection
        if state.trace is not None:
            for event in events:
                await state.trace(event, {})

    return _handler


def _trace_config() -> aiohttp.TraceConfig:
    """
    aiohttp's request signals, reported as the httpcore trace events that
    the upstream timing log of `utils.httpx` reads.
    """
    trace = aiohttp.TraceConfig()
    trace.on_connection_queued_start.append(
        _on_signal(waiting_for_connection=True)
    )
    trace.on_connection_queued_end.append(
        _on_signal(waiting_for_connection=False)
    )
    trace.on_connection_create_start.append(
        _on_signal("connection.connect_tcp.started")
    )
    trace.on_connection_create_end.append(
        _on_signal("connection.connect_tcp.complete")
    )
    trace.on_request_headers_sent.append(
        _on_signal("http11.receive_response_headers.started")
    )
    # Fires once the response headers are in
    trace.on_request_end.append(
        _on_signal(
            "http11.receive_response_headers.complete",
            "http11.receive_response_body.started",
        )
    )
    return trace


class _AiohttpTransport(AiohttpTransport):
    """
    httpcore's pool scans every pooled connection on each request event,
    which costs a lot of CPU with hundreds of concurrent streams; aiohttp's
    pool doesn't.

    aiohttp raises the same timeout for a request that waited too long for
    a pooled connection as for a slow connect. The first is reported as
    `httpx.PoolTimeout`, as httpcore does, so it maps to 503.
    """

    def get_client(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(
            limit=self.limits.max_connections or 0,
            keepalive_timeout=self.limits.keepalive_expiry,
            ssl=self.ssl_context,
        )
        return aiohttp.ClientSession(
            connector=connector, trace_configs=[_trace_config()]
        )

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        state = _RequestState(trace=request.extensions.get("trace"))
        _request_state.set(state)
        try:
            response = await super().handle_async_request(request)
        except httpx.ConnectTimeout as e:
            if state.waiting_for_connection:
                raise httpx.PoolTimeout(str(e), request=request) from e
            raise

        if state.trace is not None:
            stream = response.stream
            close_stream = stream.aclose  # type: ignore

            async def _aclose() -> None:
                await close_stream()
                await state.trace("http11.receive_response_body.complete", {})

            stream.aclose = _aclose  # type: ignore

        return response


async def _close_httpx_client(client: httpx.AsyncClient) -> None:
    await client.aclose()


def _create_httpx_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    logger.info(f"Upstream HTTP transport: {HTTP_TRANSPORT}")
    if HTTP_TRANSPORT == "aiohttp":
        return HttpxAiohttpClient(
            timeout=timeout,
            follow_redirects=True,
            event_hooks=get_tracing_event_hooks(),
            transport=_AiohttpTransport(limits=DEFAULT_CONNECTION_LIMITS),
        )
    return httpx.AsyncClient(
        timeout=timeout,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True,
        event_hooks=get_tracing_event_hooks(),
    )


@cache(_close_httpx_client)
def get_http_client() -> httpx.AsyncClient:
    return _create_httpx_client(DEFAULT_TIMEOUT)


@cache(_close_httpx_client)
def get_anthropic_httpx_client() -> httpx.AsyncClient:
    return _create_httpx_client(DEFAULT_ANTHROPIC_TIMEOUT)
