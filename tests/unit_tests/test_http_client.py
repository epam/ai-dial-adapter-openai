import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncGenerator

import anthropic
import httpx
import openai
import pytest
from aiohttp import web
from httpx_aiohttp import AiohttpTransport

from aidial_adapter_openai.exceptions.handlers import to_adapter_exception
from aidial_adapter_openai.utils.http_client import (
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_POOL_TIMEOUT,
    _AiohttpTransport,
    get_anthropic_httpx_client,
    get_http_client,
)
from aidial_adapter_openai.utils.httpx import get_tracing_event_hooks

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


@contextlib.asynccontextmanager
async def _local_server(port: int, handler) -> AsyncGenerator[str]:
    app = web.Application()
    app.router.add_route("*", "/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        await runner.cleanup()


async def test_aiohttp_pool_exhaustion_is_pool_timeout(unused_tcp_port: int):
    release = asyncio.Event()

    async def _slow(_: web.Request) -> web.Response:
        await release.wait()
        return web.Response(text="ok")

    async with _local_server(unused_tcp_port, _slow) as url:
        client = httpx.AsyncClient(
            transport=_AiohttpTransport(limits=httpx.Limits(max_connections=1)),
            timeout=httpx.Timeout(5.0, pool=0.2),
        )
        holder = asyncio.create_task(client.get(url))
        try:
            await asyncio.sleep(0.1)  # the only connection is now in use

            with pytest.raises(httpx.PoolTimeout) as exc_info:
                await client.get(url)
            assert to_adapter_exception(exc_info.value).status_code == 503
        finally:
            release.set()
            assert (await holder).status_code == 200
            await client.aclose()


async def test_aiohttp_connect_timeout_without_queueing_stays_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _connect_timeout(self, request: httpx.Request):
        raise httpx.ConnectTimeout("slow connect", request=request)

    monkeypatch.setattr(
        AiohttpTransport, "handle_async_request", _connect_timeout
    )
    with pytest.raises(httpx.ConnectTimeout) as exc_info:
        await _AiohttpTransport().handle_async_request(
            httpx.Request("GET", "http://example.com")
        )
    assert not isinstance(exc_info.value, httpx.PoolTimeout)


async def test_aiohttp_upstream_timing(
    unused_tcp_port: int, caplog: pytest.LogCaptureFixture
):
    async def _stream(request: web.Request) -> web.StreamResponse:
        await asyncio.sleep(0.05)
        response = web.StreamResponse()
        await response.prepare(request)
        for _ in range(3):
            await asyncio.sleep(0.05)
            await response.write(b"data: chunk\n\n")
        return response

    events: list[str] = []
    hooks = get_tracing_event_hooks()

    async def _record_events(request: httpx.Request) -> None:
        trace = request.extensions["trace"]

        async def _recording_trace(event: str, info: dict) -> None:
            events.append(event)
            await trace(event, info)

        request.extensions["trace"] = _recording_trace

    async with _local_server(unused_tcp_port, _stream) as url:
        client = httpx.AsyncClient(
            transport=_AiohttpTransport(),
            event_hooks={
                **hooks,
                "request": [*hooks["request"], _record_events],
            },
        )
        try:
            with caplog.at_level(logging.INFO):
                async with client.stream("GET", url) as response:
                    chunks = [c async for c in response.aiter_bytes()]
        finally:
            await client.aclose()

    assert b"".join(chunks) == b"data: chunk\n\n" * 3
    assert events == [
        "connection.connect_tcp.started",
        "connection.connect_tcp.complete",
        "http11.receive_response_headers.started",
        "http11.receive_response_headers.complete",
        "http11.receive_response_body.started",
        "http11.receive_response_body.complete",
    ]
    [timing] = [r.message for r in caplog.records if "Timing:" in r.message]
    match = re.search(
        r"Timing: (\d+) \(dns=na, connect=\w+, header=(\d+), body=(\d+)\)",
        timing,
    )
    assert match, timing
    total, header, body = map(int, match.groups())
    assert header >= 40 and body >= 100 and total >= header + body
