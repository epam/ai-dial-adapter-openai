import contextlib
import time
from typing import Any, Dict

import aiohttp

from aidial_adapter_openai.utils.log_config import logger


def _now() -> float:
    return time.monotonic()


def _elapsed_ms(start_time: float) -> int:
    return round((_now() - start_time) * 1000)


def _get_trace_config() -> aiohttp.TraceConfig:

    async def on_request_start(session, trace_config_ctx, params):
        trace_config_ctx.start_time = trace_config_ctx.trace_request_ctx[
            "start_time"
        ] = _now()

    async def on_dns_resolvehost_end(session, trace_config_ctx, params):
        if getattr(trace_config_ctx, "start_time", None):
            elapsed = _elapsed_ms(trace_config_ctx.start_time)
            trace_config_ctx.trace_request_ctx["dns"] = elapsed

    async def on_connection_create_end(session, trace_config_ctx, params):
        if getattr(trace_config_ctx, "start_time", None):
            elapsed = _elapsed_ms(trace_config_ctx.start_time)
            trace_config_ctx.trace_request_ctx["connect"] = elapsed

    async def on_response_chunk_received(session, trace_config_ctx, params):
        if getattr(trace_config_ctx, "start_time", None):
            elapsed = _elapsed_ms(trace_config_ctx.start_time)
            trace_config_ctx.trace_request_ctx["body"] = elapsed

    async def on_request_end(session, trace_config_ctx, params):
        if getattr(trace_config_ctx, "start_time", None):
            elapsed = _elapsed_ms(trace_config_ctx.start_time)
            trace_config_ctx.trace_request_ctx["header"] = elapsed

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_response_chunk_received.append(on_response_chunk_received)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.freeze()

    return trace_config


_trace_config = _get_trace_config()


def _get_tracing_timings(trace_request_ctx: dict) -> str:
    start_time = trace_request_ctx.get("start_time")
    dns = trace_request_ctx.get("dns") or "na"
    connect = trace_request_ctx.get("connect") or "na"
    header = trace_request_ctx.get("header") or "na"
    body = (
        trace_request_ctx.get("body")
        or (None if start_time is None else _elapsed_ms(start_time))
        or "na"
    )

    return f"Cumulative timings: dns={dns}, connect={connect}, header={header}, body={body}"


@contextlib.asynccontextmanager
async def post(url: str, headers: Dict[str, str], request: Any):
    ctx = {}
    async with aiohttp.ClientSession(trace_configs=[_trace_config]) as session:
        async with session.post(
            url, json=request, headers=headers, trace_request_ctx=ctx
        ) as response:
            try:
                yield response
            finally:
                logger.info(
                    f"POST {url!r} {response.status} | {_get_tracing_timings(ctx)}"
                )
