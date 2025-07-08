import time

import aiohttp


def _now() -> float:
    return time.monotonic()


def _elapsed_ms(start_time: float, last_time: float | None = None) -> int:
    last_time = last_time or _now()
    return round((last_time - start_time) * 1000)


def get_trace_config() -> aiohttp.TraceConfig:

    def _set_first(ctx):
        ctx.trace_request_ctx["first"] = ctx.trace_request_ctx["last"] = _now()

    def _set_elapsed(ctx, field: str):
        if last := ctx.trace_request_ctx.get("last"):
            ctx.trace_request_ctx[field] = _elapsed_ms(last)
            ctx.trace_request_ctx["last"] = _now()

    async def on_request_start(session, ctx, params):
        _set_first(ctx)

    async def on_dns_resolvehost_end(session, ctx, params):
        _set_elapsed(ctx, "dns")

    async def on_connection_create_end(session, ctx, params):
        _set_elapsed(ctx, "connect")

    async def on_request_end(session, ctx, params):
        _set_elapsed(ctx, "header")

    async def on_response_chunk_received(session, ctx, params):
        _set_elapsed(ctx, "body")

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_response_chunk_received.append(on_response_chunk_received)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.freeze()

    return trace_config


def get_tracing_timings(trace_request_ctx: dict) -> str:
    first = trace_request_ctx.get("first")
    last = trace_request_ctx.get("last")

    if not first or not last:
        return "na"

    dns = trace_request_ctx.get("dns") or "na"
    connect = trace_request_ctx.get("connect") or "na"
    header = trace_request_ctx.get("header") or "na"

    now = _now()
    body = trace_request_ctx.get("body") or _elapsed_ms(last, now)
    total = _elapsed_ms(first, now)

    return (
        f"{total} (dns={dns}, connect={connect}, header={header}, body={body})"
    )
