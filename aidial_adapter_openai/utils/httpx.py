import time
from typing import Any, Tuple

import httpx

from aidial_adapter_openai.utils.log_config import logger


def _now() -> float:
    return time.monotonic()


def _elapsed_ms(start_time: float, last_time: float | None = None) -> int:
    last_time = last_time or _now()
    return round((last_time - start_time) * 1000)


TraceCtx = dict[str, Any]
TraceCallback = Any  # `httpcore` doesn’t publish a public protocol for this


def _build_trace() -> Tuple[TraceCtx, TraceCallback]:
    trace_ctx: TraceCtx = {}
    starts: dict[str, float] = {}

    async def _trace(event_name: str, info: dict) -> None:
        now = _now()

        trace_ctx["first"] = trace_ctx.get("first") or now

        stem, _, suffix = event_name.rpartition(".")
        if suffix == "started":
            starts[stem] = now
        elif suffix == "complete":
            # Find the full list of events at https://www.encode.io/httpcore/extensions/
            bucket = {
                "connection.connect_tcp": "connect",
                "connection.start_tls": "connect",
                "http11.receive_response_headers": "header",
                "http2.receive_response_headers": "header",
                "http11.receive_response_body": "body",
                "http2.receive_response_body": "body",
            }.get(stem)

            if bucket and stem in starts:
                trace_ctx[bucket] = _elapsed_ms(starts[stem], now)

            trace_ctx["last"] = now

    return trace_ctx, _trace


def _get_tracing_timings(trace_request_ctx: dict) -> str:
    first = trace_request_ctx.get("first")
    last = trace_request_ctx.get("last")

    total = "na"
    if first and last:
        total = _elapsed_ms(first, last)

    connect = trace_request_ctx.get("connect") or "na"
    header = trace_request_ctx.get("header") or "na"
    body = trace_request_ctx.get("body") or "na"

    # httpx doesn’t expose a separate DNS event,
    # so we leave it as “na” for backward compatibility of the log message.
    return f"{total} (dns=na, connect={connect}, header={header}, body={body})"


async def _inject_trace_hook(request: httpx.Request) -> None:
    ctx, cb = _build_trace()
    request.extensions["trace"] = cb
    request.extensions["trace_ctx"] = ctx


async def _log_timings_hook(response: httpx.Response) -> None:
    request = response.request
    ctx: TraceCtx = request.extensions.get("trace_ctx", {})

    def _log_timing():
        logger.info(
            f"Upstream: '{request.url}'. Status: {response.status_code}. Timing: {_get_tracing_timings(ctx)}."
        )

    if response.is_closed:
        # If the body is already in memory we can log right away.
        _log_timing()
    else:
        # Otherwise delay logging until caller closes/finishes the stream.
        orig_aclose = response.aclose

        async def _aclose_and_log():
            await orig_aclose()
            _log_timing()

        response.aclose = _aclose_and_log


def get_tracing_event_hooks() -> dict:
    return {
        "request": [_inject_trace_hook],
        "response": [_log_timings_hook],
    }
