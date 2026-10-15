import json
import logging
from collections.abc import AsyncIterator
from typing import Literal

import httpx
from openai import APIError

from aidial_adapter_openai.utils.adapter_exception import AdapterException
from aidial_adapter_openai.utils.log_config import logger

SSEStreamFormat = Literal["chat-completions", "responses"]


def _format_chunk(data: dict, event: str | None = None) -> str:
    event_part = "" if event is None else f"event: {event}\n"
    data_part = "data: " + json.dumps(data, separators=(",", ":"))
    return event_part + data_part + "\n\n"


async def to_sse_stream(
    stream: AsyncIterator[dict | AdapterException], format: SSEStreamFormat
) -> AsyncIterator[str]:
    async for chunk in stream:
        if isinstance(chunk, Exception):
            yield _format_chunk(chunk.json_error())
        else:
            event = None
            if format == "responses":
                event = chunk.get("type")
            yield _format_chunk(chunk, event)

    if format == "chat-completions":
        yield "data: [DONE]\n\n"


async def to_raw_sse_stream(
    stream: AsyncIterator[str | AdapterException],
) -> AsyncIterator[str]:
    async for chunk in stream:
        if isinstance(chunk, Exception):
            yield _format_chunk(chunk.json_error())
        else:
            yield chunk


def _raise_on_error_payload(response: httpx.Response, lines: list[str]) -> None:
    """
    Mirrors the openai SDK stream: a payload with a top-level "error"
    fails the stream. Only unnamed and "error" events can carry one,
    so the regular named events are never parsed.
    """
    data_lines: list[str] = []
    for line in lines:
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event" and value != "error":
            return
        if field == "data":
            data_lines.append(value)

    try:
        data = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return

    if isinstance(data, dict) and (error := data.get("error")):
        message = error.get("message") if isinstance(error, dict) else None
        if not message or not isinstance(message, str):
            message = "An error occurred during streaming"
        raise APIError(message=message, request=response.request, body=error)


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[str]:
    """
    The upstream SSE stream regrouped into whole events, which are passed
    through as received: validating and re-encoding every event through
    the SDK models costs most of the adapter CPU on a streamed response.
    """
    debug = logger.isEnabledFor(logging.DEBUG)
    lines: list[str] = []
    try:
        async for line in response.aiter_lines():
            if line:
                lines.append(line)
                continue
            if not lines:
                continue
            event, lines = lines, []
            if event[-1].startswith("data: [DONE]"):
                return
            _raise_on_error_payload(response, event)
            text = "\n".join(event) + "\n\n"
            if debug:
                logger.debug(f"upstream event: {text}")
            yield text
        if lines and not lines[-1].startswith("data: [DONE]"):
            _raise_on_error_payload(response, lines)
            yield "\n".join(lines) + "\n\n"
    finally:
        await response.aclose()
