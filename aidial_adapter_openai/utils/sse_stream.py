import json
import re
from collections.abc import AsyncIterator

from aidial_adapter_openai.utils.adapter_exception import AdapterException

# An event ends at a blank line.
# https://html.spec.whatwg.org/multipage/server-sent-events.html
_EVENT_END = re.compile(rb"\r\n\r\n|\n\n")


def _format_chunk(data: dict) -> str:
    return "data: " + json.dumps(data, separators=(",", ":")) + "\n\n"


async def to_sse_stream(
    stream: AsyncIterator[dict | AdapterException],
) -> AsyncIterator[str]:
    async for chunk in stream:
        yield _format_chunk(
            chunk.json_error() if isinstance(chunk, Exception) else chunk
        )

    yield "data: [DONE]\n\n"


async def chunk_sse_events(
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """
    Re-chunk a raw SSE stream so that every yielded item is a whole event.

    The bytes are untouched - only the chunk boundaries move - which is what
    gives the heartbeat a safe place to splice itself into: a transport-level
    chunk may end in the middle of an event, a yielded item never does.
    """
    buffer = b""

    async for chunk in stream:
        buffer += chunk
        while match := _EVENT_END.search(buffer):
            event, buffer = buffer[: match.end()], buffer[match.end() :]
            yield event

    if buffer:
        yield buffer
