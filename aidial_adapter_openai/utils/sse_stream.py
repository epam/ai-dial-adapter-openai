import json
from typing import AsyncIterator, Literal

from aidial_adapter_openai.utils.adapter_exception import AdapterException

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
