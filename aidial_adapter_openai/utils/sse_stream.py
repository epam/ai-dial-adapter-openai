import json
from typing import Any, AsyncIterator, Mapping

from aidial_sdk.utils.errors import json_error

DATA_PREFIX = "data: "
OPENAI_END_MARKER = "[DONE]"


def format_chunk(data: str | Mapping[str, Any]) -> str:
    if isinstance(data, str):
        return DATA_PREFIX + data.strip() + "\n\n"
    else:
        return DATA_PREFIX + json.dumps(data, separators=(",", ":")) + "\n\n"


END_CHUNK = format_chunk(OPENAI_END_MARKER)


async def parse_openai_sse_stream(
    stream: AsyncIterator[bytes],
) -> AsyncIterator[dict]:
    async for line in stream:
        try:
            payload = line.decode("utf-8-sig").lstrip()  # type: ignore
        except Exception:
            yield json_error(
                message="Can't decode chunk to a string", type="runtime_error"
            )
            return

        if payload.strip() == "":
            continue

        if not payload.startswith(DATA_PREFIX):
            yield json_error(
                message="Invalid chunk format", type="runtime_error"
            )
            return

        payload = payload[len(DATA_PREFIX) :]

        if payload.strip() == OPENAI_END_MARKER:
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            yield json_error(
                message="Can't parse chunk to JSON", type="runtime_error"
            )
            return

        yield chunk


async def to_openai_sse_stream(
    stream: AsyncIterator[dict],
) -> AsyncIterator[str]:
    async for chunk in stream:
        yield format_chunk(chunk)
    yield END_CHUNK
