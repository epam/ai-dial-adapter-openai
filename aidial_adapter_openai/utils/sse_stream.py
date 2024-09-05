import json
from typing import Any, AsyncIterator, Mapping

from aidial_sdk.exceptions import runtime_server_error

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
            yield runtime_server_error(
                "Can't decode chunk to a string"
            ).json_error()
            return

        if payload.strip() == "":
            continue

        if not payload.startswith(DATA_PREFIX):
            yield runtime_server_error("Invalid chunk format").json_error()
            return

        payload = payload[len(DATA_PREFIX) :]

        if payload.strip() == OPENAI_END_MARKER:
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            yield runtime_server_error("Can't parse chunk to JSON").json_error()
            return

        yield chunk


async def to_openai_sse_stream(
    stream: AsyncIterator[dict],
) -> AsyncIterator[str]:
    async for chunk in stream:
        yield format_chunk(chunk)
    yield END_CHUNK
