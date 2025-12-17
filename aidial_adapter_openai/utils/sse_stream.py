import json
from typing import Any, AsyncIterator, Mapping

from aidial_adapter_openai.utils.adapter_exception import AdapterException

_DATA_PREFIX = "data: "
_OPENAI_END_MARKER = "[DONE]"


def _format_chunk(data: str | Mapping[str, Any]) -> str:
    if isinstance(data, str):
        return _DATA_PREFIX + data.strip() + "\n\n"
    else:
        return _DATA_PREFIX + json.dumps(data, separators=(",", ":")) + "\n\n"


_END_CHUNK = _format_chunk(_OPENAI_END_MARKER)


async def to_openai_sse_stream(
    stream: AsyncIterator[dict | AdapterException],
) -> AsyncIterator[str]:
    async for chunk in stream:
        if isinstance(chunk, Exception):
            yield _format_chunk(chunk.json_error())
        else:
            yield _format_chunk(chunk)
    yield _END_CHUNK
