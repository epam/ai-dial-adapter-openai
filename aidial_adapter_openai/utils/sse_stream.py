import json
from typing import Any, AsyncIterator, Mapping

from aidial_adapter_openai.exception_handlers import to_adapter_exception
from aidial_adapter_openai.utils.log_config import logger

_DATA_PREFIX = "data: "
_OPENAI_END_MARKER = "[DONE]"


def _format_chunk(data: str | Mapping[str, Any]) -> str:
    if isinstance(data, str):
        return _DATA_PREFIX + data.strip() + "\n\n"
    else:
        return _DATA_PREFIX + json.dumps(data, separators=(",", ":")) + "\n\n"


_END_CHUNK = _format_chunk(_OPENAI_END_MARKER)


async def to_openai_sse_stream(
    stream: AsyncIterator[dict],
) -> AsyncIterator[str]:
    try:
        async for chunk in stream:
            yield _format_chunk(chunk)
    except Exception as e:
        adapter_exception = to_adapter_exception(e)

        logger.exception(
            f"Caught exception while streaming: {type(e).__module__}.{type(e).__name__}. "
            f"Converted to the adapter exception: {adapter_exception!r}"
        )

        yield _format_chunk(adapter_exception.json_error())

    yield _END_CHUNK
