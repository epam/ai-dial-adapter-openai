import contextlib
from typing import Any, Dict

from aidial_adapter_openai.utils.aiohttp.trace_config import get_tracing_timings
from aidial_adapter_openai.utils.http_client import get_aiohttp_session
from aidial_adapter_openai.utils.log_config import logger


@contextlib.asynccontextmanager
async def post(url: str, headers: Dict[str, str], request: Any):
    ctx = {}
    async with get_aiohttp_session().post(
        url, json=request, headers=headers, trace_request_ctx=ctx
    ) as response:
        try:
            yield response
        finally:
            logger.info(
                f"Upstream: {url!r}. Status: {response.status}. Timing: {get_tracing_timings(ctx)}."
            )
