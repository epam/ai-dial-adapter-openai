from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from fastapi.requests import Request as FastAPIRequest
from fastapi.responses import Response as FastAPIResponse
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from aidial_adapter_openai.utils.adapter_exception import (
    AdapterException,
    ResponseWrapper,
    parse_adapter_exception,
)
from aidial_adapter_openai.utils.log_config import logger


def to_adapter_exception(exc: Exception) -> AdapterException:

    if isinstance(exc, (DialException, ResponseWrapper)):
        return exc

    if isinstance(exc, APIStatusError):
        # Non-streaming errors reported by `openai` library via this exception
        r = exc.response
        httpx_headers = r.headers

        # httpx library (used by openai) automatically sets
        # "Accept-Encoding:gzip,deflate" header in requests to the upstream.
        # Therefore, we may receive from the upstream gzip-encoded
        # response along with "Content-Encoding:gzip" header.
        # We either need to encode the response, or
        # remove the "Content-Encoding" header.
        if "Content-Encoding" in httpx_headers:
            del httpx_headers["Content-Encoding"]

        return parse_adapter_exception(
            status_code=r.status_code,
            headers=httpx_headers,
            content=r.text,
        )

    if isinstance(exc, APITimeoutError):
        return DialException(
            status_code=504,
            type="timeout",
            message="Request timed out",
            display_message="Request timed out. Please try again later.",
        )

    if isinstance(exc, APIConnectionError):
        return DialException(
            status_code=502,
            type="connection",
            message="Error communicating with OpenAI",
            display_message="OpenAI server is not responsive. Please try again later.",
        )

    if isinstance(exc, APIError):
        # Streaming errors reported by `openai` library via this exception
        status_code: int = 500
        if exc.code:
            try:
                status_code = int(exc.code)
            except Exception:
                pass

        return parse_adapter_exception(
            status_code=status_code,
            headers={},
            content={"error": exc.body or {}},
        )

    return InternalServerError(str(exc))


def adapter_exception_handler(
    request: FastAPIRequest, e: Exception
) -> FastAPIResponse:
    adapter_exception = to_adapter_exception(e)

    logger.error(
        f"Caught exception: {type(e).__module__}.{type(e).__name__}. "
        f"Converted to the adapter exception: {adapter_exception!r}"
    )
    return adapter_exception.to_fastapi_response()
