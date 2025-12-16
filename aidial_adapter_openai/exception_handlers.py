import re
from functools import wraps

import fastapi
import httpx
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from fastapi.requests import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from fastapi.responses import Response as FastAPIResponse
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from aidial_adapter_openai.utils.adapter_exception import (
    AdapterException,
    ResponseWrapper,
    parse_adapter_exception,
)
from aidial_adapter_openai.utils.log_config import logger


def to_adapter_exception(exc: Exception) -> AdapterException:
    return _expose_error_message_to_user(_convert_to_adapter_exception(exc))


def _convert_to_adapter_exception(exc: Exception) -> AdapterException:

    if isinstance(exc, (DialException, ResponseWrapper)):
        return exc

    if isinstance(exc, httpx.HTTPStatusError):
        r = exc.response
        if ret := parse_adapter_exception(
            status_code=r.status_code,
            headers={},
            content=r.text,
        ):
            return ret

    if isinstance(exc, APIStatusError):
        # Non-streaming errors are reported by `openai` library via this exception
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
        # Streaming errors are reported by `openai` library via this exception
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


def _truncate_long_string(s: str, *, limit: int) -> str:
    if (excess := len(s) - limit) > 0:
        ln = limit // 2
        rn = limit - ln
        return s[:ln] + f"<truncated {excess} characters>" + s[-rn:]
    return s


def _expose_error_message_to_user(exc: AdapterException) -> AdapterException:
    if isinstance(exc, DialException) and exc.status_code == 400:
        message = exc.message
        if (
            "this model does not support file content types" in message.lower()
            or "the file type you uploaded is not supported" in message.lower()
        ):
            exc.display_message = (
                exc.display_message
                or "The provided file attachments aren't supported."
            )

        match = re.search(
            r"unsupported MIME type\s+(['\"])([^'\"]+)\1", message
        )
        if match:
            mime_type = match[2]
            exc.display_message = (
                exc.display_message
                or f"The file attachments of the MIME type {mime_type!r} aren't supported."
            )

        if (
            "invalid image data" in message.lower()
            or "the image data you provided does not represent a valid image"
            in message.lower()
        ):
            exc.display_message = (
                exc.display_message
                or "The provided image attachment is either corrupt or of unsupported MIME type."
            )

        # Special handling of GPT Image 1 exception when the prompt is too long
        if "Invalid 'prompt': string too long" in message:
            exc.display_message = (
                exc.display_message or "The prompt is too long."
            )

        # Special handling of DALL·E 3 exception when the prompt is too long
        if "is too long - 'prompt'" in message:
            # DALL·E 3 is notorious for including the whole prompt in the error message,
            # therefore, we override it with a short one.
            exc.message = "The prompt is too long."
            exc.display_message = exc.display_message or exc.message

        # Just in case any other sensitive information leaked to the error message, we truncate it
        exc.message = _truncate_long_string(exc.message, limit=1024)

    return exc


def fastapi_exception_handler(
    request: FastAPIRequest, exc: Exception
) -> FastAPIResponse:
    assert isinstance(exc, fastapi.HTTPException)
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
        headers=exc.headers,
    )


def adapter_exception_handler(
    request: FastAPIRequest, e: Exception
) -> FastAPIResponse:
    adapter_exception = to_adapter_exception(e)

    logger.error(
        f"Caught exception: {type(e).__module__}.{type(e).__name__}. "
        f"Converted to the adapter exception: {adapter_exception!r}",
        exc_info=e,
    )
    return adapter_exception.to_fastapi_response()


def _to_dial_exception(exc: Exception) -> DialException:
    exc = to_adapter_exception(exc)
    if isinstance(exc, ResponseWrapper):
        return exc.to_dial_exception()
    else:
        return exc


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            dial_exception = _to_dial_exception(e)
            logger.exception(
                f"Caught exception: {type(e).__module__}.{type(e).__name__}. "
                f"The exception converted to the dial exception: {dial_exception!r}."
            )
            raise dial_exception from e

    return wrapper
