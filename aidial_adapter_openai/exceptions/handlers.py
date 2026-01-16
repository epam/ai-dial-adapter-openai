import re
from functools import wraps

import fastapi
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from fastapi.requests import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from fastapi.responses import Response as FastAPIResponse

from aidial_adapter_openai.exceptions.anthropic import convert_anthropic_errors
from aidial_adapter_openai.exceptions.application import (
    convert_application_errors,
)
from aidial_adapter_openai.exceptions.openai import convert_openai_exception
from aidial_adapter_openai.utils.adapter_exception import (
    AdapterException,
    ResponseWrapper,
)
from aidial_adapter_openai.utils.log_config import logger


def to_adapter_exception(exc: Exception) -> AdapterException:
    e = (
        convert_openai_exception(exc)
        or convert_anthropic_errors(exc)
        or convert_application_errors(exc)
        or InternalServerError(str(exc))
    )
    return _expose_error_message_to_user(e)


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
    request: FastAPIRequest, exc: Exception
) -> FastAPIResponse:
    adapter_exception = to_adapter_exception(exc)

    logger.error(
        f"Caught exception: {type(exc).__module__}.{type(exc).__name__}. "
        f"Converted to the adapter exception: {adapter_exception!r}",
        exc_info=exc,
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
