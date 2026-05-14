import http
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

_AUDIO_FILE_SIZE_LIMIT_EXCEEDED = "Audio file size exceeds the allowed limit."
_PROVIDER_AUDIO_SIZE_LIMIT_PATTERN = re.compile(
    r"Maximum content size limit \((\d+)\) exceeded \((\d+) bytes read\)"
)


def _format_size_mb(size_bytes: int) -> str:
    return f"{(size_bytes / (1024 * 1024)):.1f}".rstrip("0").rstrip(".")


def to_adapter_exception(e: Exception) -> AdapterException:
    e = (
        convert_openai_exception(e)
        or convert_anthropic_errors(e)
        or convert_application_errors(e)
        or InternalServerError(str(e))
    )
    return _expose_error_message_to_user(e)


def _truncate_long_string(s: str, *, limit: int) -> str:
    if (excess := len(s) - limit) > 0:
        ln = limit // 2
        rn = limit - ln
        return s[:ln] + f"<truncated {excess} characters>" + s[-rn:]
    return s


def _expose_error_message_to_user(e: AdapterException) -> AdapterException:
    if not isinstance(e, DialException):
        return e

    status_code = e.status_code
    message = e.message
    match status_code:
        case http.HTTPStatus.BAD_REQUEST:
            if (
                "this model does not support file content types"
                in message.lower()
                or "the file type you uploaded is not supported"
                in message.lower()
            ):
                e.display_message = (
                    e.display_message
                    or "The provided file attachments aren't supported."
                )

            match = re.search(
                r"unsupported MIME type\s+(['\"])([^'\"]+)\1", message
            )
            if match:
                mime_type = match[2]
                e.display_message = (
                    e.display_message
                    or f"The file attachments of the MIME type {mime_type!r} aren't supported."
                )

            if (
                "invalid image data" in message.lower()
                or "the image data you provided does not represent a valid image"
                in message.lower()
            ):
                e.display_message = (
                    e.display_message
                    or "The provided image attachment is either corrupt or of unsupported MIME type."
                )

            # Special handling of GPT Image 1 exception when the prompt is too long
            if "Invalid 'prompt': string too long" in message:
                e.display_message = (
                    e.display_message or "The prompt is too long."
                )

            # Special handling of DALL·E 3 exception when the prompt is too long
            if "is too long - 'prompt'" in message:
                # DALL·E 3 is notorious for including the whole prompt in the error message,
                # therefore, we override it with a short one.
                e.message = "The prompt is too long."
                e.display_message = e.display_message or e.message

            if "invalid file format" in message.lower():
                e.display_message = message

        case http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            if match := _PROVIDER_AUDIO_SIZE_LIMIT_PATTERN.search(message):
                limit_bytes = int(match.group(1))
                actual_bytes = int(match.group(2))
                e.display_message = (
                    f"Audio file size ({_format_size_mb(actual_bytes)}MB) exceeds "
                    f"the {_format_size_mb(limit_bytes)}MB limit."
                )

    # Just in case any other sensitive information leaked to the error message, we truncate it
    e.message = _truncate_long_string(e.message, limit=1024)
    return e


def fastapi_exception_handler(
    request: FastAPIRequest, e: Exception
) -> FastAPIResponse:
    assert isinstance(e, fastapi.HTTPException)
    return JSONResponse(
        status_code=e.status_code,
        content=e.detail,
        headers=e.headers,
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


def _to_dial_exception(e: Exception) -> DialException:
    e = to_adapter_exception(e)
    if isinstance(e, ResponseWrapper):
        return e.to_dial_exception()
    else:
        return e


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
