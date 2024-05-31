from functools import wraps
from typing import Any
from aidial_sdk.exceptions import HTTPException as DialException
from openai import error
from aidial_adapter_openai.errors import UserError, ValidationError
from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.log_config import logger


def to_dial_exception(e: Exception) -> DialException:
    if isinstance(e, DialException):
        return e
    if isinstance(e, UserError):
        return e.to_dial_exception()
    elif isinstance(e, ValidationError):
        return e.to_dial_exception()
    elif isinstance(e, error.Timeout):
        return DialException("Request timed out", 504, "timeout")
    elif isinstance(e, error.APIConnectionError):
        return DialException(
            "Error communicating with OpenAI", 502, "connection"
        )
    elif isinstance(e, OpenAIException):
        return DialException(e.body, e.code)
    else:
        return DialException(
            status_code=500,
            type="internal_server_error",
            message=str(e),
            code=None,
            param=None,
        )


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise to_dial_exception(e) from e

    return wrapper


def dial_exception_decorator_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise to_dial_exception(e) from e

    return wrapper


def create_error(message: str, type: str, param: Any = None, code: Any = None):
    return {
        "error": {
            "message": message,
            "type": type,
            "param": param,
            "code": code,
        }
    }
