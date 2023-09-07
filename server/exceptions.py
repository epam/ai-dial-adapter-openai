from functools import wraps
from typing import Literal, Optional, TypedDict

from google.api_core.exceptions import (
    GoogleAPICallError,
    InvalidArgument,
    PermissionDenied,
)
from google.auth.exceptions import GoogleAuthError

from llm.exception import ValidationError


class OpenAIError(TypedDict):
    type: Literal["invalid_request_error", "internal_server_error"] | str
    message: str
    param: Optional[str]
    code: Optional[str]


class OpenAIException(Exception):
    def __init__(self, status_code: int, error: OpenAIError):
        self.status_code = status_code
        self.error = error

    def __str__(self) -> str:
        return f"OpenAIException(status_code={self.status_code}, error={self.error})"


def to_open_ai_exception(e: Exception) -> OpenAIException:
    if isinstance(e, GoogleAuthError):
        return OpenAIException(
            status_code=401,
            error={
                "type": "invalid_request_error",
                "message": f"Invalid Authentication: {str(e)}",
                "code": "invalid_api_key",
                "param": None,
            },
        )

    if isinstance(e, PermissionDenied):
        return OpenAIException(
            status_code=403,
            error={
                "type": "invalid_request_error",
                "message": f"Permission denied: {str(e)}",
                "code": "permission_denied",
                "param": None,
            },
        )

    if isinstance(e, InvalidArgument):
        return OpenAIException(
            status_code=400,
            error={
                "type": "invalid_request_error",
                "message": f"Invalid argument: {str(e)}",
                "code": "invalid_argument",
                "param": None,
            },
        )

    if isinstance(e, GoogleAPICallError):
        return OpenAIException(
            status_code=e.code or 500,
            error={
                "type": "invalid_request_error",
                "message": f"Invalid argument: {str(e)}",
                "code": None,
                "param": None,
            },
        )

    if isinstance(e, ValidationError):
        return OpenAIException(
            status_code=422,
            error={
                "type": "invalid_request_error",
                "message": e.message,
                "code": "invalid_argument",
                "param": None,
            },
        )

    return OpenAIException(
        status_code=500,
        error={
            "type": "internal_server_error",
            "message": str(e),
            "code": None,
            "param": None,
        },
    )


def open_ai_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise to_open_ai_exception(e)

    return wrapper
