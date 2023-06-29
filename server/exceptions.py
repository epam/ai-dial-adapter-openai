from functools import wraps
from typing import Literal, Optional, TypedDict

from google.api_core.exceptions import InvalidArgument, PermissionDenied
from google.auth.exceptions import GoogleAuthError

from utils.printing import print_exception


class OpenAIError(TypedDict):
    type: Literal["invalid_request_error", "internal_server_error"] | str
    message: str
    param: Optional[str]
    code: Optional[str]


class OpenAIException(Exception):
    def __init__(self, status_code: int, error: OpenAIError):
        self.status_code = status_code
        self.error = error


def error_handling_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GoogleAuthError as e:
            raise OpenAIException(
                status_code=401,
                error={
                    "type": "invalid_request_error",
                    "message": f"Invalid Authentication: {str(e)}",
                    "code": "invalid_api_key",
                    "param": None,
                },
            )
        except PermissionDenied as e:
            raise OpenAIException(
                status_code=403,
                error={
                    "type": "invalid_request_error",
                    "message": f"Permission denied: {str(e)}",
                    "code": "permission_denied",
                    "param": None,
                },
            )
        except InvalidArgument as e:
            raise OpenAIException(
                status_code=400,
                error={
                    "type": "invalid_request_error",
                    "message": f"Invalid argument: {str(e)}",
                    "code": "invalid_argument",
                    "param": None,
                },
            )
        except Exception as e:
            print_exception()
            raise OpenAIException(
                status_code=500,
                error={
                    "type": "internal_server_error",
                    "message": str(e),
                    "code": None,
                    "param": None,
                },
            )

    return wrapper
