from typing import Any, Awaitable, Optional, TypeVar

from fastapi.responses import Response
from openai import APIConnectionError, APIStatusError, APITimeoutError


class HTTPException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        type: str = "runtime_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.type = type
        self.param = param
        self.code = code

    def __repr__(self):
        return "%s(message=%r, status_code=%r, type=%r, param=%r, code=%r)" % (
            self.__class__.__name__,
            self.message,
            self.status_code,
            self.type,
            self.param,
            self.code,
        )


def create_error(
    message: str, type: str | None, param: Any = None, code: Any = None
) -> dict:
    return {
        "error": {
            "message": message,
            "type": type,
            "param": param,
            "code": code,
        }
    }


T = TypeVar("T")


async def handle_exceptions(call: Awaitable[T]) -> T | Response:
    try:
        return await call
    except APIStatusError as e:
        r = e.response
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=r.headers,
        )
    except APITimeoutError:
        raise HTTPException("Request timed out", 504, "timeout")
    except APIConnectionError:
        raise HTTPException(
            "Error communicating with OpenAI", 502, "connection"
        )
