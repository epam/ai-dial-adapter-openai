import pydantic
from aidial_sdk._errors import pydantic_validation_exception_handler
from aidial_sdk.exceptions import HTTPException as DialException
from fastapi import Request
from fastapi.responses import Response
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError


def openai_exception_handler(request: Request, e: DialException):
    if isinstance(e, APIStatusError):
        r = e.response
        headers = r.headers

        # Avoid encoding the error message when the original response was encoded.
        if "Content-Encoding" in headers:
            del headers["Content-Encoding"]

        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=headers,
        )

    if isinstance(e, APITimeoutError):
        raise DialException(
            status_code=504,
            type="timeout",
            message="Request timed out",
            display_message="Request timed out. Please try again later.",
        )

    if isinstance(e, APIConnectionError):
        raise DialException(
            status_code=502,
            type="connection",
            message="Error communicating with OpenAI",
            display_message="OpenAI server is not responsive. Please try again later.",
        )

    if isinstance(e, APIError):
        raise DialException(
            status_code=getattr(e, "status_code", None) or 500,
            message=e.message,
            type=e.type,
            code=e.code,
            param=e.param,
            display_message=None,
        )


def pydantic_exception_handler(request: Request, exc: pydantic.ValidationError):
    return pydantic_validation_exception_handler(request, exc)


def dial_exception_handler(request: Request, exc: DialException):
    return exc.to_fastapi_response()
