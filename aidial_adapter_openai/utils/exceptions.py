from typing import Any

from aidial_sdk.exceptions import HTTPException as DialException
from fastapi.responses import Response
from openai import error

from aidial_adapter_openai.openai_override import OpenAIException


def create_error(message: str, type: str, param: Any = None, code: Any = None):
    return {
        "error": {
            "message": message,
            "type": type,
            "param": param,
            "code": code,
        }
    }


async def handle_exceptions(call):
    try:
        return await call
    except OpenAIException as e:
        return Response(status_code=e.code, headers=e.headers, content=e.body)
    except error.Timeout:
        raise DialException(
            "Request timed out",
            504,
            "timeout",
            display_message="Request timed out. Please try again later.",
        )
    except error.APIConnectionError:
        raise DialException(
            "Error communicating with OpenAI",
            502,
            "connection",
            display_message="OpenAI server is not responsive. Please try again later.",
        )
