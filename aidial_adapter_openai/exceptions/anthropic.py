import re

from aidial_sdk.exceptions import HTTPException as DialException
from openai import APIStatusError

from aidial_adapter_openai.utils.adapter_exception import AdapterException


def _create_error(
    status_code: int, message: str, headers: dict[str, str] | None = None
) -> DialException:
    return DialException(
        status_code=status_code,
        type=_get_exception_type(status_code),
        message=message,
        headers=headers,
    )


def _get_exception_type(status_code: int) -> str | None:
    if status_code in {400, 422}:
        return "invalid_request_error"
    if status_code == 500:
        return "internal_server_error"
    return None


def _get_error_message(e: APIStatusError) -> str:
    if isinstance(body := e.body, dict):
        if isinstance((msg := body.get("message")), str):
            return msg
    return e.message


def _parse_streaming_error(text: str) -> DialException | None:
    # Unfortunately, anthropic SDK obscures the original error message:
    # https://github.com/anthropics/anthropic-sdk-python/blob/8b244157a7d03766bec645b0e1dc213c6d462165/src/anthropic/lib/bedrock/_stream_decoder.py#L57-L58
    # So we have to parse it manually.

    prefix = "Bad response code, expected 200: "
    if not text.startswith(prefix):
        return None
    text = text.removeprefix(prefix)

    code_pattern = re.search(r"'status_code':\s*(\d+)", text)
    message_pattern = re.search(r"\"message\":\s*\"(.*?)\"", text)

    code = int(code_pattern.group(1)) if code_pattern else None
    message = str(message_pattern.group(1)) if message_pattern else None

    if code and message:
        message = message.replace("\\'", "'")
        return _create_error(code, message)
    return None


def _copy_headers(e: APIStatusError, keys: list[str]) -> dict[str, str] | None:
    copied_headers: dict[str, str] = {}
    for key in keys:
        if key in e.response.headers:
            copied_headers[key] = e.response.headers[key]
    return copied_headers or None


def convert_anthropic_errors(e: Exception) -> AdapterException | None:
    if isinstance(e, APIStatusError):
        message = _get_error_message(e)
        # We want to save Retry-After header if it's present:
        # https://platform.claude.com/docs/en/api/rate-limits#tier-1

        headers = _copy_headers(e, ["Retry-After"])
        return _create_error(e.status_code, message, headers)

    if isinstance(e, ValueError):
        exc = _parse_streaming_error(str(e))
        if exc is not None:
            return exc

    return None
