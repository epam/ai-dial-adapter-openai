import json
from collections.abc import MutableMapping
from typing import Any

from aidial_sdk.exceptions import HTTPException as DialException
from fastapi.responses import Response as FastAPIResponse


class ResponseWrapper(Exception):
    content: str
    status_code: int
    headers: MutableMapping[str, str] | None

    def __init__(
        self,
        *,
        content: str,
        status_code: int,
        headers: MutableMapping[str, str] | None,
    ) -> None:
        super().__init__(content)
        self.content = content
        self.status_code = status_code
        self.headers = headers

    def __repr__(self):
        # headers field is omitted deliberately
        # since it may contain sensitive information
        return f"{self.__class__.__name__}(content={self.content!r}, status_code={self.status_code!r})"

    def to_fastapi_response(self) -> FastAPIResponse:
        return FastAPIResponse(
            status_code=self.status_code,
            content=self.content,
            headers=self.headers,
        )

    def json_error(self) -> dict:
        return {
            "error": {
                "message": self.content,
                "code": self.status_code,
            }
        }

    def to_dial_exception(self) -> DialException:
        return DialException(
            status_code=self.status_code,
            message=self.content,
            headers=dict(self.headers or {}),
        )


AdapterException = ResponseWrapper | DialException


def _parse_dial_exception(
    *, status_code: int, headers: MutableMapping[str, str], content: Any
) -> DialException | None:
    if isinstance(content, str):
        stripped_content = content.strip()
        parsers = (
            lambda: json.loads(content),
            lambda: json.JSONDecoder().raw_decode(stripped_content)[0],
        )

        for parser in parsers:
            try:
                obj = parser()
                break
            except json.JSONDecodeError:
                continue
        else:
            return None
    else:
        obj = content

    # The content length is invalidated as soon as
    # the original content is lost
    if "Content-Length" in headers:
        del headers["Content-Length"]

    if (
        isinstance(obj, dict)
        and (error := obj.get("error"))
        and isinstance(error, dict)
    ):
        error = error.copy()
        message = error.pop("message", None) or "Unknown error"
        code = error.pop("code", None)
        type = error.pop("type", None)
        param = error.pop("param", None)
        display_message = error.pop("display_message", None)

        # Content filter codes for DALL-E3 and GPT-Image-1 are different
        # from the GPT content filter code.
        if code in [
            "content_policy_violation",
            "moderation_blocked",
            "contentFilter",
        ]:
            code = "content_filter"
        return DialException(
            status_code=status_code,
            message=message,
            type=type,
            param=param,
            code=code,
            display_message=display_message,
            headers=dict(headers.items()),
            **error,
        )

    return None


def parse_adapter_exception(
    *, status_code: int, headers: MutableMapping[str, str], content: Any
) -> AdapterException:
    return _parse_dial_exception(
        status_code=status_code, headers=headers, content=content
    ) or ResponseWrapper(
        status_code=status_code, headers=headers, content=str(content)
    )
