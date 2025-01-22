import json
from typing import Any, MutableMapping

from aidial_sdk.exceptions import HTTPException as DialException
from fastapi.responses import Response as FastAPIResponse


class ResponseWrapper(Exception):
    content: Any
    status_code: int
    headers: MutableMapping[str, str] | None

    def __init__(
        self,
        *,
        content: Any,
        status_code: int,
        headers: MutableMapping[str, str] | None,
    ) -> None:
        super().__init__(str(content))
        self.content = content
        self.status_code = status_code
        self.headers = headers

    def __repr__(self):
        # headers field is omitted deliberately
        # since it may contain sensitive information
        return "%s(content=%r, status_code=%r)" % (
            self.__class__.__name__,
            self.content,
            self.status_code,
        )

    def to_fastapi_response(self) -> FastAPIResponse:
        return FastAPIResponse(
            status_code=self.status_code,
            content=self.content,
            headers=self.headers,
        )

    def json_error(self) -> dict:
        return {
            "error": {
                "message": str(self.content),
                "code": int(self.status_code),
            }
        }


AdapterException = ResponseWrapper | DialException


def _parse_dial_exception(
    *, status_code: int, headers: MutableMapping[str, str], content: Any
) -> DialException | None:
    if isinstance(content, str):
        try:
            obj = json.loads(content)
        except Exception:
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
        message = error.get("message") or "Unknown error"
        code = error.get("code")
        type = error.get("type")
        param = error.get("param")
        display_message = error.get("display_message")

        return DialException(
            status_code=status_code,
            message=message,
            type=type,
            param=param,
            code=code,
            display_message=display_message,
            headers=dict(headers.items()),
        )

    return None


def parse_adapter_exception(
    *, status_code: int, headers: MutableMapping[str, str], content: Any
) -> AdapterException:
    return _parse_dial_exception(
        status_code=status_code, headers=headers, content=content
    ) or ResponseWrapper(
        status_code=status_code, headers=headers, content=content
    )
