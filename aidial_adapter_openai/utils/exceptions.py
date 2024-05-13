from typing import Any, Optional


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
):
    return {
        "error": {
            "message": message,
            "type": type,
            "param": param,
            "code": code,
        }
    }
