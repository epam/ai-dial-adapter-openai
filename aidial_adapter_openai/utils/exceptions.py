from typing import Any


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
