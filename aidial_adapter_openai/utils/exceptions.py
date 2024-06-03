from typing import Any


def create_error(message: str, type: str, param: Any = None, code: Any = None):
    return {
        "error": {
            "message": message,
            "type": type,
            "param": param,
            "code": code,
        }
    }
