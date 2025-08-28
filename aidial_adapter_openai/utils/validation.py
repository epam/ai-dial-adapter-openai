from typing import Any

from aidial_sdk.exceptions import InvalidRequestError


def ensure_dict(name: str, value: Any) -> dict:
    if isinstance(value, dict):
        return value
    raise InvalidRequestError(
        f"{name!r} expected to be a dictionary, but got {type(value).__name__!r}"
    )


def ensure_str_or_none(name: str, value: Any) -> str | None:
    if isinstance(value, str) or value is None:
        return value
    raise InvalidRequestError(
        f"{name!r} expected to be a string, but got {type(value).__name__!r}"
    )


def ensure_list_or_str(name: str, value: Any) -> list | str:
    if isinstance(value, (str, list)):
        return value
    raise InvalidRequestError(
        f"{name!r} expected to be a list or string, but got {type(value).__name__!r}"
    )


def ensure_str(name: str, value: Any) -> str:
    if isinstance(value, str):
        return value
    raise InvalidRequestError(
        f"{name!r} expected to be a string, but got {type(value).__name__!r}"
    )


def ensure_list(name: str, value: Any) -> list:
    if isinstance(value, list):
        return value
    raise InvalidRequestError(
        f"{name!r} expected to be a list, but got {type(value).__name__!r}"
    )
