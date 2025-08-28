from typing import Any, Tuple, Type, Union

from aidial_sdk.exceptions import InvalidRequestError


def _ensure_type(
    name: str, value: Any, expected: Union[Type, Tuple[Type, ...]]
) -> Any:
    expected = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, expected):
        return value
    expected_names = " or ".join(ty.__name__ for ty in expected)
    raise InvalidRequestError(
        f"{name!r} expected to be {expected_names}, but got {type(value).__name__}"
    )


def ensure_dict(name: str, value: Any) -> dict:
    return _ensure_type(name, value, dict)


def ensure_str_or_none(name: str, value: Any) -> str | None:
    return _ensure_type(name, value, (str, type(None)))


def ensure_list_or_str(name: str, value: Any) -> list | str:
    return _ensure_type(name, value, (list, str))


def ensure_str(name: str, value: Any) -> str:
    return _ensure_type(name, value, str)


def ensure_list(name: str, value: Any) -> list:
    return _ensure_type(name, value, list)
