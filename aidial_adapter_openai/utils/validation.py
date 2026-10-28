from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Final, Never

import pydantic
from aidial_sdk.exceptions import InvalidRequestError

MISSING: Final = object()
"""The `dict.get` default for a required field,
which tells a missing field apart from a null one."""


def _type_name(ty: type) -> str:
    return "null" if ty is type(None) else ty.__name__


def _repr(value: object) -> str:
    return "null" if value is None else repr(value)


def _ensure_type(
    name: str, value: Any, expected: type | tuple[type, ...]
) -> Any:
    expected = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, expected):
        return value
    if value is MISSING:
        raise InvalidRequestError(f"Missing required parameter: {name!r}")
    expected_names = " or ".join(map(_type_name, expected))
    raise InvalidRequestError(
        f"{name!r} expected to be {expected_names}, but got {_type_name(type(value))}"
    )


def ensure_dict(name: str, value: Any, *required_keys: str) -> dict:
    value = _ensure_type(name, value, dict)
    for key in required_keys:
        if key not in value:
            raise InvalidRequestError(
                f"Missing required parameter: '{name}.{key}'"
            )
    return value


def ensure_str_or_none(name: str, value: Any) -> str | None:
    return _ensure_type(name, value, (str, type(None)))


def ensure_list_or_str(name: str, value: Any) -> list | str:
    return _ensure_type(name, value, (list, str))


def ensure_str(name: str, value: Any) -> str:
    return _ensure_type(name, value, str)


def ensure_list(name: str, value: Any) -> list:
    return _ensure_type(name, value, list)


def invalid_value(name: str, value: Never) -> InvalidRequestError:
    """`assert_never` for a value coming from an unvalidated request:
    the match is still checked for exhaustiveness statically,
    while at runtime an unexpected value is a client error."""
    return InvalidRequestError(f"Invalid value of {name!r}: {_repr(value)}")


@contextmanager
def request_validation(*path_prefix: str) -> Generator[None, None, None]:
    try:
        yield
    except pydantic.ValidationError as e:
        error = e.errors()[0]
        path = ".".join(map(str, (*path_prefix, *error["loc"])))
        raise InvalidRequestError(
            f"Invalid request. Path: '{path}', error: {error['msg']}"
        ) from e
