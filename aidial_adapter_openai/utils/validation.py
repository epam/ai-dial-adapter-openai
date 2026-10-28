from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, LiteralString, Never, TypeVar

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
    # JSON true/false aren't numbers, though bool subclasses int in Python
    is_bool_as_number = isinstance(value, bool) and bool not in expected
    if isinstance(value, expected) and not is_bool_as_number:
        return value
    if value is MISSING:
        raise InvalidRequestError(f"Missing required parameter: {name!r}")
    expected_names = " or ".join(map(_type_name, expected))
    if len(expected) > 1:
        expected_names = f"either {expected_names}"
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


_Path = tuple[str | int, ...]
_Model = TypeVar("_Model", bound=pydantic.BaseModel)
# LiteralString, unlike str, keeps the literal types of the arguments
_Literal = TypeVar("_Literal", bound=LiteralString)


def _format_path(path: _Path) -> str:
    """("messages", 0, "content") -> "messages[0].content\" """
    ret = ""
    for segment in path:
        if isinstance(segment, int):
            ret += f"[{segment}]"
        else:
            ret += f".{segment}" if ret else segment
    return ret or "request"


@dataclass(frozen=True, slots=True)
class RequestValue:
    """A value of an unvalidated request together with its path in the request.

    Nothing is validated upfront: each accessor validates only the step
    it takes, and an error names the exact path of the offending value.
    """

    obj: Any
    path: _Path = ()

    def unwrap(self) -> Any:
        return self.obj

    def __getitem__(self, key: int | str) -> "RequestValue":
        """A list item by index (negative counts from the end)
        or a required dict field by key."""
        if isinstance(key, int):
            items = self._ensure(list)
            index = key if key >= 0 else len(items) + key
            if not 0 <= index < len(items):
                raise InvalidRequestError(
                    f"{self._location!r} has no item at index {key}"
                )
            return self._child(index, items[index])

        fields = self._ensure(dict)
        if key not in fields:
            location = _format_path((*self.path, key))
            raise InvalidRequestError(
                f"Missing required parameter: {location!r}"
            )
        return self._child(key, fields[key])

    def get(self, key: str) -> "RequestValue | None":
        """An optional dict field: None when it's missing or null."""
        value = self._ensure(dict).get(key)
        return None if value is None else self._child(key, value)

    def get_str(self, key: str) -> str | None:
        return None if (value := self.get(key)) is None else value.to_str()

    def get_int(self, key: str) -> int | None:
        return None if (value := self.get(key)) is None else value.to_int()

    def to_list(self) -> list["RequestValue"]:
        return [self._child(i, v) for i, v in enumerate(self._ensure(list))]

    def to_dict(self) -> dict[str, "RequestValue"]:
        return {k: self._child(k, v) for k, v in self._ensure(dict).items()}

    def to_str(self) -> str:
        return self._ensure(str)

    def to_int(self) -> int:
        return self._ensure(int)

    def to_str_or_list(self) -> str | list["RequestValue"]:
        value = self._ensure((str, list))
        return value if isinstance(value, str) else self.to_list()

    def to_literal(self, *supported: _Literal) -> _Literal:
        """The value as one of the supported string literals,
        e.g. `to_literal("user", "system")` is `Literal["user", "system"]`."""
        value = self.to_str()
        for literal in supported:
            if literal == value:
                return literal
        raise self.invalid(*supported)

    def to_model(self, cls: type[_Model]) -> _Model:
        try:
            return cls.model_validate(self.obj)
        except pydantic.ValidationError as e:
            error = e.errors()[0]
            location = _format_path((*self.path, *error["loc"]))
            raise InvalidRequestError(
                f"Invalid request. Path: {location!r}, error: {error['msg']}"
            ) from e

    def invalid(self, *supported: str) -> InvalidRequestError:
        """For the catch-all case of a match over the value."""
        message = f"Invalid value of {self._location!r}: {_repr(self.obj)}"
        if supported:
            message += f". Supported values: {', '.join(map(repr, supported))}"
        return InvalidRequestError(message)

    @property
    def _location(self) -> str:
        return _format_path(self.path)

    def _child(self, key: int | str, value: Any) -> "RequestValue":
        return RequestValue(value, (*self.path, key))

    def _ensure(self, expected: type | tuple[type, ...]) -> Any:
        return _ensure_type(self._location, self.obj, expected)
