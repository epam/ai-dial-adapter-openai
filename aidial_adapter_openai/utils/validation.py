from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, LiteralString, TypeVar, overload

import pydantic
from aidial_sdk.exceptions import InvalidRequestError

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


def _type_name(ty: type) -> str:
    return "null" if ty is type(None) else ty.__name__


def _repr(value: object) -> str:
    return "null" if value is None else repr(value)


def to_request_error(
    e: pydantic.ValidationError, path: _Path = ()
) -> InvalidRequestError:
    error = e.errors()[0]
    location = _format_path((*path, *error["loc"]))
    return InvalidRequestError(
        f"Invalid request. Path: '{location}', error: {error['msg']}"
    )


@dataclass(frozen=True, slots=True, eq=False)
class RequestValue:
    """A value of an unvalidated request together with its path in the request.

    Nothing is validated upfront: each accessor validates only the step
    it takes, and an error names the exact path of the offending value.
    """

    obj: Any
    path: _Path = ()

    def __str__(self) -> str:
        return str(self.obj)

    def __bool__(self) -> bool:
        return bool(self.obj)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RequestValue):
            other = other.obj
        return self.obj == other

    def unwrap(self) -> Any:
        return self.obj

    def unwrap_dict(self) -> dict[str, Any]:
        return self._ensure(dict)

    def __iter__(self) -> Iterator["RequestValue"]:
        return iter(self.to_list())

    def __setitem__(self, key: str, value: Any) -> None:
        self._ensure(dict)[key] = value

    @overload
    def __getitem__(self, key: int | str) -> "RequestValue": ...

    @overload
    def __getitem__(self, key: slice) -> list["RequestValue"]: ...

    def __getitem__(
        self, key: int | str | slice
    ) -> "RequestValue | list[RequestValue]":
        """A list item by index (negative counts from the end),
        list items by slice or a required dict field by key."""
        if isinstance(key, slice):
            return self.to_list()[key]

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

    def pop(self, key: str) -> "RequestValue | None":
        """Removes an optional dict field: None when it's missing or null."""
        value = self._ensure(dict).pop(key, None)
        return None if value is None else self._child(key, value)

    def copy(self) -> "RequestValue":
        """A shallow copy of a dict value."""
        return RequestValue(self._ensure(dict).copy(), self.path)

    def update(self, **fields: Any) -> "RequestValue":
        """A shallow copy of a dict value with the given fields replaced."""
        return RequestValue({**self._ensure(dict), **fields}, self.path)

    def pop_int(self, key: str) -> int | None:
        return None if (value := self.pop(key)) is None else value.to_int()

    def get_str(self, key: str) -> str | None:
        return None if (value := self.get(key)) is None else value.to_str()

    def get_int(self, key: str) -> int | None:
        return None if (value := self.get(key)) is None else value.to_int()

    def get_float(self, key: str) -> float | None:
        return None if (value := self.get(key)) is None else value.to_float()

    def get_bool(self, key: str) -> bool | None:
        return None if (value := self.get(key)) is None else value.to_bool()

    def get_dict(self, key: str) -> dict[str, Any] | None:
        """An optional dict field as is."""
        return None if (value := self.get(key)) is None else value.unwrap_dict()

    def to_list(self) -> list["RequestValue"]:
        return [self._child(i, v) for i, v in enumerate(self._ensure(list))]

    def to_dict(self) -> dict[str, "RequestValue"]:
        return {k: self._child(k, v) for k, v in self._ensure(dict).items()}

    def to_str(self) -> str:
        return self._ensure(str)

    def to_int(self) -> int:
        return self._ensure(int)

    def to_float(self) -> float:
        return self._ensure((int, float))

    def to_bool(self) -> bool:
        return self._ensure(bool)

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
            raise to_request_error(e, self.path) from e

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
        expected = expected if isinstance(expected, tuple) else (expected,)
        # JSON true/false aren't numbers, though bool subclasses int in Python
        is_bool_as_number = isinstance(self.obj, bool) and bool not in expected
        if isinstance(self.obj, expected) and not is_bool_as_number:
            return self.obj
        expected_names = " or ".join(map(_type_name, expected))
        if len(expected) > 1:
            expected_names = f"either {expected_names}"
        raise InvalidRequestError(
            f"{self._location!r} expected to be {expected_names}, "
            f"but got {_type_name(type(self.obj))}"
        )
