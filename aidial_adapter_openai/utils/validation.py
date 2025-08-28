from __future__ import annotations

from dataclasses import dataclass
from typing import List

from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel


@dataclass
class _PathContext:
    validator: RequestValidationMixin
    original_path: List[str | int]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type in (TypeError, LookupError, AttributeError):
            raise InvalidRequestError(
                message=str(exc_value),
                param=self.validator.display_path(),
            ) from exc_value

        self.validator.path = self.original_path
        return False


class RequestValidationMixin(BaseModel):
    path: List[str | int] = []

    def display_path(self) -> str:
        return "".join(
            f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in self.path
        ).lstrip(".")

    def path_(self, *key: str | int) -> _PathContext:
        old_path = self.path.copy()
        self.path.extend(key)
        return _PathContext(validator=self, original_path=old_path)
