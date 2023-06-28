from typing import Callable, TypeVar

T = TypeVar("T")

Unary = Callable[[T], T]


def identity(x: T) -> T:
    return x
