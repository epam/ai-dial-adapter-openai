from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


def once(func: Callable[_P, _T]) -> Callable[_P, _T | None]:
    _called = False

    def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T | None:
        nonlocal _called
        if _called:
            return None
        _called = True
        return func(*args, **kwargs)

    return _wrapper
