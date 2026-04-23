from collections.abc import Callable, Sequence
from typing import TypeVar

import pytest

_T = TypeVar("_T")


def maybe_parametrized_fixture(
    *, params: Sequence[_T], ids: Callable[[_T], str], skip_reason: str
):
    def decorator(func):
        @pytest.fixture(
            params=list(params) or [None],
            ids=lambda x: ids(x) if x is not None else "none",
        )
        def wrapper(request):
            value = request.param
            if value is None:
                pytest.skip(skip_reason)
            return func(value)

        return wrapper

    return decorator
