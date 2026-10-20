import inspect
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, TypeVar

from aidial_sdk.exceptions import InvalidRequestError


@lru_cache(maxsize=64)
def _method_signature(func: Callable[..., Any]) -> inspect.Signature:
    signature = inspect.signature(func)
    return signature.replace(parameters=list(signature.parameters.values())[1:])


def _inspect_signature(
    func: Callable[..., Awaitable[Any]],
) -> inspect.Signature:
    method = func if hasattr(func, "__func__") else inspect.unwrap(func)
    return _method_signature(method.__func__)


T = TypeVar("T")


async def call_with_extra_body(
    func: Callable[..., Awaitable[T]], arg: dict
) -> T:
    if has_kwargs_argument(func):
        return await func(**arg)

    expected_args = set(_inspect_signature(func).parameters.keys())
    actual_args = set(arg.keys())

    extra_args = actual_args - expected_args

    if extra_args and "extra_body" not in expected_args:
        raise InvalidRequestError(
            f"Extra arguments aren't supported: {extra_args}."
        )

    arg["extra_body"] = arg.get("extra_body") or {}

    for extra_arg in extra_args:
        arg["extra_body"][extra_arg] = arg[extra_arg]
        del arg[extra_arg]

    return await func(**arg)


def has_kwargs_argument(func: Callable[..., Awaitable[Any]]) -> bool:
    """
    Determines if the given function accepts a variable keyword argument (**kwargs).
    """
    signature = _inspect_signature(func)
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False
