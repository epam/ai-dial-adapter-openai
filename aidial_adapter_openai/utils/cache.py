import json
from collections.abc import Callable, Coroutine
from typing import Generic, ParamSpec, Protocol, TypeVar

from aidial_adapter_openai.utils.log_config import logger as log

_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)


class _CachedFunction(Protocol, Generic[_P, _R]):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    async def clear(self): ...


def cache(
    close: Callable[[_R], Coroutine[None, None, None]] | None = None,
) -> Callable[[Callable[_P, _R]], _CachedFunction[_P, _R]]:
    def wrapper(f: Callable[_P, _R]) -> _CachedFunction[_P, _R]:
        class wrapped:
            _cache: dict[str, _R]

            def __init__(self) -> None:
                self._cache = {}

            def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
                key = json.dumps(
                    {"args": args, "kwargs": kwargs}, sort_keys=True
                )

                if (value := self._cache.get(key)) is None:
                    value = self._cache[key] = f(*args, **kwargs)

                return value

            async def clear(self):
                entries = self._cache
                self._cache = {}

                func_name = f"{f.__module__}.{f.__qualname__}"
                log.debug(f"Clearing cache {func_name}")

                for key, value in entries.items():
                    log.debug(f"Closing cached value {func_name}({key})")

                    try:
                        if close is not None:
                            await close(value)
                    except Exception as e:
                        log.error(f"Error on closing the task: {e}")

        return wrapped()

    return wrapper
