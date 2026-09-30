import json
from collections import OrderedDict
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
    maxsize: int | None = None,
) -> Callable[[Callable[_P, _R]], _CachedFunction[_P, _R]]:
    if maxsize is not None and close is not None:
        raise ValueError("A bounded cache cannot close the evicted values")

    def wrapper(f: Callable[_P, _R]) -> _CachedFunction[_P, _R]:
        func_name = f"{f.__module__}.{f.__qualname__}"

        class wrapped:
            _cache: OrderedDict[str, _R]

            def __init__(self) -> None:
                self._cache = OrderedDict()

            def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
                key = json.dumps(
                    {"args": args, "kwargs": kwargs}, sort_keys=True
                )

                if (value := self._cache.get(key)) is None:
                    value = self._cache[key] = f(*args, **kwargs)
                    self._evict_overflow()
                else:
                    self._cache.move_to_end(key)

                return value

            def _evict_overflow(self) -> None:
                while maxsize is not None and len(self._cache) > maxsize:
                    evicted, _ = self._cache.popitem(last=False)

                    log.debug(
                        f"Evicting the least recently used entry of "
                        f"{func_name}({evicted}), "
                        f"{len(self._cache)} entries left"
                    )

            async def clear(self):
                entries = self._cache
                self._cache = OrderedDict()

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
