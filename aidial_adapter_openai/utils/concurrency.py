import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_THREAD_POOL_SIZE = os.getenv("THREAD_POOL_SIZE")
_THREAD_POOL_SIZE = (
    int(_THREAD_POOL_SIZE) if _THREAD_POOL_SIZE is not None else None
)

_THREAD_POOL = ThreadPoolExecutor(max_workers=_THREAD_POOL_SIZE)


_T = TypeVar("_T")


async def run_in_threadpool(func: Callable[[], _T]) -> _T:
    return await asyncio.get_running_loop().run_in_executor(_THREAD_POOL, func)
