import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
A = TypeVar("A")


async def make_async(func: Callable[[A], T], arg: A) -> T:
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, arg)
