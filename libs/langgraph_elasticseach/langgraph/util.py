from typing import Callable, TypeVar
import asyncio

T = TypeVar("T")

def syncify(func: Callable[..., T], *args, **kwargs) -> T:
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(func(*args, **kwargs), loop)
        return future.result()
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))