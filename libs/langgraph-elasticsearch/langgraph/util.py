from functools import wraps
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

def validate_before_execution(func):
    """Decorator para validar before_execution antes de executar uma função."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "before_execution") and callable(getattr(self, "before_execution")):
            if not self.before_execution():
                return []
        else:
            raise AttributeError("The method 'before_execution' is not defined or callable.")
        return func(self, *args, **kwargs)
    return wrapper