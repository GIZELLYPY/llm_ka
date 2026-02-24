import time
import random
from typing import Callable, TypeVar

T = TypeVar("T")

def retry_once(fn: Callable[[], T], *, sleep_s: float = 1.0) -> T:
    """
    Retry exactly onde for transient/parsin failures.
    Adds a small jitter to avoid thundering herd when batching.
    """

    try:
        return fn()
    except Exception:
        time.sleep(sleep_s + random.random() * 0.5)
        return fn()