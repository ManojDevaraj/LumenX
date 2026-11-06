"""CPU stage timings for the classical release; no GPU timing claims."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timer(timings: dict[str, float], name: str) -> Iterator[None]:
    """Accumulate wall-clock milliseconds for the named CPU operation."""
    start = perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (perf_counter() - start) * 1000
