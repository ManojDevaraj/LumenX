"""One bounded worker lane for model, frame, and metric operations."""

from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Iterator


class LaneBusy(RuntimeError):
    """The bounded still-request queue is full."""


_END = object()


class EngineLane:
    """Serialize all engine calls on one worker thread and bridge event streams."""

    def __init__(self, max_pending: int = 2):
        self.max_pending = max(0, int(max_pending))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumenx-engine")
        self._lock = threading.Lock()
        self._outstanding = 0
        self._active = False
        self._closed = False
        self._live_sessions = 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return max(0, self._outstanding - (1 if self._active else 0))

    @property
    def live_sessions(self) -> int:
        with self._lock:
            return self._live_sessions

    def set_live_sessions(self, value: int) -> None:
        with self._lock:
            self._live_sessions = max(0, int(value))

    def _reserve(self, *, live: bool = False) -> bool:
        with self._lock:
            if self._closed or self._outstanding >= self.max_pending + 1:
                return False
            if live and self._outstanding:
                return False
            self._outstanding += 1
            return True

    def _started(self) -> None:
        with self._lock:
            self._active = True

    def _released(self) -> None:
        with self._lock:
            self._active = False
            self._outstanding = max(0, self._outstanding - 1)

    async def warmup(self, function: Callable[[], Any]) -> Any:
        if not self._reserve():
            raise LaneBusy("engine lane is busy")

        def invoke() -> Any:
            self._started()
            try:
                return function()
            finally:
                self._released()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, invoke)

    async def call(self, function: Callable[[], Any], *, live: bool = False) -> Any:
        if not self._reserve(live=live):
            if live:
                return None
            raise LaneBusy("engine lane is full")

        def invoke() -> Any:
            self._started()
            try:
                return function()
            finally:
                self._released()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, invoke)

    async def stream(
        self,
        function: Callable[[], Iterator[Any]],
        cancel: threading.Event,
        *,
        live: bool = False,
        started_callback: Callable[[], None] | None = None,
    ) -> AsyncIterator[Any]:
        if not self._reserve(live=live):
            if live:
                return
            raise LaneBusy("engine lane is full")
        items: queue.Queue[Any] = queue.Queue(maxsize=3)

        def produce() -> None:
            self._started()
            try:
                if started_callback is not None:
                    started_callback()
                for item in function():
                    while not cancel.is_set():
                        try:
                            items.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if cancel.is_set():
                        break
            except BaseException as exc:
                while not cancel.is_set():
                    try:
                        items.put(exc, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            finally:
                while not cancel.is_set():
                    try:
                        items.put(_END, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                self._released()

        self._executor.submit(produce)
        try:
            while True:
                try:
                    item = items.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                if item is _END:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            cancel.set()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

