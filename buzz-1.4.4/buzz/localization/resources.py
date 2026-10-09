"""Cancellable shared resource gates for concurrent localization jobs."""

from __future__ import annotations

from contextlib import contextmanager
import threading
import time
from typing import Callable, Iterator


class ResourceCancelled(RuntimeError):
    pass


class _FairGate:
    """A cancellable FIFO gate with a configurable number of permits."""

    def __init__(self, capacity: int):
        self._available = capacity
        self._condition = threading.Condition()
        self._waiters: list[object] = []

    def acquire(
        self,
        *,
        cancel_event: threading.Event | None,
        waiting: Callable[[], None] | None,
    ) -> None:
        ticket = object()
        notified = False
        wait_started = time.monotonic()

        with self._condition:
            if self._available and not self._waiters:
                self._available -= 1
                return
            self._waiters.append(ticket)

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._waiters.remove(ticket)
                    self._condition.notify_all()
                    raise ResourceCancelled("Localization was cancelled")
                if self._available and self._waiters[0] is ticket:
                    self._waiters.pop(0)
                    self._available -= 1
                    return

                # Wake promptly on release, and poll cancellation at a bounded
                # 100 ms cadence without ever degenerating into a busy loop.
                self._condition.wait(timeout=0.1)
                if waiting is not None and not notified and time.monotonic() - wait_started >= 0.1:
                    # Match the semaphore path: notify once only after this
                    # acquisition has actually had to wait.
                    try:
                        waiting()
                    except Exception:
                        self._waiters.remove(ticket)
                        self._condition.notify_all()
                        raise
                    notified = True

    def release(self) -> None:
        with self._condition:
            self._available += 1
            self._condition.notify_all()


class LocalizationResourceScheduler:
    """Process-wide gates shared by every localization worker.

    ``prepare``, ``asr`` and ``render`` intentionally name the *same*
    semaphore: all are CPU/RAM intensive on the target machine and must never
    overlap.
    """

    def __init__(
        self, *, prepare: int = 1, asr: int = 1, gemini: int = 2,
        gemini_timing: int = 1, tts: int = 1, render: int = 1,
    ):
        for name, value in {
            "prepare": prepare, "asr": asr, "gemini": gemini,
            "gemini_timing": gemini_timing, "tts": tts, "render": render,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} limit must be a positive integer")
        # Keep the public aliases for the pipeline stages, but do not permit
        # any of them to create an independent CPU-heavy pool.
        if prepare != 1 or asr != 1 or render != 1:
            raise ValueError("the shared cpu-heavy gate has capacity 1")
        if gemini_timing != 1:
            raise ValueError("the Gemini timing gate has capacity 1")
        cpu_heavy = threading.Semaphore(1)
        self._gates = {
            "prepare": cpu_heavy,
            "asr": cpu_heavy,
            "gemini": threading.Semaphore(gemini),
            "gemini_timing": _FairGate(gemini_timing),
            "tts": _FairGate(tts),
            "render": cpu_heavy,
        }

    @contextmanager
    def acquire(
        self,
        resource: str,
        *,
        cancel_event: threading.Event | None = None,
        waiting: Callable[[], None] | None = None,
    ) -> Iterator[None]:
        if resource not in self._gates:
            raise ValueError(f"Unknown localization resource: {resource}")
        if cancel_event is not None and cancel_event.is_set():
            raise ResourceCancelled("Localization was cancelled")
        gate = self._gates[resource]
        if isinstance(gate, _FairGate):
            gate.acquire(cancel_event=cancel_event, waiting=waiting)
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise ResourceCancelled("Localization was cancelled")
                yield
            finally:
                gate.release()
            return

        notified = False
        while not gate.acquire(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise ResourceCancelled("Localization was cancelled")
            if waiting is not None and not notified:
                waiting()
                notified = True
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise ResourceCancelled("Localization was cancelled")
            yield
        finally:
            gate.release()


DEFAULT_RESOURCE_SCHEDULER = LocalizationResourceScheduler()
