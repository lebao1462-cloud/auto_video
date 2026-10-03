"""Cancellable shared resource gates for concurrent localization jobs."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Callable, Iterator


class ResourceCancelled(RuntimeError):
    pass


class LocalizationResourceScheduler:
    """Process-wide gates shared by every localization worker.

    ``asr`` and ``render`` intentionally name the *same* semaphore: both are
    CPU/RAM intensive on the target machine and must never overlap.
    """

    def __init__(self, *, asr: int = 1, gemini: int = 2, tts: int = 2, render: int = 1):
        for name, value in {"asr": asr, "gemini": gemini, "tts": tts, "render": render}.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} limit must be a positive integer")
        # Keep the public aliases for the pipeline stages, but do not permit
        # either of them to create an independent CPU-heavy pool.
        if asr != 1 or render != 1:
            raise ValueError("the shared cpu-heavy gate has capacity 1")
        cpu_heavy = threading.Semaphore(1)
        self._gates = {
            "asr": cpu_heavy,
            "gemini": threading.Semaphore(gemini),
            "tts": threading.Semaphore(tts),
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
        gate = self._gates[resource]
        if cancel_event is not None and cancel_event.is_set():
            raise ResourceCancelled("Localization was cancelled")
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
