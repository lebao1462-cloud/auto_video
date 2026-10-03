import threading
import time

import pytest

from buzz.localization.resources import LocalizationResourceScheduler, ResourceCancelled


@pytest.mark.parametrize("resource,limit", [("asr", 1), ("gemini", 2), ("tts", 2), ("render", 1)])
def test_resource_limits_concurrent_users(resource, limit):
    scheduler = LocalizationResourceScheduler()
    entered = 0
    maximum = 0
    lock = threading.Lock()
    start = threading.Barrier(4)

    def use_gate():
        nonlocal entered, maximum
        start.wait()
        with scheduler.acquire(resource):
            with lock:
                entered += 1
                maximum = max(maximum, entered)
            time.sleep(0.03)
            with lock:
                entered -= 1

    threads = [threading.Thread(target=use_gate) for _ in range(3)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()
    assert maximum == limit


def test_waiting_notified_once_and_cancellation_is_prompt():
    scheduler = LocalizationResourceScheduler(tts=1)
    cancel_event = threading.Event()
    waiting = []
    result = []
    with scheduler.acquire("tts"):
        def blocked():
            try:
                with scheduler.acquire("tts", cancel_event=cancel_event, waiting=lambda: waiting.append(1)):
                    pass
            except ResourceCancelled:
                result.append("cancelled")
        thread = threading.Thread(target=blocked)
        thread.start()
        time.sleep(0.15)
        cancel_event.set()
        thread.join(0.5)
    assert result == ["cancelled"]
    assert waiting == [1]


def test_cpu_heavy_gate_prevents_asr_and_render_overlap():
    scheduler = LocalizationResourceScheduler()
    active = maximum = 0
    lock = threading.Lock()
    start = threading.Barrier(3)

    def use(resource):
        nonlocal active, maximum
        start.wait()
        with scheduler.acquire(resource):
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.04)
            with lock:
                active -= 1

    threads = [threading.Thread(target=use, args=(resource,)) for resource in ("asr", "render")]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_release_after_exception_and_validation():
    scheduler = LocalizationResourceScheduler()
    with pytest.raises(RuntimeError):
        with scheduler.acquire("tts"):
            raise RuntimeError("boom")
    with scheduler.acquire("tts"):
        pass
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            LocalizationResourceScheduler(asr=value)
    with pytest.raises(ValueError, match="Unknown"):
        with scheduler.acquire("unknown"):
            pass
