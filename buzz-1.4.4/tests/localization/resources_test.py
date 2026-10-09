import threading
import time

import pytest

from buzz.localization.resources import LocalizationResourceScheduler, ResourceCancelled


@pytest.mark.parametrize("resource,limit", [
    ("prepare", 1), ("asr", 1), ("gemini", 2), ("gemini_timing", 1), ("tts", 1), ("render", 1),
])
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


@pytest.mark.parametrize("resource", ["prepare", "tts"])
def test_waiting_notified_once_and_cancellation_is_prompt(resource):
    scheduler = LocalizationResourceScheduler(tts=1)
    cancel_event = threading.Event()
    waiting = []
    result = []
    with scheduler.acquire(resource):
        def blocked():
            try:
                with scheduler.acquire(resource, cancel_event=cancel_event, waiting=lambda: waiting.append(1)):
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


def test_cpu_heavy_gate_prevents_prepare_asr_and_render_overlap():
    scheduler = LocalizationResourceScheduler()
    active = maximum = 0
    lock = threading.Lock()
    start = threading.Barrier(4)

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

    threads = [
        threading.Thread(target=use, args=(resource,))
        for resource in ("prepare", "asr", "render")
    ]
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
        with pytest.raises(ValueError):
            LocalizationResourceScheduler(prepare=value)
    with pytest.raises(ValueError, match="Unknown"):
        with scheduler.acquire("unknown"):
            pass


def test_tts_waiters_are_fifo_when_a_worker_reacquires():
    scheduler = LocalizationResourceScheduler(tts=1)
    queued = {name: threading.Event() for name in "ABC"}
    start = {name: threading.Event() for name in "ABC"}
    order = []
    lock = threading.Lock()

    def use_tts(name):
        start[name].wait()
        for _ in range(2):
            with scheduler.acquire("tts", waiting=queued[name].set):
                with lock:
                    order.append(name)

    with scheduler.acquire("tts"):
        threads = [threading.Thread(target=use_tts, args=(name,)) for name in "ABC"]
        for thread in threads:
            thread.start()
        for name in "ABC":
            start[name].set()
            assert queued[name].wait(1)
    for thread in threads:
        thread.join(1)
        assert not thread.is_alive()

    assert order == list("ABCABC")


def test_gemini_timing_waiters_are_fifo_while_normal_gemini_allows_two():
    scheduler = LocalizationResourceScheduler()
    queued = {name: threading.Event() for name in "AB"}
    order = []

    def use_timing(name):
        with scheduler.acquire("gemini_timing", waiting=queued[name].set):
            order.append(name)

    with scheduler.acquire("gemini_timing"):
        first = threading.Thread(target=use_timing, args=("A",))
        second = threading.Thread(target=use_timing, args=("B",))
        first.start()
        assert queued["A"].wait(1)
        second.start()
        assert queued["B"].wait(1)
    first.join(1)
    second.join(1)
    assert order == ["A", "B"]

    with scheduler.acquire("gemini"), scheduler.acquire("gemini"):
        pass


def test_tts_cancelling_queued_waiter_allows_next_waiter():
    scheduler = LocalizationResourceScheduler(tts=1)
    cancelled = threading.Event()
    queued_b = threading.Event()
    queued_c = threading.Event()
    entered = []

    def use_b():
        with pytest.raises(ResourceCancelled):
            with scheduler.acquire("tts", cancel_event=cancelled, waiting=queued_b.set):
                entered.append("B")

    def use_c():
        with scheduler.acquire("tts", waiting=queued_c.set):
            entered.append("C")

    with scheduler.acquire("tts"):
        b = threading.Thread(target=use_b)
        c = threading.Thread(target=use_c)
        b.start()
        assert queued_b.wait(1)
        c.start()
        assert queued_c.wait(1)
        cancelled.set()
        b.join(1)
        assert not b.is_alive()
    c.join(1)
    assert not c.is_alive()
    assert entered == ["C"]


def test_tts_capacity_two_admits_two_fifo_waiters_after_release():
    scheduler = LocalizationResourceScheduler(tts=2)
    queued = {name: threading.Event() for name in "ABC"}
    start = {name: threading.Event() for name in "ABC"}
    admitted = {name: threading.Event() for name in "ABC"}
    leave = {name: threading.Event() for name in "ABC"}
    release = [threading.Event(), threading.Event()]
    holders_ready = [threading.Event(), threading.Event()]
    entered = []

    def hold(index):
        with scheduler.acquire("tts"):
            holders_ready[index].set()
            release[index].wait()

    def wait_for_tts(name):
        start[name].wait()
        with scheduler.acquire("tts", waiting=queued[name].set):
            entered.append(name)
            admitted[name].set()
            leave[name].wait()

    holders = [threading.Thread(target=hold, args=(index,)) for index in range(2)]
    for holder in holders:
        holder.start()
    for ready in holders_ready:
        assert ready.wait(1)
    waiters = [threading.Thread(target=wait_for_tts, args=(name,)) for name in "ABC"]
    for waiter in waiters:
        waiter.start()
    for name in "ABC":
        start[name].set()
        assert queued[name].wait(1)

    release[0].set()
    assert admitted["A"].wait(1)
    assert entered == ["A"]
    release[1].set()
    assert admitted["B"].wait(1)
    assert entered == list("AB")
    leave["A"].set()
    assert admitted["C"].wait(1)
    assert entered == list("ABC")
    leave["B"].set()
    leave["C"].set()
    for thread in holders + waiters:
        thread.join(1)
        assert not thread.is_alive()
    assert entered == list("ABC")
