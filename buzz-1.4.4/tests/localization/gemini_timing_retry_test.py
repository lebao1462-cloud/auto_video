import threading
import pytest

from buzz.localization.providers import LocalizationProviderError
from buzz.localization import workflow


def test_transient_504_retries_until_success(monkeypatch):
    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (0.0,))
    attempts = []
    messages = []

    def generate():
        attempts.append(1)
        if len(attempts) < 3:
            raise LocalizationProviderError(
                "Gemini batch translation shortening failed: 504 DEADLINE_EXCEEDED"
            )
        return ["success"]

    assert workflow._retry_gemini_timing_transient_failure(
        generate, waiting_callback=messages.append,
    ) == ["success"]
    assert len(attempts) == 3
    assert len(messages) == 2
    assert all("504" in msg for msg in messages)


def test_dns_failure_retries_until_success_with_network_status(monkeypatch):
    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (0.0,))
    attempts, messages = [], []

    def generate():
        attempts.append(1)
        if len(attempts) == 1:
            raise LocalizationProviderError(
                "Gemini batch translation shortening failed: [Errno 11001] getaddrinfo failed"
            )
        return ["success"]

    assert workflow._retry_gemini_timing_transient_failure(
        generate, waiting_callback=messages.append,
    ) == ["success"]
    assert len(attempts) == 2
    assert len(messages) == 1
    assert "DNS/network" in messages[0]


@pytest.mark.parametrize("message", (
    "Gemini daily quota is exhausted for every configured model",
    "Gemini batch translation shortening failed: 400 INVALID_ARGUMENT",
    "Gemini batch translation shortening returned unusable structured output",
    "Gemini batch translation shortening failed: API_KEY_INVALID",
))
def test_nontransient_gemini_error_fails_fast(monkeypatch, message):
    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (0.0,))
    calls = []
    def generate():
        calls.append(1)
        raise LocalizationProviderError(message)
    with pytest.raises(LocalizationProviderError):
        workflow._retry_gemini_timing_transient_failure(generate)
    assert len(calls) == 1


def test_gemini_retry_backoff_can_be_cancelled(monkeypatch):
    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (300.0,))
    cancel_event = threading.Event()
    calls = []
    def generate():
        calls.append(1)
        raise LocalizationProviderError("Gemini: 504 DEADLINE_EXCEEDED")
    def status(_message):
        cancel_event.set()
    with pytest.raises(workflow.LocalizationCancelled):
        workflow._retry_gemini_timing_transient_failure(
            generate, cancel_event=cancel_event, waiting_callback=status,
        )
    assert len(calls) == 1


def test_gemini_dns_retry_backoff_can_be_cancelled(monkeypatch):
    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (300.0,))
    cancel_event = threading.Event()
    calls = []

    def generate():
        calls.append(1)
        raise LocalizationProviderError("[Errno 11001] getaddrinfo failed")

    with pytest.raises(workflow.LocalizationCancelled):
        workflow._retry_gemini_timing_transient_failure(
            generate, cancel_event=cancel_event,
            waiting_callback=lambda _message: cancel_event.set(),
        )
    assert len(calls) == 1


def test_transient_retry_is_bounded_when_requested(monkeypatch):
    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (0.0,))
    calls = []

    def generate():
        calls.append(1)
        raise LocalizationProviderError("Gemini: 504 DEADLINE_EXCEEDED")

    with pytest.raises(LocalizationProviderError):
        workflow._retry_gemini_timing_transient_failure(generate, max_retries=1)
    assert len(calls) == 2


def test_dns_retry_is_bounded_when_requested(monkeypatch):
    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (0.0,))
    calls = []

    def generate():
        calls.append(1)
        raise LocalizationProviderError("[Errno 11001] getaddrinfo failed")

    with pytest.raises(LocalizationProviderError):
        workflow._retry_gemini_timing_transient_failure(generate, max_retries=1)
    assert len(calls) == 2
