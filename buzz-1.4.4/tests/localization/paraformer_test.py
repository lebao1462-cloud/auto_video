from pathlib import Path
import os
import sys
import threading
import time
from types import SimpleNamespace
import unicodedata

import pytest

from buzz.localization.paraformer import (
    _MODEL_DIRECTORIES,
    _segments_from_tokens,
    _token_items,
    paraformer_model_is_cached,
    paraformer_model_paths,
    reset_paraformer_model_cache,
)


def _payload():
    tokens = ["这", "是", "一", "个", "wifi", "设", "置", "APP", "的", "演", "示", "测", "试", "请", "看", "完"]
    return {
        "raw_text": " ".join(tokens),
        "text": "这是一个wifi设置APP的演示，测试，请看完。",
        "timestamp": [[index * 300, (index + 1) * 300] for index in range(len(tokens))],
        "sentence_info": [{"text": "must not be used for token alignment"}],
    }


def _words(value: str) -> str:
    return "".join(char for char in value if not char.isspace() and not unicodedata.category(char).startswith("P"))


def _service_test_worker(requests, responses):
    """Module-level target so this exercises Windows spawn semantics."""
    loads = 1  # Represents one model construction for this worker lifetime.
    while True:
        request = requests.get()
        if request is None:
            return
        request_id, value = request
        if value == "block":
            while True:
                time.sleep(0.1)
        if value == "error":
            responses.put((request_id, False, "FunASR test error"))
            continue
        responses.put((request_id, True, {"pid": os.getpid(), "loads": loads}))


def _crashing_worker(requests, responses):
    """Module-level crash target: safe to import in Windows spawned children."""
    request_id, value = requests.get()
    if value == "always-crash":
        os._exit(37)
    marker = Path(value)
    if not marker.exists():
        marker.write_text("crashed", encoding="utf-8")
        os._exit(23)
    # Simulate a late response from a retired request; it must be ignored.
    responses.put((request_id - 1, True, {"stale": True}))
    responses.put((request_id, True, {"pid": os.getpid(), "attempt": 2}))


def test_worker_is_restarted_after_every_success_and_after_cancel():
    from buzz.localization import paraformer

    service = paraformer._ParaformerWorkerService(_service_test_worker)
    try:
        first = service.generate("ok")
        second = service.generate("ok")
        assert first["pid"] != second["pid"]
        assert second["loads"] == 1
        assert service._process is None

        cancelled = threading.Event()
        outcome = []
        def run_blocked():
            try:
                service.generate("block", cancelled)
            except Exception as exc:
                outcome.append(exc)
        thread = threading.Thread(target=run_blocked)
        thread.start()
        time.sleep(0.1)
        started = time.monotonic()
        cancelled.set()
        thread.join(2)
        assert not thread.is_alive()
        assert time.monotonic() - started < 2
        assert len(outcome) == 1
        assert isinstance(outcome[0], paraformer.LocalizationTranscriptionCancelled)
        restarted = service.generate("ok")
        assert restarted["pid"] != first["pid"]
        assert service._process is None
    finally:
        service.shutdown()


def test_worker_crash_retries_once_with_fresh_worker_and_ignores_stale_response(tmp_path: Path):
    from buzz.localization import paraformer

    service = paraformer._ParaformerWorkerService(_crashing_worker)
    marker = tmp_path / "first-attempt-crashed"
    try:
        result = service.generate(str(marker))
        assert result["attempt"] == 2
        assert marker.exists()
        assert service._next_request == 2
        assert service._process is None
    finally:
        service.shutdown()


def test_worker_crash_reports_exit_code_after_one_retry():
    from buzz.localization import paraformer

    service = paraformer._ParaformerWorkerService(_crashing_worker)
    try:
        with pytest.raises(RuntimeError, match="exit code 37"):
            service.generate("always-crash")
        assert service._next_request == 2
        assert service._process is None
    finally:
        service.shutdown()


def test_normal_worker_error_is_not_retried_or_retained():
    from buzz.localization import paraformer

    service = paraformer._ParaformerWorkerService(_service_test_worker)
    try:
        with pytest.raises(RuntimeError, match="FunASR test error"):
            service.generate("error")
        assert service._next_request == 1
        assert service._process is None
    finally:
        service.shutdown()


def test_raw_text_is_authoritative_and_preserves_latin_tokens():
    payload = _payload()
    items = _token_items(payload)
    transcript = "".join(segment.text for segment in _segments_from_tokens(payload))

    assert [item[0] for item in items][4] == "wifi"
    assert [item[0] for item in items][7] == "APP"
    assert _words(transcript) == _words(payload["raw_text"])
    assert "wifi" in transcript and "APP" in transcript
    assert all(items[index][1] >= items[index - 1][2] for index in range(1, len(items)))


def test_punctuation_attaches_to_the_just_consumed_chinese_token():
    payload = {
        "raw_text": "\u4f60 \u597d \u5417 \u8bf7 \u7b54",
        "text": "\u4f60\uff0c\u597d\uff01\u5417\uff1f\u8bf7\u7b54\u3002",
        "timestamp": [[index * 500, (index + 1) * 500] for index in range(5)],
    }

    segments = _segments_from_tokens(payload)

    assert "".join(segment.text for segment in segments) == payload["text"]
    assert all(segment.text for segment in segments)
    assert all(segment.end > segment.start for segment in segments)
    assert all(
        segment.start >= previous.end
        for previous, segment in zip(segments, segments[1:])
    )
    assert segments[-1].end == 2.5


@pytest.mark.parametrize(
    ("raw_text", "text", "punctuated_token"),
    [
        ("\u8bbe \u7f6e wifi \u5b8c \u6210", "\u8bbe\u7f6ewifi\uff0c\u5b8c\u6210\u3002", "wifi\uff0c"),
        ("\u6253 \u5f00 APP \u5427", "\u6253\u5f00APP\uff01\u5427\u3002", "APP\uff01"),
    ],
)
def test_punctuation_follows_whole_latin_tokens(raw_text: str, text: str, punctuated_token: str):
    tokens = raw_text.split()
    payload = {
        "raw_text": raw_text,
        "text": text,
        "timestamp": [[index * 500, (index + 1) * 500] for index in range(len(tokens))],
    }

    transcript = "".join(segment.text for segment in _segments_from_tokens(payload))

    assert transcript == text
    assert punctuated_token in transcript


def test_raw_text_timestamp_count_mismatch_is_rejected():
    payload = _payload()
    payload["timestamp"] = payload["timestamp"][:-1]
    with pytest.raises(ValueError, match="count mismatch"):
        _token_items(payload)


def test_cache_requires_punctuation_model(tmp_path: Path):
    for name in ("model", "vad"):
        (tmp_path / "models" / _MODEL_DIRECTORIES[name] / "snapshots" / "master").mkdir(parents=True)
    assert not paraformer_model_is_cached(tmp_path)
    expected = tmp_path / "models" / _MODEL_DIRECTORIES["punc"] / "snapshots" / "master"
    expected.mkdir(parents=True)
    assert paraformer_model_is_cached(tmp_path)
    assert paraformer_model_paths(tmp_path)["punc"] == str(expected)


def test_frozen_app_prefers_bundled_modelscope_cache(monkeypatch, tmp_path: Path):
    from buzz.localization import paraformer

    app_dir = tmp_path / "AutoVideo"
    bundled = app_dir / "modelscope-cache"
    (bundled / "models").mkdir(parents=True)
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)
    monkeypatch.setattr(paraformer.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paraformer.sys, "executable", str(app_dir / "AutoVideo.exe"))

    assert paraformer.modelscope_cache_path() == bundled


@pytest.mark.skip(reason="The model now runs in a spawned worker; covered by worker lifecycle test.")
def test_shared_model_is_constructed_once_and_generate_is_serialized(monkeypatch, tmp_path: Path):
    from buzz.localization import paraformer

    reset_paraformer_model_cache()
    calls = []
    active = maximum = 0
    lock = threading.Lock()

    class FakeModel:
        def generate(self, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.04)
                return [{"raw_text": "你 好", "text": "你好", "timestamp": [[0, 200], [200, 400]]}]
            finally:
                with lock:
                    active -= 1

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeModel()

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=build))
    monkeypatch.setattr(paraformer, "paraformer_runtime_available", lambda: True)
    monkeypatch.setattr(paraformer, "configure_modelscope_cache", lambda: tmp_path)
    monkeypatch.setattr(paraformer, "extract_audio_track", lambda source, output, **kwargs: Path(output).write_bytes(b"wav"))
    videos = [tmp_path / f"video-{index}.mp4" for index in range(2)]
    for video in videos:
        video.write_bytes(b"video")
    threads = [threading.Thread(target=paraformer.transcribe_with_paraformer, args=(str(video),)) for video in videos]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    assert maximum == 1
    reset_paraformer_model_cache()



def test_worker_timeout_retries_only_once_and_cleans_up():
    """A live-but-silent ASR subprocess must not block the GUI indefinitely."""
    from buzz.localization import paraformer

    service = paraformer._ParaformerWorkerService(
        _service_test_worker, timeout_seconds=2.0,
    )
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="Paraformer ASR had no progress for"):
            service.generate("block")
        # Spawning twice can be slow under Windows antivirus scans.
        assert time.monotonic() - started < 30
        assert service._next_request == 2
        assert service._process is None
        # Windows can take several seconds just to spawn the interpreter.
        service.timeout_seconds = 20.0
        assert service.generate("ok")["loads"] == 1
    finally:
        service.shutdown()


def test_worker_rejects_invalid_hard_timeout():
    from buzz.localization import paraformer
    with pytest.raises(ValueError, match="timeout"):
        paraformer._ParaformerWorkerService(timeout_seconds=0)
