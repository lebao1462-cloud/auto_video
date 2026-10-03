from pathlib import Path
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
