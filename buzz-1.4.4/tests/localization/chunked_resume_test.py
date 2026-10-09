import json
import os

from buzz.localization import chunked
from buzz.localization.transcript import LocalizationSegment, LocalizationTranscript
from buzz.localization.translation import _checkpoint_fingerprint
from buzz.transcriber.transcriber import Task, TranscriptionOptions


class _TranslationProvider:
    model = "test"


def _options(language="en"):
    return TranscriptionOptions(language=language, task=Task.TRANSCRIBE)


def test_prepared_clip_reuse_rejects_newer_source_and_bad_duration(tmp_path, monkeypatch):
    source, clip = tmp_path / "source.mp4", tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    clip.write_bytes(b"clip")
    monkeypatch.setattr(chunked, "probe_media_duration", lambda _: 2.0)
    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    os.utime(clip, ns=(2_000_000_000, 2_000_000_000))
    assert chunked._reusable_prepared_clip(clip, source, 2.0)
    os.utime(source, ns=(3_000_000_000, 3_000_000_000))
    assert not chunked._reusable_prepared_clip(clip, source, 2.0)
    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(chunked, "probe_media_duration", lambda _: 3.0)
    assert not chunked._reusable_prepared_clip(clip, source, 2.0)


def test_asr_checkpoint_requires_matching_asr_settings(tmp_path):
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"clip")
    transcript = LocalizationTranscript("en", str(clip), (LocalizationSegment(0, 1, "hello"),))
    checkpoint = tmp_path / "localization-asr-checkpoint.json"
    fingerprint = chunked._asr_fingerprint(clip, _options(), "model-a", "whisper")
    chunked._write_asr_checkpoint(checkpoint, fingerprint, transcript)
    assert chunked._transcript_from_checkpoint(checkpoint, fingerprint, clip) == transcript
    changed = chunked._asr_fingerprint(clip, _options("zh"), "model-a", "whisper")
    assert chunked._transcript_from_checkpoint(checkpoint, changed, clip) is None


def test_legacy_checkpoints_reconstruct_only_when_translation_fingerprint_matches(tmp_path):
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"clip")
    workspace = tmp_path / "source_localization_work"
    workspace.mkdir()
    provider = _TranslationProvider()
    candidate = LocalizationTranscript("en", str(clip), (LocalizationSegment(0.0, 1.0, "hello"),))
    segment_id = "segment-000000-anything"
    (workspace / "translation-checkpoint.json").write_text(json.dumps({
        "fingerprint": _checkpoint_fingerprint(candidate, provider, "vi"),
        "segments": {segment_id: {"source_text": "hello", "translated_text": "xin chao"}},
    }), encoding="utf-8")
    (workspace / "timing-recovery-checkpoint.json").write_text(json.dumps({
        "segment_count": 1, "segment_bounds": [[0, 1]], "shortened_text": {"0": "xin chao"},
    }), encoding="utf-8")
    assert chunked._legacy_transcript_from_checkpoints(
        tmp_path, clip, _options(), "whisper", provider,
    ) == candidate
    data = json.loads((workspace / "translation-checkpoint.json").read_text(encoding="utf-8"))
    data["fingerprint"] = "wrong"
    (workspace / "translation-checkpoint.json").write_text(json.dumps(data), encoding="utf-8")
    assert chunked._legacy_transcript_from_checkpoints(
        tmp_path, clip, _options(), "whisper", provider,
    ) is None


def test_legacy_checkpoint_rejects_invalid_timing_entries(tmp_path):
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"clip")
    candidate = LocalizationTranscript("en", str(clip), (LocalizationSegment(0.0, 1.0, "hello"),))
    workspace = tmp_path / "source_localization_work"
    workspace.mkdir()
    provider = _TranslationProvider()
    (workspace / "translation-checkpoint.json").write_text(json.dumps({
        "fingerprint": _checkpoint_fingerprint(candidate, provider, "vi"),
        "segments": {"segment-000000-anything": {"source_text": "hello"}},
    }), encoding="utf-8")
    (workspace / "timing-recovery-checkpoint.json").write_text(json.dumps({
        "segment_count": 1, "segment_bounds": [[0, 1]], "shortened_text": {"0": ""},
    }), encoding="utf-8")
    assert chunked._legacy_transcript_from_checkpoints(
        tmp_path, clip, _options(), "whisper", provider,
    ) is None


def test_gemini_transport_timeout_keeps_legacy_chunk_identity():
    """Raising request timeout must not invalidate already-rendered chunks."""
    from buzz.localization.chunked import _provider_identity
    from buzz.localization.providers import GeminiTranslationProvider

    previous = GeminiTranslationProvider(api_key="offline-test-key", timeout=60.0)
    updated = GeminiTranslationProvider(api_key="offline-test-key", timeout=120.0)
    custom = GeminiTranslationProvider(api_key="offline-test-key", timeout=180.0)
    assert _provider_identity(previous) == _provider_identity(updated)
    assert _provider_identity(updated) == _provider_identity(custom)
    assert _provider_identity(updated)["settings"]["timeout"] == 60.0
