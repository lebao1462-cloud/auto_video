import json
import os
from dataclasses import replace

import pytest

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


def test_reconcile_completed_part_requires_audited_source_and_explicit_confirmation(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    part_dir = tmp_path / "part_0001"
    source.write_bytes(b"original-source")
    part_dir.mkdir()
    (part_dir / "source.vi.mp4").write_bytes(b"rendered-video")
    (part_dir / "source.vi.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nxin chao\n", encoding="utf-8",
    )
    monkeypatch.setattr(chunked, "probe_media_duration", lambda _: 2.0)
    source_hash = chunked._file_record(source)["sha256"]

    with pytest.raises(ValueError, match="explicit operator confirmation"):
        chunked.reconcile_completed_part(
            source=source, part_dir=part_dir, fingerprint="f" * 64,
            chunk_seconds=2.0, subtitle_format="srt",
            expected_source_sha256=source_hash, confirmation="no",
        )
    with pytest.raises(ValueError, match="does not match"):
        chunked.reconcile_completed_part(
            source=source, part_dir=part_dir, fingerprint="f" * 64,
            chunk_seconds=2.0, subtitle_format="srt",
            expected_source_sha256="0" * 64,
            confirmation="RECOVER_COMPLETED_PART",
        )

    manifest = chunked.reconcile_completed_part(
        source=source, part_dir=part_dir, fingerprint="f" * 64,
        chunk_seconds=2.0, subtitle_format="srt",
        expected_source_sha256=source_hash, confirmation="RECOVER_COMPLETED_PART",
    )
    assert manifest["recovered"]["source_sha256"] == source_hash
    assert chunked._load_completed_part(
        part_dir / "localization-part-complete.json", "f" * 64,
        part_dir / "source.vi.mp4", part_dir / "source.vi.srt", 2.0,
    ) is not None


def test_rendered_part_growth_budget_allows_verified_long_tts_but_bounds_short_clips():
    from buzz.localization.workflow import LocalizationWorkflowOptions
    from buzz.localization.still_frame import StillFrameExtensionOptions

    options = LocalizationWorkflowOptions(output_directory="unused")
    # Part 0002 needed 31.033s internal plus 2.633s terminal: 33.667s.
    assert chunked._maximum_rendered_part_extension_seconds(900, options) == 135.0
    assert 33.667 < chunked._maximum_rendered_part_extension_seconds(900, options)
    assert chunked._maximum_rendered_part_extension_seconds(180, options) == 9.0
    limited = replace(options, still_frame_extension=StillFrameExtensionOptions(
        max_seconds_total=18.0,
    ))
    assert chunked._maximum_rendered_part_extension_seconds(900, limited) == 18.0


def test_part_fingerprint_excludes_runtime_still_frame_fallback(tmp_path):
    from buzz.localization.still_frame import StillFrameExtensionOptions
    from buzz.localization.workflow import LocalizationWorkflowOptions

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    options = LocalizationWorkflowOptions(output_directory=str(tmp_path))
    common = (source, 3, 2700.0, 900.0, _options(), "model", _TranslationProvider(),
              _TranslationProvider(), options, "whisper")
    baseline = chunked._part_fingerprint(*common)
    narrowed = replace(options, still_frame_extension=StillFrameExtensionOptions(
        max_seconds_total=18.0, max_seconds_per_event=.5,
    ))
    assert chunked._part_fingerprint(*(*common[:-2], narrowed, common[-1])) == baseline
