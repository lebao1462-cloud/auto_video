from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
import subprocess
import threading
import time
import wave

import pytest

from buzz.localization.audio_mix import probe_media_duration
from buzz.localization.providers import EdgeTTSProvider
from buzz.localization.resources import LocalizationResourceScheduler
from buzz.localization.transcript import (
    LocalizationSegment, LocalizationTranscript, LocalizationTranscriptionCancelled,
)
from buzz.localization.translation import (
    TranslatedLocalizationSegment,
    TranslatedLocalizationTranscript,
    TranslationProvider,
    _segment_id,
)
from buzz.localization.tts import TTSRequest, TTSResult, TTSServiceTemporarilyUnavailable
from buzz.localization.tts import (
    SynthesizedLocalizationSegment,
    SynthesizedLocalizationTranscript,
    synthesize_segment,
)
from buzz.localization.timing import TimingPolicy, TimingSynchronizationError
from buzz.localization.still_frame import StillFrameExtensionOptions
from buzz.localization.workflow import (
    LocalizationCancelled,
    LocalizationStage,
    LocalizationWorkflowOptions,
    cleanup_localization_workspace,
    localize_video,
    _coalesce_trailing_tiny_translation,
    _load_timing_recovery_checkpoint,
    _persist_timing_recovery_checkpoint,
    _synchronize_with_translation_recovery,
    _timing_recovery_fingerprint,
    TIMING_RECOVERY_TEMPORARY_TTS_RETRIES,
    TIMING_RECOVERY_BATCH_SIZE,
)
from buzz.localization.final_render import FinalRenderOptions
from buzz.transcriber.transcriber import Task, TranscriptionOptions


class FakeTranslationProvider:
    def translate(self, text, source_language, target_language):
        assert source_language in {"en", "zh"}
        assert target_language == "vi"
        return "Xin chào mọi người"


class FakeTTSProvider:
    def synthesize(self, request: TTSRequest):
        output = Path(request.output_file_stem + ".wav")
        output.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 16000
        duration = 1.0
        with wave.open(str(output), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * int(sample_rate * duration))
        return TTSResult(
            audio_file=str(output),
            audio_duration=duration,
            provider="fake",
            voice=request.voice or "vi-test",
        )


def _translated_for_boundary(source_language="zh", *, final_end=10.0):
    return TranslatedLocalizationTranscript(
        source_language=source_language, target_language="vi", source_file="source.mp4",
        segments=(
            TranslatedLocalizationSegment(7.0, 9.2, "previous", "truoc"),
            TranslatedLocalizationSegment(9.3, final_end, "final", "sau"),
        ),
    )


def test_coalesces_tiny_trailing_chinese_translation():
    transcript = _translated_for_boundary()

    result = _coalesce_trailing_tiny_translation(transcript, 10.15)

    assert len(result.segments) == 1
    assert result.segments[0].start == 7.0
    assert result.segments[0].end == 10.0
    assert result.segments[0].source_text == "previousfinal"
    assert result.segments[0].translated_text == "truoc sau"


def test_coalesces_tiny_trailing_english_translation_with_space():
    transcript = _translated_for_boundary("en")

    result = _coalesce_trailing_tiny_translation(transcript, 10.15)

    assert result.segments[0].source_text == "previous final"
    assert result.segments[0].translated_text == "truoc sau"


@pytest.mark.parametrize("final_end, media_end", [
    (10.31, 10.4),  # final duration is not tiny
    (10.0, 10.26),  # too much media tail
])
def test_does_not_coalesce_trailing_translation_outside_boundary_limits(final_end, media_end):
    transcript = _translated_for_boundary(final_end=final_end)

    result = _coalesce_trailing_tiny_translation(transcript, media_end)

    assert result is transcript


def test_does_not_coalesce_when_gap_or_combined_span_is_too_large():
    transcript = _translated_for_boundary()
    excessive_gap = replace(
        transcript, segments=(replace(transcript.segments[0], end=8.8), transcript.segments[1])
    )
    excessive_span = replace(
        transcript, segments=(replace(transcript.segments[0], start=-0.1), transcript.segments[1])
    )

    assert _coalesce_trailing_tiny_translation(excessive_gap, 10.15) is excessive_gap
    assert _coalesce_trailing_tiny_translation(excessive_span, 10.15) is excessive_span


def _synthesized_for_recovery(tmp_path, durations=(1.0, 2.083)):
    segments = []
    for index, duration in enumerate(durations):
        audio = tmp_path / f"segment-{index:06d}.wav"
        audio.write_bytes(b"audio")
        segments.append(SynthesizedLocalizationSegment(
            start=float(index), end=float(index + 1),
            source_text=f"source {index}", translated_text=f"translation {index}",
            audio_file=str(audio), audio_duration=duration,
            provider="fake", voice="vi-test",
        ))
    return SynthesizedLocalizationTranscript(
        source_language="en", target_language="vi", source_file="source.mp4",
        segments=tuple(segments),
    )


class ShorteningProvider:
    def __init__(self):
        self.calls = []

    def shorten_translation(self, **kwargs):
        self.calls.append(kwargs)
        return "bản dịch ngắn"


class RepeatedShorteningProvider(ShorteningProvider):
    """Avoid TTS cache hits while exercising bounded recovery attempts."""

    def shorten_translation(self, **kwargs):
        self.calls.append(kwargs)
        return f"shortened translation {len(self.calls)}"


class RecoveryTTSProvider:
    def __init__(self, durations):
        self.durations = iter(durations)
        self.requests = []

    def synthesize(self, request):
        self.requests.append(request)
        output = Path(request.output_file_stem + ".wav")
        output.write_bytes(b"replacement")
        return TTSResult(
            audio_file=str(output), audio_duration=next(self.durations),
            provider="fake", voice=request.voice,
        )

    def synthesize_batch(self, _requests):
        raise AssertionError("timing recovery must synthesize one segment at a time")


class RecoveryEdgeTTSProvider(EdgeTTSProvider):
    """Offline Edge stand-in for timing-recovery rate tests."""

    def __init__(self, durations):
        super().__init__(retry_delays=(0,))
        self.batch_size = 1
        self.durations = iter(durations)
        self.requests = []

    def synthesize(self, request):
        self.requests.append(request)
        output = Path(request.output_file_stem + ".wav")
        output.write_bytes(b"replacement")
        return TTSResult(
            audio_file=str(output), audio_duration=next(self.durations),
            provider="edge", voice=request.voice,
        )


class ConcurrentRecoveryTTSProvider:
    max_concurrency = 3

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.requests = []

    def synthesize(self, request):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.requests.append(request)
        try:
            time.sleep(0.1)
            output = Path(request.output_file_stem + ".wav")
            output.write_bytes(request.text.encode())
            return TTSResult(
                audio_file=str(output), audio_duration=1.05, provider="fake",
                voice=request.voice, metadata={"text": request.text},
            )
        finally:
            with self._lock:
                self.active -= 1


def test_timing_recovery_serializes_tts_and_applies_by_segment_index(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    tts = ConcurrentRecoveryTTSProvider()
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0, 1.0)),
        BatchProvider(), tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert tts.max_active == 1
    assert [segment.translated_text for segment in timed.segments] == [
        "short-segment-000000", "short-segment-000001", "translation 2",
    ]
    assert [segment.provider_metadata for segment in timed.segments[:2]] == [
        {"text": "short-segment-000000"}, {"text": "short-segment-000001"},
    ]


def test_timing_recovery_tts_is_serial_without_declared_max_concurrency(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    class SerialRecoveryTTSProvider:
        __init__ = ConcurrentRecoveryTTSProvider.__init__
        synthesize = ConcurrentRecoveryTTSProvider.synthesize

    tts = SerialRecoveryTTSProvider()
    _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0, 1.0)),
        BatchProvider(), tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert tts.max_active == 1


def test_timing_recovery_stops_before_later_tts_after_exception(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return ["first", "second"]

    class FailingTTS:
        def __init__(self):
            self.requests = []

        def synthesize(self, request):
            self.requests.append(request)
            raise RuntimeError("Edge unavailable")

    tts = FailingTTS()
    with pytest.raises(RuntimeError, match="Edge unavailable"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0)), BatchProvider(), tts,
            LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
        )
    assert [request.text for request in tts.requests] == ["first"]


def test_timing_recovery_temporary_edge_outage_releases_resource_and_retries_batch(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    class FlakyBatchTTS:
        batch_size = 4

        def __init__(self):
            self.calls = []

        def synthesize_batch(self, requests):
            self.calls.append([request.text for request in requests])
            if len(self.calls) == 1:
                raise TTSServiceTemporarilyUnavailable(7)
            results = []
            for request in requests:
                output = Path(request.output_file_stem + ".wav")
                output.write_bytes(b"replacement")
                results.append(TTSResult(str(output), 1.05, "fake", request.voice))
            return results

    class ImmediateEvent:
        def wait(self, _delay):
            return False

        def is_set(self):
            return False

    class RecordingScheduler:
        def __init__(self):
            self.events = []

        @contextmanager
        def acquire(self, resource, **_kwargs):
            self.events.append(("enter", resource))
            try:
                yield
            finally:
                self.events.append(("exit", resource))

    waiting = []
    scheduler = RecordingScheduler()
    provider = FlakyBatchTTS()
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0)), BatchProvider(), provider,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), resource_scheduler=scheduler),
        cancel_event=ImmediateEvent(), waiting_callback=waiting.append,
    )

    assert provider.calls == [["short-segment-000000", "short-segment-000001"]] * 2
    assert scheduler.events == [("enter", "gemini_timing"), ("exit", "gemini_timing"),
                                ("enter", "tts"), ("exit", "tts"),
                                ("enter", "tts"), ("exit", "tts")]
    assert any("Gemini processed 2/2" in message for message in waiting)
    assert waiting[-1] == "Edge TTS tạm thời không khả dụng, tự thử lại sau 7 giây"
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_timing_recovery_temporary_edge_outage_honors_cancellation(tmp_path):
    class Shortener:
        def shorten_translation(self, **_kwargs):
            return "short"

    class UnavailableTTS:
        def synthesize(self, _request):
            raise TTSServiceTemporarilyUnavailable(60)

    class CancelledEvent:
        def is_set(self):
            return False

        def wait(self, _delay):
            return True

    with pytest.raises(LocalizationCancelled):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(2.0,)), Shortener(), UnavailableTTS(),
            LocalizationWorkflowOptions(output_directory=str(tmp_path)),
            cancel_event=CancelledEvent(),
        )


def test_timing_recovery_stops_after_persistent_temporary_outage_for_segment(tmp_path):
    class Shortener:
        def shorten_translation(self, **_kwargs):
            return "short"

    class UnavailableTTS:
        def __init__(self):
            self.calls = 0

        def synthesize(self, _request):
            self.calls += 1
            raise TTSServiceTemporarilyUnavailable(0)

    tts = UnavailableTTS()
    with pytest.raises(RuntimeError, match="Edge TTS remained unavailable"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(2.0,)), Shortener(), tts,
            LocalizationWorkflowOptions(output_directory=str(tmp_path)),
        )

    assert tts.calls == TIMING_RECOVERY_TEMPORARY_TTS_RETRIES + 1


def test_timing_recovery_stops_after_persistent_temporary_outage_for_batch(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    class UnavailableBatchTTS:
        batch_size = 2

        def __init__(self):
            self.calls = 0

        def synthesize_batch(self, _requests):
            self.calls += 1
            raise TTSServiceTemporarilyUnavailable(0)

    tts = UnavailableBatchTTS()
    with pytest.raises(RuntimeError, match="Edge TTS remained unavailable"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0)), BatchProvider(), tts,
            LocalizationWorkflowOptions(output_directory=str(tmp_path)),
        )

    assert tts.calls == TIMING_RECOVERY_TEMPORARY_TTS_RETRIES + 1


def test_timing_recovery_checkpoint_saves_each_success_before_later_failure(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return ["first", "second"]

    class FailsSecond(RecoveryTTSProvider):
        def synthesize(self, request):
            if request.text == "second":
                self.requests.append(request)
                raise RuntimeError("second failed")
            return super().synthesize(request)

    path = tmp_path / "timing-recovery-checkpoint.json"
    with pytest.raises(RuntimeError, match="second failed"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0)), BatchProvider(),
            FailsSecond([1.0]), LocalizationWorkflowOptions(output_directory=str(tmp_path)),
            checkpoint_path=path, checkpoint_fingerprint="fingerprint",
        )
    assert __import__("json").loads(path.read_text(encoding="utf-8"))["shortened_text"] == {"0": "first"}


def test_timing_recovery_checkpoint_applies_saved_text_before_tts_and_reuses_cache(tmp_path):
    translated = TranslatedLocalizationTranscript(
        source_language="en", target_language="vi", source_file="source.mp4",
        segments=(TranslatedLocalizationSegment(0, 1, "source", "original"),),
    )
    options = LocalizationWorkflowOptions(output_directory=str(tmp_path))
    fingerprint = _timing_recovery_fingerprint(translated, options, 1.0)
    path = tmp_path / "timing-recovery-checkpoint.json"
    _persist_timing_recovery_checkpoint(path, fingerprint, translated.segments, 0, "shortened")
    restored = _load_timing_recovery_checkpoint(path, fingerprint, translated)
    provider = RecoveryTTSProvider([1.0])
    from buzz.localization.tts import synthesize_for_localization
    synthesize_for_localization(restored, provider, tmp_path / "tts")
    synthesize_for_localization(restored, provider, tmp_path / "tts")
    assert restored.segments[0].translated_text == "shortened"
    assert [request.text for request in provider.requests] == ["shortened"]


def test_timing_recovery_fingerprint_includes_source_file(tmp_path):
    options = LocalizationWorkflowOptions(output_directory=str(tmp_path))
    first = TranslatedLocalizationTranscript("en", "vi", "first.mp4", (
        TranslatedLocalizationSegment(0, 1, "source", "translation"),
    ))
    second = replace(first, source_file="second.mp4")

    assert _timing_recovery_fingerprint(first, options, 1.0) != _timing_recovery_fingerprint(
        second, options, 1.0,
    )


@pytest.mark.parametrize("contents", ["not json", '{"fingerprint":"wrong","segment_count":1,"segment_bounds":[[0,1]],"shortened_text":{"0":"short"}}',
    '{"fingerprint":"match","segment_count":1,"segment_bounds":[[0,1]],"shortened_text":{"9":"short"}}',
    '{"fingerprint":"match","segment_count":1,"segment_bounds":[[0,1]],"shortened_text":{"0":"!!!"}}'])
def test_timing_recovery_ignores_invalid_checkpoint_safely(tmp_path, contents):
    translated = TranslatedLocalizationTranscript("en", "vi", "source.mp4", (
        TranslatedLocalizationSegment(0, 1, "source", "original"),
    ))
    path = tmp_path / "timing-recovery-checkpoint.json"
    path.write_text(contents, encoding="utf-8")
    assert _load_timing_recovery_checkpoint(path, "match", translated) is translated


def test_timing_recovery_checkpoint_replaces_later_shortening(tmp_path):
    path = tmp_path / "timing-recovery-checkpoint.json"
    segments = _synthesized_for_recovery(tmp_path, durations=(1.0, 1.0)).segments
    _persist_timing_recovery_checkpoint(path, "fingerprint", segments, 1, "first short")
    _persist_timing_recovery_checkpoint(path, "fingerprint", segments, 1, "shorter")
    assert __import__("json").loads(path.read_text(encoding="utf-8"))["shortened_text"] == {"1": "shorter"}


def test_timing_recovery_batches_current_failures_and_retries_only_remaining(tmp_path):
    class BatchProvider:
        def __init__(self):
            self.calls = []

        def shorten_translations(self, **kwargs):
            self.calls.append(kwargs["segments"])
            return [f"ngan {item['id']} lan {len(self.calls)}"
                    for item in kwargs["segments"]]

    provider = BatchProvider()
    tts = RecoveryTTSProvider([1.05, 2.0, 1.05])
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0, 1.0)), provider, tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert [[item["id"] for item in batch] for batch in provider.calls] == [
        ["segment-000000", "segment-000001"], ["segment-000001"],
    ]
    assert [request.output_file_stem.rsplit("-", 1)[-1] for request in tts.requests] == [
        "000000", "000001", "000001",
    ]
    assert timed.segments[2].translated_text == "translation 2"
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_timing_recovery_splits_overflows_into_small_batches_in_segment_order(tmp_path):
    class BatchProvider:
        def __init__(self):
            self.calls = []

        def shorten_translations(self, **kwargs):
            self.calls.append(kwargs["segments"])
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    provider = BatchProvider()
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0,) * 65),
        provider,
        RecoveryTTSProvider([1.05] * 65),
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert [[item["id"] for item in batch] for batch in provider.calls] == [
        [f"segment-{index:06d}" for index in range(0, 12)],
        [f"segment-{index:06d}" for index in range(12, 24)],
        [f"segment-{index:06d}" for index in range(24, 36)],
        [f"segment-{index:06d}" for index in range(36, 48)],
        [f"segment-{index:06d}" for index in range(48, 60)],
        [f"segment-{index:06d}" for index in range(60, 65)],
    ]
    assert [segment.translated_text for segment in timed.segments] == [
        f"short-segment-{index:06d}" for index in range(65)
    ]


def test_timing_recovery_structured_output_failure_splits_only_failed_batch(tmp_path):
    from buzz.localization.providers import LocalizationProviderError

    class StructuredFailureProvider:
        def __init__(self):
            self.calls = []

        def shorten_translations(self, **kwargs):
            ids = [item["id"] for item in kwargs["segments"]]
            self.calls.append(ids)
            if len(ids) == 12 and ids[0] == "segment-000012":
                raise LocalizationProviderError("Gemini returned unusable structured output")
            return [f"short-{identifier}" for identifier in ids]

    provider = StructuredFailureProvider()
    _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0,) * 60), provider,
        RecoveryTTSProvider([1.05] * 60),
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert provider.calls == [
        [f"segment-{index:06d}" for index in range(0, 12)],
        [f"segment-{index:06d}" for index in range(12, 24)],
        [f"segment-{index:06d}" for index in range(12, 18)],
        [f"segment-{index:06d}" for index in range(18, 24)],
        [f"segment-{index:06d}" for index in range(24, 36)],
        [f"segment-{index:06d}" for index in range(36, 48)],
        [f"segment-{index:06d}" for index in range(48, 60)],
    ]


@pytest.mark.parametrize("message", [
    "503 UNAVAILABLE", "Server disconnected without sending a response",
])
def test_timing_recovery_transport_errors_do_not_split_batches(tmp_path, message, monkeypatch):
    from buzz.localization.providers import LocalizationProviderError
    from buzz.localization import workflow

    monkeypatch.setattr(workflow, "GEMINI_TIMING_TRANSIENT_RETRY_DELAYS", (0.0,))

    class TransportFailureProvider:
        def __init__(self):
            self.calls = []

        def shorten_translations(self, **kwargs):
            self.calls.append(kwargs["segments"])
            if len(self.calls) == 1:
                raise LocalizationProviderError(message)
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    provider = TransportFailureProvider()
    _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0,) * 30), provider,
        RecoveryTTSProvider([1.05] * 30),
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )
    assert [len(batch) for batch in provider.calls] == [12, 12, 12, 6]

def test_timing_recovery_batch_size_is_small_and_bounded():
    assert TIMING_RECOVERY_BATCH_SIZE == 12


def test_timing_recovery_batches_explicit_tts_provider_and_preserves_order(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    class ExplicitBatchTTSProvider:
        batch_size = 4

        def __init__(self):
            self.batch_calls = []
            self.single_calls = []

        def synthesize(self, request):
            self.single_calls.append(request)
            raise AssertionError("explicit recovery provider must use batch synthesis")

        def synthesize_batch(self, requests):
            self.batch_calls.append(list(requests))
            results = []
            for request in requests:
                output = Path(request.output_file_stem + ".wav")
                output.write_bytes(request.text.encode())
                results.append(TTSResult(
                    audio_file=str(output), audio_duration=1.05,
                    provider="batch-fake", voice=request.voice,
                    metadata={"text": request.text},
                ))
            return results

    tts = ExplicitBatchTTSProvider()
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0,) * 10),
        BatchProvider(), tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert [len(batch) for batch in tts.batch_calls] == [4, 4, 2]
    assert not tts.single_calls
    assert [request.text for batch in tts.batch_calls for request in batch] == [
        f"short-segment-{index:06d}" for index in range(10)
    ]
    assert [segment.translated_text for segment in timed.segments] == [
        f"short-segment-{index:06d}" for index in range(10)
    ]
    assert [Path(segment.audio_file).stem for segment in timed.segments] == [
        f"segment-{index:06d}" for index in range(10)
    ]
    assert [segment.provider_metadata for segment in timed.segments] == [
        {"text": f"short-segment-{index:06d}"} for index in range(10)
    ]
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_timing_recovery_retries_after_timing_proportional_shortening_is_still_long(tmp_path):
    class BatchProvider:
        def __init__(self):
            self.calls = []

        def shorten_translations(self, **kwargs):
            self.calls.append(kwargs["segments"])
            return [["lan dau ngan"], ["lan hai ngan"]][len(self.calls) - 1]

    available_duration = 9.734958 / 1.193
    audio = tmp_path / "segment-000000.wav"
    audio.write_bytes(b"audio")
    original = " ".join(f"tu{index}" for index in range(35))
    synthesized = SynthesizedLocalizationTranscript(
        source_language="en", target_language="vi", source_file="source.mp4",
        segments=(SynthesizedLocalizationSegment(
            start=0.0, end=available_duration, source_text="source",
            translated_text=original, audio_file=str(audio), audio_duration=9.734958,
            provider="fake", voice="vi-test",
        ),),
    )
    provider = BatchProvider()
    tts = RecoveryTTSProvider([9.2, 8.8])

    timed = _synchronize_with_translation_recovery(
        synthesized, provider, tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert provider.calls[0][0]["required_rate"] == pytest.approx(1.193)
    assert [batch[0]["translated_text"] for batch in provider.calls] == [
        original, "lan dau ngan",
    ]
    assert [request.text for request in tts.requests] == ["lan dau ngan", "lan hai ngan"]
    assert timed.segments[0].playback_rate < 1.10


@pytest.mark.parametrize(
    ("required_rate", "expected_rate"),
    [(1.13, "+4%"), (1.16, "+7%"), (1.202, "+10%")],
)
def test_timing_recovery_final_edge_rate_recovers_mild_residual_overflow(
    tmp_path, required_rate, expected_rate,
):
    translator = RepeatedShorteningProvider()
    attempts = 4
    tts = RecoveryEdgeTTSProvider([required_rate] * attempts + [1.08])

    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(required_rate,)), translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            timing_recovery_attempts=attempts,
        ),
    )

    assert len(translator.calls) == attempts
    assert len(tts.requests) == attempts + 1
    assert [request.options for request in tts.requests[:-1]] == [{}] * attempts
    assert tts.requests[-1].options["rate"] == expected_rate
    assert timed.segments[0].playback_rate <= 1.10


def test_timing_recovery_final_edge_rate_escalates_to_ten_percent_when_first_retry_misses(
    tmp_path,
):
    translator = RepeatedShorteningProvider()
    attempts = 4
    tts = RecoveryEdgeTTSProvider([1.148] * attempts + [1.16, 1.08])

    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(1.148,)), translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            timing_recovery_attempts=attempts,
        ),
    )

    assert len(translator.calls) == attempts
    assert len(tts.requests) == attempts + 2
    assert tts.requests[-2].options["rate"] == "+6%"
    assert tts.requests[-1].options["rate"] == "+10%"
    assert timed.segments[0].playback_rate <= 1.10


def test_timing_recovery_daily_gemini_quota_uses_bounded_edge_fallback(tmp_path):
    from buzz.localization.providers import LocalizationProviderError

    class QuotaBatchProvider:
        def __init__(self):
            self.calls = 0

        def shorten_translations(self, **kwargs):
            self.calls += 1
            raise LocalizationProviderError(
                "Gemini daily quota is exhausted for every configured model."
            )

    translator = QuotaBatchProvider()
    tts = RecoveryEdgeTTSProvider([1.08])

    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(1.148,)), translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            timing_recovery_attempts=4,
        ),
    )

    assert translator.calls == 1
    assert len(tts.requests) == 1
    assert tts.requests[0].options["rate"] == "+6%"
    assert timed.segments[0].playback_rate <= 1.10


def test_timing_recovery_large_edge_residual_fails_normally(tmp_path):
    translator = RepeatedShorteningProvider()
    attempts = 4
    tts = RecoveryEdgeTTSProvider([1.25] * attempts)

    with pytest.raises(TimingSynchronizationError, match="after 4 concise-translation retries"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(1.25,)), translator, tts,
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path), timing_policy=TimingPolicy(),
                timing_recovery_attempts=attempts,
                still_frame_extension=StillFrameExtensionOptions(enabled=False),
            ),
        )

    assert len(translator.calls) == len(tts.requests) == attempts
    assert all(request.options == {} for request in tts.requests)


def test_timing_recovery_custom_edge_rate_is_not_overridden_after_semantic_retries(tmp_path):
    translator = RepeatedShorteningProvider()
    attempts = 4
    tts = RecoveryEdgeTTSProvider([1.16] * attempts)

    with pytest.raises(TimingSynchronizationError, match="after 4 concise-translation retries"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(1.16,)), translator, tts,
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path), timing_policy=TimingPolicy(),
                timing_recovery_attempts=attempts, tts_options={"rate": "+2%"},
                still_frame_extension=StillFrameExtensionOptions(enabled=False),
            ),
        )

    assert len(translator.calls) == len(tts.requests) == attempts
    assert all(request.options == {"rate": "+2%"} for request in tts.requests)


@pytest.mark.parametrize(
    ("residual_rate", "durations", "expected_final_rate"),
    [
        (1.13, [1.30, 1.30, 1.30, 1.13, 1.09], "+4%"),
        (1.16, [1.30, 1.30, 1.30, 1.16, 1.09], "+7%"),
    ],
)
def test_timing_recovery_final_edge_rate_retry_recovers_after_concise_retries(
    tmp_path, residual_rate, durations, expected_final_rate,
):
    class RepeatedShorteningProvider(ShorteningProvider):
        def shorten_translation(self, **kwargs):
            self.calls.append(kwargs)
            return f"short {len(self.calls)}"

    translator = RepeatedShorteningProvider()
    tts = RecoveryEdgeTTSProvider(durations)
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(1.30,)), translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            timing_recovery_attempts=4,
        ),
    )

    assert residual_rate > 1.10
    assert len(translator.calls) == 4
    assert tts.requests[-1].text == "short 4"
    assert tts.requests[-1].options["rate"] == expected_final_rate
    assert timed.segments[0].playback_rate <= 1.10


def test_timing_recovery_final_edge_rate_retry_rejects_large_overflow(tmp_path):
    class RepeatedShorteningProvider(ShorteningProvider):
        def shorten_translation(self, **kwargs):
            self.calls.append(kwargs)
            return f"short {len(self.calls)}"

    translator = RepeatedShorteningProvider()
    tts = RecoveryEdgeTTSProvider([1.30, 1.30, 1.30, 1.25])
    with pytest.raises(TimingSynchronizationError, match="after 4 concise-translation retries"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(1.30,)), translator, tts,
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path), timing_policy=TimingPolicy(),
                timing_recovery_attempts=4,
                still_frame_extension=StillFrameExtensionOptions(enabled=False),
            ),
        )

    assert len(translator.calls) == 4
    assert len(tts.requests) == 4


def test_timing_recovery_final_edge_rate_retry_preserves_custom_rate(tmp_path):
    class RepeatedShorteningProvider(ShorteningProvider):
        def shorten_translation(self, **kwargs):
            self.calls.append(kwargs)
            return f"short {len(self.calls)}"

    translator = RepeatedShorteningProvider()
    tts = RecoveryEdgeTTSProvider([1.30, 1.30, 1.30, 1.16])
    with pytest.raises(TimingSynchronizationError, match="after 4 concise-translation retries"):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path, durations=(1.30,)), translator, tts,
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path), timing_policy=TimingPolicy(),
                timing_recovery_attempts=4, tts_options={"rate": "+2%"},
                still_frame_extension=StillFrameExtensionOptions(enabled=False),
            ),
        )

    assert len(translator.calls) == 4
    assert [request.options for request in tts.requests] == [{"rate": "+2%"}] * 4


def test_timing_recovery_shortens_and_regenerates_only_failing_segment(tmp_path):
    original = _synthesized_for_recovery(tmp_path)
    translator = ShorteningProvider()
    tts = RecoveryTTSProvider([1.05])
    seed_provider = RecoveryTTSProvider([1.0])
    synthesize_segment(TTSRequest(
        text=original.segments[0].translated_text,
        language="vi",
        output_file_stem=str(tmp_path / "segment-000000"),
        voice="vi-test",
    ), seed_provider)
    unaffected_checkpoint = tmp_path / "segment-000000.localization-tts.json"
    unaffected_contents = unaffected_checkpoint.read_bytes()

    timed = _synchronize_with_translation_recovery(
        original, translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path),
            timing_policy=TimingPolicy(),
        ),
    )

    assert len(translator.calls) == 1
    assert translator.calls[0]["translated_text"] == "translation 1"
    assert len(tts.requests) == 1
    assert tts.requests[0].output_file_stem.endswith("segment-000001")
    assert timed.segments[0].translated_text == "translation 0"
    assert timed.segments[0].audio_file == original.segments[0].audio_file
    assert timed.segments[1].translated_text == "bản dịch ngắn"
    assert timed.segments[1].playback_rate == pytest.approx(1.05)
    assert unaffected_checkpoint.read_bytes() == unaffected_contents
    assert (tmp_path / "segment-000001.localization-tts.json").is_file()


def test_timing_recovery_is_bounded_and_reports_final_failure(tmp_path):
    class RepeatedShorteningProvider(ShorteningProvider):
        def shorten_translation(self, **kwargs):
            self.calls.append(kwargs)
            return f"bản dịch ngắn {len(self.calls)}"

    translator = RepeatedShorteningProvider()
    tts = RecoveryTTSProvider([2.05, 2.02])

    with pytest.raises(
        TimingSynchronizationError,
        match="after 2 concise-translation retries.*hard playback-rate ceiling",
    ):
        _synchronize_with_translation_recovery(
            _synthesized_for_recovery(tmp_path), translator, tts,
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path),
                timing_policy=TimingPolicy(),
                timing_recovery_attempts=2,
                still_frame_extension=StillFrameExtensionOptions(enabled=False),
            ),
        )

    assert len(translator.calls) == 2
    assert len(tts.requests) == 2


def test_timing_recovery_never_exceeds_hard_playback_rate_ceiling(tmp_path):
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path), ShorteningProvider(),
        RecoveryTTSProvider([1.105]),
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path),
            timing_policy=TimingPolicy(),
        ),
    )

    assert timed.segments[1].playback_rate == 1.10
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_still_frame_fallback_handles_111_rate_without_a_shortening_provider(tmp_path):
    """A resumed job can retain its checkpoints when no new Gemini call is possible."""
    original = _synthesized_for_recovery(tmp_path, durations=(1.0, 1.111))

    timed = _synchronize_with_translation_recovery(
        original, FakeTranslationProvider(), RecoveryTTSProvider([]),
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            still_frame_extension=StillFrameExtensionOptions(enabled=True, fps=30),
        ),
        media_end=2.0,
    )

    assert len(timed.video_extensions) == 1
    assert timed.video_extensions[0].segment_index == 1
    assert timed.video_extensions[0].duration == pytest.approx(1 / 30)
    assert all(segment.playback_rate <= 1.10 for segment in timed.segments)


def test_timing_recovery_does_not_shorten_final_segment_that_fits_media_tail(tmp_path):
    translator = ShorteningProvider()
    tts = RecoveryTTSProvider([])

    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(1.0, 1.8)),
        translator,
        tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
        media_end=3.0,
    )

    assert not translator.calls
    assert not tts.requests
    assert timed.segments[-1].timing_action == "borrow_gap"


def create_test_video(path: Path, duration: int = 2):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def fake_transcribe(video_path, transcription_options, model_path):
    assert Path(video_path).is_file()
    assert transcription_options.task == Task.TRANSCRIBE
    return LocalizationTranscript(
        source_file=video_path,
        source_language="en",
        segments=(
            LocalizationSegment(
                start=0.25,
                end=1.50,
                text="Hello everyone",
            ),
        ),
    )


def test_end_to_end_workflow_creates_final_mp4_and_subtitle(tmp_path):
    source = tmp_path / "source.mp4"
    output_dir = tmp_path / "output"
    create_test_video(source)

    progress = []
    result = localize_video(
        str(source),
        TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path",
        FakeTranslationProvider(),
        FakeTTSProvider(),
        LocalizationWorkflowOptions(
            output_directory=str(output_dir),
            render_options=FinalRenderOptions(subtitle_mode="soft"),
        ),
        progress_callback=progress.append,
        transcribe_func=fake_transcribe,
    )

    assert Path(result.final_video.video_file).is_file()
    assert Path(result.subtitle.subtitle_file).is_file()
    assert Path(result.localized_audio_file).is_file()
    assert result.final_video.subtitle_included is True
    assert result.subtitle.cue_count == 1

    stages = [item.stage for item in progress]
    assert stages == [
        LocalizationStage.TRANSCRIBE,
        LocalizationStage.TRANSLATE,
        LocalizationStage.TTS,
        LocalizationStage.TIMING,
        LocalizationStage.EXTRACT_AUDIO,
        LocalizationStage.MIX_AUDIO,
        LocalizationStage.SUBTITLES,
        LocalizationStage.RENDER,
        LocalizationStage.COMPLETED,
    ]
    assert progress[-1].fraction == 1.0

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1",
            result.final_video.video_file,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "codec_type=video" in probe.stdout
    assert "codec_type=audio" in probe.stdout
    assert "codec_type=subtitle" in probe.stdout

    duration_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            result.final_video.video_file,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(duration_probe.stdout.strip()) == pytest.approx(2.0, abs=0.1)


def test_initial_tts_edge_cooldown_reports_progress_without_sleeping(tmp_path):
    source = tmp_path / "source.mp4"
    create_test_video(source)

    class FlakyTTS(FakeTTSProvider):
        def __init__(self):
            self.calls = 0

        def synthesize(self, request):
            self.calls += 1
            if self.calls == 1:
                raise TTSServiceTemporarilyUnavailable(9)
            return super().synthesize(request)

    class ImmediateEvent:
        def is_set(self):
            return False

        def wait(self, _delay):
            return False

    progress = []
    localize_video(
        str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", FakeTranslationProvider(), FlakyTTS(),
        LocalizationWorkflowOptions(output_directory=str(tmp_path / "output")),
        cancel_event=ImmediateEvent(), progress_callback=progress.append,
        transcribe_func=fake_transcribe,
    )

    assert any("Edge TTS tạm thời không khả dụng, tự thử lại sau 9 giây" in item.message
               for item in progress)


def test_initial_tts_edge_cooldown_honors_cancellation_without_sleeping(tmp_path):
    source = tmp_path / "source.mp4"
    create_test_video(source)

    class UnavailableTTS:
        def synthesize(self, _request):
            raise TTSServiceTemporarilyUnavailable(60)

    class CancelledEvent:
        def is_set(self):
            return False

        def wait(self, _delay):
            return True

    with pytest.raises(LocalizationCancelled):
        localize_video(
            str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
            "unused-model-path", FakeTranslationProvider(), UnavailableTTS(),
            LocalizationWorkflowOptions(output_directory=str(tmp_path / "output")),
            cancel_event=CancelledEvent(), transcribe_func=fake_transcribe,
        )


def test_workflow_loads_matching_recovery_checkpoint_before_initial_tts(tmp_path):
    source = tmp_path / "source.mp4"
    output_dir = tmp_path / "output"
    create_test_video(source)
    duration = probe_media_duration(source)
    options = LocalizationWorkflowOptions(output_directory=str(output_dir))
    translation_provider = FakeTranslationProvider()
    translated_text = translation_provider.translate("Hello everyone", "en", "vi")
    translated = TranslatedLocalizationTranscript(
        source_language="en", target_language="vi", source_file=str(source),
        segments=(TranslatedLocalizationSegment(
            0.25, 1.50, "Hello everyone", translated_text,
        ),),
    )
    workspace = output_dir / "source_localization_work"
    recovery_checkpoint = workspace / "timing-recovery-checkpoint.json"
    fingerprint = _timing_recovery_fingerprint(translated, options, duration)
    _persist_timing_recovery_checkpoint(
        recovery_checkpoint, fingerprint, translated.segments, 0, "shortened",
    )

    class CountingTTS(FakeTTSProvider):
        def __init__(self):
            self.requests = []

        def synthesize(self, request):
            self.requests.append(request.text)
            return super().synthesize(request)

    provider = CountingTTS()
    from buzz.localization.tts import synthesize_for_localization
    recovered = _load_timing_recovery_checkpoint(recovery_checkpoint, fingerprint, translated)
    synthesize_for_localization(recovered, provider, workspace / "tts")
    assert provider.requests == ["shortened"]

    localize_video(
        str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", translation_provider, provider, options,
        transcribe_func=fake_transcribe,
    )

    assert provider.requests == ["shortened"]


def test_workflow_can_be_cancelled_before_start(tmp_path):
    source = tmp_path / "source.mp4"
    create_test_video(source)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(LocalizationCancelled):
        localize_video(
            str(source),
            TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
            "unused-model-path",
            FakeTranslationProvider(),
            FakeTTSProvider(),
            LocalizationWorkflowOptions(output_directory=str(tmp_path / "output")),
            cancel_event=cancel,
            transcribe_func=fake_transcribe,
        )


def test_workflow_maps_paraformer_cancellation_and_passes_event(tmp_path):
    source = tmp_path / "source.mp4"
    create_test_video(source)
    cancel = threading.Event()

    def cancelled_transcribe(*_args, **kwargs):
        assert kwargs["cancel_event"] is cancel
        raise LocalizationTranscriptionCancelled("cancelled in ASR")

    with pytest.raises(LocalizationCancelled, match="cancelled in ASR"):
        localize_video(
            str(source), TranscriptionOptions(language="zh", task=Task.TRANSCRIBE),
            "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
            LocalizationWorkflowOptions(output_directory=str(tmp_path / "output")),
            cancel_event=cancel, transcribe_func=cancelled_transcribe,
            asr_provider="paraformer-zh",
        )


def test_cleanup_removes_only_workspace(tmp_path):
    source = tmp_path / "source.mp4"
    output_dir = tmp_path / "output"
    create_test_video(source)

    result = localize_video(
        str(source),
        TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path",
        FakeTranslationProvider(),
        FakeTTSProvider(),
        LocalizationWorkflowOptions(output_directory=str(output_dir)),
        transcribe_func=fake_transcribe,
    )

    final_video = Path(result.final_video.video_file)
    subtitle = Path(result.subtitle.subtitle_file)
    workspace = Path(result.workspace)

    cleanup_localization_workspace(result)

    assert not workspace.exists()
    assert final_video.is_file()
    assert subtitle.is_file()


def test_workflow_rejects_missing_input(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        localize_video(
            str(tmp_path / "missing.mp4"),
            TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
            "model",
            FakeTranslationProvider(),
            FakeTTSProvider(),
            LocalizationWorkflowOptions(output_directory=str(tmp_path / "output")),
            transcribe_func=fake_transcribe,
        )


@pytest.mark.parametrize(
    "source_language,source_text,translated_text,route",
    [
        ("en", "Hello everyone", "Xin chào mọi người", ["en", "vi"]),
        ("zh", "大家好", "Xin chào mọi người", ["zh", "en", "vi"]),
    ],
)
def test_end_to_end_workflow_accepts_argos_provider(
    monkeypatch,
    tmp_path,
    source_language,
    source_text,
    translated_text,
    route,
):
    from buzz.localization.providers import ArgosTranslationProvider

    source = tmp_path / f"source-{source_language}.mp4"
    output_dir = tmp_path / f"output-{source_language}"
    create_test_video(source)

    def fake_worker(command, src, target="vi", text=None):
        assert src == source_language
        if command == "route":
            return {"ok": True, "route": route}
        assert text == source_text
        return {
            "ok": True,
            "route": route,
            "text": translated_text,
        }

    monkeypatch.setattr(
        "buzz.localization.providers._run_argos_worker",
        fake_worker,
    )

    def transcribe(video_path, transcription_options, model_path):
        return LocalizationTranscript(
            source_file=video_path,
            source_language=source_language,
            segments=(
                LocalizationSegment(
                    start=0.25,
                    end=1.50,
                    text=source_text,
                ),
            ),
        )

    result = localize_video(
        str(source),
        TranscriptionOptions(language=source_language, task=Task.TRANSCRIBE),
        "unused-model-path",
        ArgosTranslationProvider(),
        FakeTTSProvider(),
        LocalizationWorkflowOptions(
            output_directory=str(output_dir),
            render_options=FinalRenderOptions(subtitle_mode="soft"),
        ),
        transcribe_func=transcribe,
    )

    assert Path(result.final_video.video_file).is_file()
    subtitle_text = Path(result.subtitle.subtitle_file).read_text(encoding="utf-8")
    assert translated_text in subtitle_text


def test_end_to_end_workflow_accepts_gemini_provider(monkeypatch, tmp_path):
    from buzz.localization.providers import GeminiTranslationProvider

    segment_id = _segment_id(
        0,
        LocalizationSegment(start=0.25, end=1.50, text="Hello everyone"),
    )

    class Response:
        text = (
            '{"segments":[{"id":"' + segment_id
            + '","corrected_text":"Hello everyone",'
            '"translated_text":"Xin chào mọi người",'
            '"revised_translated_text":"Xin chào mọi người"}]}'
        )

    class Models:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "gemini-test"
            return Response()

    class Client:
        models = Models()

    source = tmp_path / "source-gemini.mp4"
    output_dir = tmp_path / "output-gemini"
    create_test_video(source)

    provider = GeminiTranslationProvider(
        api_key="test-key",
        model="gemini-test",
    )
    provider._client = Client()

    result = localize_video(
        str(source),
        TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path",
        provider,
        FakeTTSProvider(),
        LocalizationWorkflowOptions(
            output_directory=str(output_dir),
            render_options=FinalRenderOptions(subtitle_mode="soft"),
        ),
        transcribe_func=fake_transcribe,
    )

    assert Path(result.final_video.video_file).is_file()
    assert "Xin chào mọi người" in Path(
        result.subtitle.subtitle_file
    ).read_text(encoding="utf-8")

def test_chunked_video_joins_parts_and_offsets_subtitles(tmp_path):
    source = tmp_path / "long_source.mp4"
    create_test_video(source, duration=4)
    output = tmp_path / "output"
    progress = []
    class RecordingScheduler(LocalizationResourceScheduler):
        def __init__(self):
            super().__init__()
            self.resources = []

        def acquire(self, resource, **kwargs):
            self.resources.append(resource)
            return super().acquire(resource, **kwargs)

    scheduler = RecordingScheduler()
    result = localize_video(
        str(source),
        TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
        LocalizationWorkflowOptions(
            output_directory=str(output), chunk_duration_seconds=2,
            render_options=FinalRenderOptions(subtitle_mode="soft"),
            resource_scheduler=scheduler,
        ),
        progress_callback=progress.append,
        transcribe_func=fake_transcribe,
    )
    assert result.subtitle.cue_count == 2
    text = Path(result.subtitle.subtitle_file).read_text(encoding="utf-8")
    assert "00:00:00,250 --> 00:00:01,500" in text
    times = [line for line in text.splitlines() if "-->" in line]
    assert len(times) == 2
    start, end = times[1].split(" --> ")
    assert start.startswith("00:00:02,")
    assert end.startswith("00:00:03,")
    assert 250 <= int(start.rsplit(",", 1)[1]) < 350
    assert int(end.rsplit(",", 1)[1]) - int(start.rsplit(",", 1)[1]) == 250
    assert Path(result.localized_audio_file).is_file()
    assert scheduler.resources.count("prepare") == 2
    assert progress[-1].fraction == 1.0
    assert any("Part 2/2" in item.message for item in progress)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type", "-of", "json",
         result.final_video.video_file],
        check=True, capture_output=True, text=True,
    )
    import json
    streams = json.loads(probe.stdout)
    assert float(streams["format"]["duration"]) == pytest.approx(4.0, abs=0.2)
    assert {item["codec_type"] for item in streams["streams"]} == {
        "video", "audio", "subtitle",
    }
    packets = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         result.final_video.video_file],
        check=True, capture_output=True, text=True,
    )
    second_start = 2 + int(start.rsplit(",", 1)[1]) / 1000
    assert any(
        abs(float(pts) - second_start) < 0.01
        for pts in packets.stdout.splitlines() if pts.strip()
    )


def test_chunked_provider_identity_ignores_rate_limit_fallback_order():
    from dataclasses import dataclass
    from buzz.localization.chunked import _provider_identity

    @dataclass
    class Provider:
        api_key: str
        model: str
        timeout: float = 60.0
        rate_limit_fallback_models: tuple[str, ...] = ()

    first = Provider(
        "secret", "gemini-primary",
        rate_limit_fallback_models=("fallback-a", "fallback-b"),
    )
    second = Provider(
        "different-secret", "gemini-primary",
        rate_limit_fallback_models=("fallback-c",),
    )
    changed_model = Provider("secret", "gemini-other")

    assert _provider_identity(first) == _provider_identity(second)
    assert _provider_identity(first) != _provider_identity(changed_model)


def test_chunked_video_resumes_completed_parts_and_reprocesses_invalid_artifact(
    tmp_path,
):
    source = tmp_path / "resume_source.mp4"
    create_test_video(source, duration=4)
    output = tmp_path / "output"
    transcribe_calls = []
    translation_calls = []
    tts_calls = []

    def counting_transcribe(video_path, transcription_options, model_path):
        transcribe_calls.append(video_path)
        return fake_transcribe(video_path, transcription_options, model_path)

    class CountingTranslationProvider(FakeTranslationProvider):
        def translate(self, text, source_language, target_language):
            translation_calls.append(text)
            return super().translate(text, source_language, target_language)

    class CountingTTSProvider(FakeTTSProvider):
        def synthesize(self, request):
            tts_calls.append(request.text)
            return super().synthesize(request)

    options = LocalizationWorkflowOptions(
        output_directory=str(output), chunk_duration_seconds=2,
        render_options=FinalRenderOptions(subtitle_mode="soft"),
    )
    provider = CountingTranslationProvider()
    tts_provider = CountingTTSProvider()

    localize_video(
        str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", provider, tts_provider, options,
        transcribe_func=counting_transcribe,
    )
    assert len(transcribe_calls) == len(translation_calls) == len(tts_calls) == 2

    progress = []
    localize_video(
        str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", provider, tts_provider, options,
        progress_callback=progress.append, transcribe_func=counting_transcribe,
    )
    assert len(transcribe_calls) == len(translation_calls) == len(tts_calls) == 2
    assert sum("reused completed part" in item.message for item in progress) == 2

    workspace = output / "resume_source_localization_work"
    (workspace / "part_0002" / "source.vi.mp4").unlink()
    result = localize_video(
        str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", provider, tts_provider, options,
        transcribe_func=counting_transcribe,
    )
    assert len(transcribe_calls) == len(translation_calls) == len(tts_calls) == 3
    assert result.subtitle.cue_count == 2
    assert "00:00:02," in Path(result.subtitle.subtitle_file).read_text(encoding="utf-8")
    assert probe_media_duration(result.final_video.video_file) == pytest.approx(4.0, abs=0.2)


def test_chunked_video_changed_option_and_source_invalidate_completed_parts(tmp_path):
    source = tmp_path / "invalidate_source.mp4"
    create_test_video(source, duration=4)
    output = tmp_path / "output"
    calls = []

    def counting_transcribe(video_path, transcription_options, model_path):
        calls.append(video_path)
        return fake_transcribe(video_path, transcription_options, model_path)

    def run(*, voice=None):
        return localize_video(
            str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
            "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
            LocalizationWorkflowOptions(
                output_directory=str(output), chunk_duration_seconds=2,
                tts_voice=voice,
            ),
            transcribe_func=counting_transcribe,
        )

    run()
    assert len(calls) == 2
    run(voice="vi-VN-NamMinhNeural")
    assert len(calls) == 4
    source.touch()
    run(voice="vi-VN-NamMinhNeural")
    assert len(calls) == 6


def test_chunked_video_failure_does_not_write_completion_manifest(tmp_path):
    source = tmp_path / "failure_source.mp4"
    create_test_video(source, duration=4)
    output = tmp_path / "output"

    def failing_transcribe(*args, **kwargs):
        raise RuntimeError("test transcription failure")

    with pytest.raises(RuntimeError, match="test transcription failure"):
        localize_video(
            str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
            "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
            LocalizationWorkflowOptions(
                output_directory=str(output), chunk_duration_seconds=2,
            ),
            transcribe_func=failing_transcribe,
        )

    manifest = (
        output / "failure_source_localization_work" / "part_0001"
        / "localization-part-complete.json"
    )
    assert not manifest.exists()


def test_chunked_video_cancellation_between_parts(tmp_path):
    source = tmp_path / "long_source.mp4"
    create_test_video(source, duration=4)
    cancel = threading.Event()

    def stop_after_first_part(progress):
        if "Part 1/2" in progress.message and progress.stage == LocalizationStage.RENDER:
            cancel.set()

    with pytest.raises(LocalizationCancelled):
        localize_video(
            str(source),
            TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
            "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path / "out"), chunk_duration_seconds=2,
            ),
            progress_callback=stop_after_first_part, cancel_event=cancel,
            transcribe_func=fake_transcribe,
        )


def test_chunked_video_cancels_blocked_prepare_and_removes_partial_clip(tmp_path, monkeypatch):
    from buzz.localization import chunked
    from buzz.localization.audio_mix import AudioMixingCancelled

    source = tmp_path / "long_source.mp4"
    create_test_video(source, duration=2)
    cancelled = threading.Event()

    def blocked_prepare(command, *, cancel_event=None):
        assert command[command.index("-threads") + 1] == "4"
        clip = Path(command[-1])
        clip.write_bytes(b"partial")
        assert cancel_event.wait(2)
        raise AudioMixingCancelled("Audio processing was cancelled")

    monkeypatch.setattr(chunked, "_run_ffmpeg", blocked_prepare)
    trigger = threading.Timer(0.1, cancelled.set)
    trigger.start()
    try:
        with pytest.raises(LocalizationCancelled, match="cancelled"):
            localize_video(
                str(source), TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
                "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
                LocalizationWorkflowOptions(
                    output_directory=str(tmp_path / "out"), chunk_duration_seconds=2,
                ),
                cancel_event=cancelled, transcribe_func=fake_transcribe,
            )
    finally:
        trigger.cancel()
    assert not (tmp_path / "out" / "long_source_localization_work" / "part_0001" / "source.mp4").exists()


def test_chunked_paraformer_transcription_receives_cancel_event(tmp_path, monkeypatch):
    from buzz.localization import chunked

    source = tmp_path / "long_source.mp4"
    create_test_video(source, duration=2)
    cancelled = threading.Event()

    def prepare(command, *, cancel_event=None):
        Path(command[-1]).write_bytes(b"clip")

    def cancelled_paraformer(video_path, transcription_options, model_path, **kwargs):
        assert kwargs["asr_provider"] == "paraformer-zh"
        assert kwargs["cancel_event"] is cancelled
        raise LocalizationTranscriptionCancelled("Paraformer cancelled")

    monkeypatch.setattr(chunked, "_run_ffmpeg", prepare)
    with pytest.raises(LocalizationCancelled, match="Paraformer cancelled"):
        localize_video(
            str(source), TranscriptionOptions(language="zh", task=Task.TRANSCRIBE),
            "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path / "out"), chunk_duration_seconds=2,
            ),
            cancel_event=cancelled, transcribe_func=cancelled_paraformer,
            asr_provider="paraformer-zh",
        )

@pytest.mark.parametrize("mode", ["soft", "burn", "none"])
def test_chunked_video_preserves_silent_part(tmp_path, mode):
    source = tmp_path / "long_source.mp4"
    create_test_video(source, duration=4)
    calls = 0

    def transcribe_first_part_only(video_path, transcription_options, model_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return fake_transcribe(video_path, transcription_options, model_path)
        return LocalizationTranscript(
            source_file=video_path, source_language="en", segments=(),
        )

    result = localize_video(
        str(source),
        TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path / "output"), chunk_duration_seconds=2,
            render_options=FinalRenderOptions(subtitle_mode=mode),
        ),
        transcribe_func=transcribe_first_part_only,
    )
    assert calls == 2
    assert result.subtitle.cue_count == 1
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         result.final_video.video_file],
        check=True, capture_output=True, text=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(4.0, abs=0.2)


def test_chunked_vtt_merges_with_absolute_offsets(tmp_path):
    from buzz.localization.chunked import _merge_subtitles
    first = tmp_path / "part1.vtt"
    second = tmp_path / "part2.vtt"
    first.write_text("WEBVTT\n\n00:00:00.500 --> 00:00:01.500\nMột\n", encoding="utf-8")
    second.write_text("WEBVTT\n\n00:00:00.200 --> 00:00:01.200\nHai\n", encoding="utf-8")
    final = tmp_path / "final.vtt"
    assert _merge_subtitles([(first, 0), (second, 900)], final, "vtt") == 2
    text = final.read_text(encoding="utf-8")
    assert "00:15:00.200 --> 00:15:01.200" in text


def test_natural_timing_defaults_to_four_recovery_attempts(tmp_path):
    assert LocalizationWorkflowOptions(output_directory=str(tmp_path)).timing_recovery_attempts == 4


def test_terminal_asr_overrun_uses_local_tail_without_translation_or_tts(tmp_path):
    audio = tmp_path / "segment-000145.wav"
    audio.write_bytes(b"audio")
    synthesized = SynthesizedLocalizationTranscript(
        source_file="source.mp4", source_language="en", target_language="vi",
        segments=(SynthesizedLocalizationSegment(
            899.97, 902.20, "last source", "dÃ²ng cuá»‘i", str(audio), 4.56,
            "test", "voice",
        ),),
    )
    translator = ShorteningProvider()
    tts = RecoveryTTSProvider([1.0])
    timed = _synchronize_with_translation_recovery(
        synthesized, translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            still_frame_extension=StillFrameExtensionOptions(
                fps=30, max_terminal_seconds=5,
            ),
        ),
        media_end=900.021334,
    )
    assert not translator.calls
    assert not tts.requests
    assert timed.segments[-1].translated_text == "dÃ²ng cuá»‘i"
    assert timed.video_extensions[0].at == pytest.approx(900.021334)
    assert timed.segments[-1].end <= 900.021334 + timed.video_extensions[0].duration
    assert timed.segments[-1].playback_rate <= 1.10


def test_terminal_asr_overrun_refusal_is_actionable_without_remote_calls(tmp_path):
    audio = tmp_path / "segment.wav"
    audio.write_bytes(b"audio")
    synthesized = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(9.9, 12, "a", "b", str(audio), 4.56, "test", None),
        ),
    )
    translator, tts = ShorteningProvider(), RecoveryTTSProvider([1.0])
    with pytest.raises(TimingSynchronizationError, match="terminal still-frame extension was refused"):
        _synchronize_with_translation_recovery(
            synthesized, translator, tts,
            LocalizationWorkflowOptions(
                output_directory=str(tmp_path),
                still_frame_extension=StillFrameExtensionOptions(enabled=False),
            ), media_end=10,
        )
    assert not translator.calls
    assert not tts.requests

def test_timing_recovery_auto_stretches_after_exhausted_gemini_without_cutting(tmp_path):
    translator = RepeatedShorteningProvider()
    tts = RecoveryTTSProvider([1.30] * 4)
    original = _synthesized_for_recovery(tmp_path, durations=(1.0, 1.30))
    policy = TimingPolicy()
    timed = _synchronize_with_translation_recovery(
        original, translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=policy,
            timing_recovery_attempts=4,
            still_frame_extension=StillFrameExtensionOptions(
                enabled=True, fps=30, max_seconds_per_event=0.5,
            ),
        ),
        media_end=2,
    )
    assert len(translator.calls) == len(tts.requests) == 4
    assert len(timed.segments) == 2
    assert len(timed.video_extensions) == 1
    assert timed.video_extensions[0].segment_index == 1
    assert timed.video_extensions[0].at == pytest.approx(2)
    assert timed.video_extensions[0].duration <= 0.5
    assert timed.segments[1].audio_duration == pytest.approx(1.30)
    assert all(x.playback_rate <= policy.max_playback_rate for x in timed.segments)


def _synthesized_with_misleading_preceding_gap(tmp_path):
    """The source gap is slightly smaller than the synchronizer's needed borrow."""
    segments = []
    timings = ((0.0, 1.24, 1.22), (1.25, 2.25, 1.106))
    for index, (start, end, duration) in enumerate(timings):
        audio = tmp_path / f"gap-segment-{index:06d}.wav"
        audio.write_bytes(b"audio")
        segments.append(SynthesizedLocalizationSegment(
            start=start, end=end,
            source_text=f"source {index}", translated_text=f"translation {index}",
            audio_file=str(audio), audio_duration=duration,
            provider="fake", voice="vi-test",
        ))
    return SynthesizedLocalizationTranscript(
        source_language="en", target_language="vi", source_file="source.mp4",
        segments=tuple(segments),
    )


def test_timing_recovery_uses_exact_failure_when_preflight_overflow_misses_gap(tmp_path):
    translator = ShorteningProvider()
    tts = RecoveryTTSProvider([0.9])

    timed = _synchronize_with_translation_recovery(
        _synthesized_with_misleading_preceding_gap(tmp_path), translator, tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert len(translator.calls) == len(tts.requests) == 1
    assert translator.calls[0]["required_rate"] == pytest.approx(1.106)
    assert tts.requests[0].text == "bản dịch ngắn"
    assert timed.segments[1].playback_rate <= 1.10


def test_timing_recovery_uses_bounded_still_frame_when_exact_failure_is_missed(tmp_path):
    translator = RepeatedShorteningProvider()
    tts = RecoveryTTSProvider([1.106])

    timed = _synchronize_with_translation_recovery(
        _synthesized_with_misleading_preceding_gap(tmp_path), translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            timing_recovery_attempts=1,
            still_frame_extension=StillFrameExtensionOptions(
                enabled=True, fps=30, max_seconds_per_event=0.5,
            ),
        ),
    )

    assert len(translator.calls) == len(tts.requests) == 1
    assert len(timed.video_extensions) == 1
    assert timed.video_extensions[0].segment_index == 1
    assert timed.video_extensions[0].duration <= 0.5
    assert all(segment.playback_rate <= 1.10 for segment in timed.segments)



def test_terminal_asr_overrun_keeps_tail_while_recovering_earlier_voice(tmp_path):
    """The real 146-line job also has earlier rate overflow; tail fix must not exit early."""
    timeline = (
        (0.0, 1.0, 1.40),
        (1.0, 2.0, 1.00),
        (899.97, 902.20, 4.56),
    )
    segments = []
    for i, (start, end, audio_duration) in enumerate(timeline):
        path = tmp_path / f"source-segment-{i:06d}.wav"
        path.write_bytes(b"audio")
        segments.append(SynthesizedLocalizationSegment(
            start, end, f"source {i}", f"Vietnamese {i}",
            str(path), audio_duration, "fake", "voice",
        ))
    synthesized = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", tuple(segments),
    )
    translator = ShorteningProvider()
    tts = RecoveryTTSProvider([0.9])
    timed = _synchronize_with_translation_recovery(
        synthesized, translator, tts,
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            still_frame_extension=StillFrameExtensionOptions(
                fps=30, max_terminal_seconds=5,
            ),
        ),
        media_end=900.021334,
    )
    assert len(translator.calls) == len(tts.requests) == 1
    assert len(timed.segments) == 3
    assert timed.segments[-1].source_text == "source 2"
    assert timed.segments[-1].translated_text == "Vietnamese 2"
    assert len(timed.video_extensions) == 1
    assert timed.video_extensions[0].at == pytest.approx(900.021334)
    assert timed.segments[-1].end <= 900.021334 + timed.video_extensions[0].duration
    assert all(segment.playback_rate <= 1.10 for segment in timed.segments)


def test_terminal_and_internal_freezes_share_the_total_extension_cap(tmp_path):
    # A neighboring segment is required to make the first utterance overflow.
    # Without it the 899s silent gap is available and no internal freeze occurs.
    timeline = (
        (0.0, 1.0, 1.40),
        (1.0, 2.0, 0.40),
        (899.97, 902.20, 4.56),
    )
    segments = []
    for index, (start, end, audio_duration) in enumerate(timeline):
        audio = tmp_path / f"segment-{index:06d}.wav"
        audio.write_bytes(b"audio")
        segments.append(SynthesizedLocalizationSegment(
            start, end, f"source {index}", f"translation {index}",
            str(audio), audio_duration, "fake", "voice",
        ))
    timed = _synchronize_with_translation_recovery(
        SynthesizedLocalizationTranscript("source.mp4", "en", "vi", tuple(segments)),
        RepeatedShorteningProvider(), RecoveryTTSProvider([1.40]),
        LocalizationWorkflowOptions(
            output_directory=str(tmp_path), timing_policy=TimingPolicy(),
            timing_recovery_attempts=1,
            still_frame_extension=StillFrameExtensionOptions(
                fps=30, max_seconds_total=5, max_terminal_seconds=5,
            ),
        ),
        media_end=900.021334,
    )
    assert [event.segment_index for event in timed.video_extensions] == [0, 2]
    assert sum(event.duration for event in timed.video_extensions) <= 5
    assert timed.video_extensions[-1].at == pytest.approx(900.021334)
    assert all(segment.playback_rate <= 1.10 for segment in timed.segments)
