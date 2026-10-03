import json
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
import threading
import time

import pytest

from buzz.localization.translation import (
    TranslatedLocalizationSegment,
    TranslatedLocalizationTranscript,
)
from buzz.localization.tts import (
    SynthesizedLocalizationSegment,
    TTSRequest,
    TTSResult,
    synthesize_for_localization,
)


class FakeTTSProvider:
    def __init__(self, durations=None, provider="fake", voice="vi-test"):
        self.durations = durations or {}
        self.provider = provider
        self.voice = voice
        self.calls = []

    def synthesize(self, request: TTSRequest):
        self.calls.append(request)
        audio_file = request.output_file_stem + ".wav"
        Path(audio_file).write_bytes(b"fake audio")
        return TTSResult(
            audio_file=audio_file,
            audio_duration=self.durations.get(request.text, 1.25),
            provider=self.provider,
            voice=request.voice or self.voice,
            metadata={"format": "wav"},
        )


def make_transcript(segments=None, target_language="vi"):
    return TranslatedLocalizationTranscript(
        source_file="input.mp4",
        source_language="en",
        target_language=target_language,
        segments=tuple(
            segments
            if segments is not None
            else [
                TranslatedLocalizationSegment(
                    start=0.0,
                    end=3.2,
                    source_text="Hello everyone",
                    translated_text="Xin chào mọi người",
                )
            ]
        ),
    )


def test_synthesizes_vietnamese_text_unchanged(tmp_path):
    transcript = make_transcript()
    provider = FakeTTSProvider()

    result = synthesize_for_localization(
        transcript, provider, tmp_path, voice="vi-female"
    )

    request = provider.calls[0]
    assert request.text == "Xin chào mọi người"
    assert request.language == "vi"
    assert request.output_file_stem == str(tmp_path / "segment-000000")
    assert request.voice == "vi-female"
    assert result.segments == (
        SynthesizedLocalizationSegment(
            start=0.0,
            end=3.2,
            source_text="Hello everyone",
            translated_text="Xin chào mọi người",
            audio_file=str(tmp_path / "segment-000000.wav"),
            audio_duration=1.25,
            provider="fake",
            voice="vi-female",
            provider_metadata={"format": "wav"},
        ),
    )


def test_preserves_segment_order_and_audio_mapping(tmp_path):
    transcript = make_transcript(
        segments=[
            TranslatedLocalizationSegment(5.0, 7.0, "Second", "Thứ hai"),
            TranslatedLocalizationSegment(0.0, 2.0, "First", "Thứ nhất"),
        ]
    )
    provider = FakeTTSProvider({"Thứ hai": 0.8, "Thứ nhất": 1.1})

    result = synthesize_for_localization(transcript, provider, tmp_path)

    assert [segment.translated_text for segment in result.segments] == [
        "Thứ hai",
        "Thứ nhất",
    ]
    assert [segment.audio_file for segment in result.segments] == [
        str(tmp_path / "segment-000000.wav"),
        str(tmp_path / "segment-000001.wav"),
    ]
    assert [segment.audio_duration for segment in result.segments] == [0.8, 1.1]


def test_preserves_unicode_and_provider_metadata(tmp_path):
    text = "Tôi đang học tiếng Việt ở Hà Nội."
    transcript = make_transcript(
        segments=[TranslatedLocalizationSegment(0.2, 2.9, "Source", text)]
    )
    provider = FakeTTSProvider(provider="example", voice="vi-VN-1")

    result = synthesize_for_localization(transcript, provider, tmp_path)

    assert result.segments[0].translated_text == text
    assert result.segments[0].provider == "example"
    assert result.segments[0].voice == "vi-VN-1"
    assert result.segments[0].provider_metadata == {"format": "wav"}


def test_passes_provider_options_without_mutating_them(tmp_path):
    options = {"rate": "+5%", "pitch": "default"}
    provider = FakeTTSProvider()

    synthesize_for_localization(
        make_transcript(), provider, tmp_path, options=options
    )

    assert dict(provider.calls[0].options) == options
    assert options == {"rate": "+5%", "pitch": "default"}


def test_output_is_json_compatible(tmp_path):
    result = synthesize_for_localization(
        make_transcript(), FakeTTSProvider(), tmp_path
    )

    encoded = json.dumps(result.to_dict(), ensure_ascii=False)

    assert json.loads(encoded) == result.to_dict()
    assert "Xin chào mọi người" in encoded


def test_rejects_empty_transcript_without_calling_provider(tmp_path):
    provider = FakeTTSProvider()

    with pytest.raises(ValueError, match="empty transcript"):
        synthesize_for_localization(
            make_transcript(segments=[]), provider, tmp_path
        )

    assert provider.calls == []


@pytest.mark.parametrize("text", ["", "   ", None])
def test_rejects_empty_translated_segment(text, tmp_path):
    transcript = make_transcript(
        segments=[TranslatedLocalizationSegment(0, 1, "Source", text)]
    )
    provider = FakeTTSProvider()

    with pytest.raises(ValueError, match="empty translated segment"):
        synthesize_for_localization(transcript, provider, tmp_path)

    assert provider.calls == []


def test_rejects_non_vietnamese_target_language(tmp_path):
    with pytest.raises(ValueError, match="target language"):
        synthesize_for_localization(
            make_transcript(target_language="en"),
            FakeTTSProvider(),
            tmp_path,
        )


@pytest.mark.parametrize("output_directory", ["", None])
def test_rejects_missing_output_directory(output_directory):
    with pytest.raises(ValueError, match="Output directory"):
        synthesize_for_localization(
            make_transcript(), FakeTTSProvider(), output_directory
        )


def test_propagates_provider_exception(tmp_path):
    class FailingProvider:
        def synthesize(self, request):
            raise TimeoutError("tts timed out")

    with pytest.raises(TimeoutError, match="tts timed out"):
        synthesize_for_localization(
            make_transcript(), FailingProvider(), tmp_path
        )


def test_rejects_invalid_provider_response(tmp_path):
    class InvalidProvider:
        def synthesize(self, request):
            return {"audio_file": request.output_file_stem + ".wav"}

    with pytest.raises(ValueError, match="invalid response"):
        synthesize_for_localization(
            make_transcript(), InvalidProvider(), tmp_path
        )


@pytest.mark.parametrize("duration", [0, -1, None, "1.0", True])
def test_rejects_invalid_audio_duration(duration, tmp_path):
    class InvalidDurationProvider:
        def synthesize(self, request):
            audio_file = request.output_file_stem + ".wav"
            Path(audio_file).write_bytes(b"fake")
            return TTSResult(
                audio_file=audio_file,
                audio_duration=duration,
                provider="fake",
            )

    with pytest.raises(ValueError, match="invalid audio duration"):
        synthesize_for_localization(
            make_transcript(), InvalidDurationProvider(), tmp_path
        )


def test_rejects_missing_audio_file(tmp_path):
    class MissingFileProvider:
        def synthesize(self, request):
            return TTSResult(
                audio_file=request.output_file_stem + ".wav",
                audio_duration=1.0,
                provider="fake",
            )

    with pytest.raises(ValueError, match="does not exist"):
        synthesize_for_localization(
            make_transcript(), MissingFileProvider(), tmp_path
        )


@pytest.mark.parametrize("provider_value", ["", "   ", None])
def test_rejects_invalid_provider_metadata(provider_value, tmp_path):
    class InvalidMetadataProvider:
        def synthesize(self, request):
            audio_file = request.output_file_stem + ".wav"
            Path(audio_file).write_bytes(b"fake")
            return TTSResult(
                audio_file=audio_file,
                audio_duration=1.0,
                provider=provider_value,
            )

    with pytest.raises(ValueError, match="provider metadata"):
        synthesize_for_localization(
            make_transcript(), InvalidMetadataProvider(), tmp_path
        )


def test_does_not_mutate_input_transcript(tmp_path):
    transcript = make_transcript()
    original = transcript.to_dict()

    synthesize_for_localization(
        transcript, FakeTTSProvider(), tmp_path
    )

    assert transcript.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        transcript.target_language = "en"


def test_successful_segments_resume_after_later_provider_failure(tmp_path):
    transcript = make_transcript(segments=[
        TranslatedLocalizationSegment(0, 1, "A", "Một"),
        TranslatedLocalizationSegment(1, 2, "B", "Hai"),
        TranslatedLocalizationSegment(2, 3, "C", "Ba"),
    ])

    class RestartableProvider(FakeTTSProvider):
        def __init__(self, fail):
            super().__init__()
            self.fail = fail

        def synthesize(self, request):
            if self.fail and request.text == "Ba":
                self.calls.append(request)
                raise TimeoutError("provider unavailable")
            return super().synthesize(request)

    first_provider = RestartableProvider(True)
    with pytest.raises(TimeoutError, match="provider unavailable"):
        synthesize_for_localization(transcript, first_provider, tmp_path)

    second_provider = RestartableProvider(False)
    result = synthesize_for_localization(transcript, second_provider, tmp_path)

    assert [request.text for request in first_provider.calls] == ["Một", "Hai", "Ba"]
    assert [request.text for request in second_provider.calls] == ["Ba"]
    assert len(result.segments) == 3


@pytest.mark.parametrize("change", ["text", "voice", "options"])
def test_request_change_invalidates_only_affected_segment(tmp_path, change):
    original = make_transcript(segments=[
        TranslatedLocalizationSegment(0, 1, "A", "Một"),
        TranslatedLocalizationSegment(1, 2, "B", "Hai"),
    ])
    synthesize_for_localization(
        original, FakeTTSProvider(), tmp_path, voice="vi-1", options={"rate": "+0%"}
    )
    changed = original
    voice = "vi-1"
    options = {"rate": "+0%"}
    if change == "text":
        changed = make_transcript(segments=[
            original.segments[0],
            TranslatedLocalizationSegment(1, 2, "B", "Hai mới"),
        ])
    elif change == "voice":
        # Apply the changed setting only to segment 1 by synthesizing it directly.
        from buzz.localization.tts import synthesize_segment
        provider = FakeTTSProvider()
        synthesize_segment(TTSRequest(
            "Hai", "vi", str(tmp_path / "segment-000001"), "vi-2", options
        ), provider)
        assert [request.text for request in provider.calls] == ["Hai"]
        assert (tmp_path / "segment-000000.localization-tts.json").is_file()
        return
    else:
        from buzz.localization.tts import synthesize_segment
        provider = FakeTTSProvider()
        synthesize_segment(TTSRequest(
            "Hai", "vi", str(tmp_path / "segment-000001"), voice, {"rate": "+5%"}
        ), provider)
        assert [request.text for request in provider.calls] == ["Hai"]
        assert (tmp_path / "segment-000000.localization-tts.json").is_file()
        return

    provider = FakeTTSProvider()
    synthesize_for_localization(changed, provider, tmp_path, voice=voice, options=options)
    assert [request.text for request in provider.calls] == ["Hai mới"]


@pytest.mark.parametrize("damage", ["missing", "empty", "changed"])
def test_missing_or_corrupt_checkpointed_audio_is_regenerated(tmp_path, damage):
    transcript = make_transcript()
    first = synthesize_for_localization(transcript, FakeTTSProvider(), tmp_path)
    audio = Path(first.segments[0].audio_file)
    if damage == "missing":
        audio.unlink()
    elif damage == "empty":
        audio.write_bytes(b"")
    else:
        audio.write_bytes(b"different audio")

    provider = FakeTTSProvider()
    synthesize_for_localization(transcript, provider, tmp_path)

    assert len(provider.calls) == 1
    assert audio.read_bytes() == b"fake audio"


def test_provider_failure_never_writes_completion_checkpoint(tmp_path):
    class PartialFailure:
        def synthesize(self, request):
            Path(request.output_file_stem + ".wav").write_bytes(b"partial")
            raise TimeoutError("failed after partial output")

    with pytest.raises(TimeoutError, match="partial output"):
        synthesize_for_localization(make_transcript(), PartialFailure(), tmp_path)

    assert not (tmp_path / "segment-000000.localization-tts.json").exists()
    retry = FakeTTSProvider()
    synthesize_for_localization(make_transcript(), retry, tmp_path)
    assert len(retry.calls) == 1


def test_segment_context_is_released_between_sequential_segments(tmp_path):
    transcript = make_transcript(segments=[
        TranslatedLocalizationSegment(0, 1, "one", "một"),
        TranslatedLocalizationSegment(1, 2, "two", "hai"),
    ])
    events = []

    @contextmanager
    def segment_context(index, total):
        events.append(("enter", index, total))
        yield
        events.append(("exit", index, total))

    progress = []
    synthesize_for_localization(
        transcript, FakeTTSProvider(), tmp_path,
        segment_context=segment_context,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert events == [
        ("enter", 0, 2), ("exit", 0, 2),
        ("enter", 1, 2), ("exit", 1, 2),
    ]
    assert progress == [(1, 2), (2, 2)]


def test_two_jobs_overlap_globally_but_each_job_tts_is_serial(tmp_path):
    from buzz.localization.resources import LocalizationResourceScheduler

    transcript = make_transcript(segments=[
        TranslatedLocalizationSegment(0, 1, "one", "một"),
        TranslatedLocalizationSegment(1, 2, "two", "hai"),
    ])
    scheduler = LocalizationResourceScheduler()
    lock = threading.Lock()
    active = maximum = 0
    per_job_active = {"one": 0, "two": 0}
    per_job_maximum = {"one": 0, "two": 0}

    class SlowProvider(FakeTTSProvider):
        def __init__(self, job):
            super().__init__()
            self.job = job

        def synthesize(self, request):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                per_job_active[self.job] += 1
                per_job_maximum[self.job] = max(
                    per_job_maximum[self.job], per_job_active[self.job])
            try:
                time.sleep(0.04)
                return super().synthesize(request)
            finally:
                with lock:
                    active -= 1
                    per_job_active[self.job] -= 1

    start = threading.Barrier(3)

    def run(job):
        start.wait()
        synthesize_for_localization(
            transcript, SlowProvider(job), tmp_path / job,
            segment_context=lambda _index, _total: scheduler.acquire("tts"),
        )

    threads = [threading.Thread(target=run, args=(job,)) for job in ("one", "two")]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert maximum == 2
    assert per_job_maximum == {"one": 1, "two": 1}
