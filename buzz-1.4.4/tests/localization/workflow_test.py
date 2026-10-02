from pathlib import Path
import subprocess
import threading
import time
import wave

import pytest

from buzz.localization.audio_mix import probe_media_duration
from buzz.localization.transcript import LocalizationSegment, LocalizationTranscript
from buzz.localization.translation import TranslationProvider, _segment_id
from buzz.localization.tts import TTSRequest, TTSResult
from buzz.localization.tts import (
    SynthesizedLocalizationSegment,
    SynthesizedLocalizationTranscript,
    synthesize_segment,
)
from buzz.localization.timing import TimingPolicy, TimingSynchronizationError
from buzz.localization.workflow import (
    LocalizationCancelled,
    LocalizationStage,
    LocalizationWorkflowOptions,
    cleanup_localization_workspace,
    localize_video,
    _synchronize_with_translation_recovery,
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


def test_timing_recovery_regenerates_batch_concurrently_and_applies_by_segment_index(tmp_path):
    class BatchProvider:
        def shorten_translations(self, **kwargs):
            return [f"short-{item['id']}" for item in kwargs["segments"]]

    tts = ConcurrentRecoveryTTSProvider()
    timed = _synchronize_with_translation_recovery(
        _synthesized_for_recovery(tmp_path, durations=(2.0, 2.0, 1.0)),
        BatchProvider(), tts,
        LocalizationWorkflowOptions(output_directory=str(tmp_path), timing_policy=TimingPolicy()),
    )

    assert tts.max_active > 1
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
            '"translated_text":"Xin chào mọi người"}]}'
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
    result = localize_video(
        str(source),
        TranscriptionOptions(language="en", task=Task.TRANSCRIBE),
        "unused-model-path", FakeTranslationProvider(), FakeTTSProvider(),
        LocalizationWorkflowOptions(
            output_directory=str(output), chunk_duration_seconds=2,
            render_options=FinalRenderOptions(subtitle_mode="soft"),
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
