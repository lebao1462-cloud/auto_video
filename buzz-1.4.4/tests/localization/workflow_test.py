from pathlib import Path
import subprocess
import threading
import wave

import pytest

from buzz.localization.transcript import LocalizationSegment, LocalizationTranscript
from buzz.localization.translation import TranslationProvider
from buzz.localization.tts import TTSRequest, TTSResult
from buzz.localization.workflow import (
    LocalizationCancelled,
    LocalizationStage,
    LocalizationWorkflowOptions,
    cleanup_localization_workspace,
    localize_video,
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


def create_test_video(path: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
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
