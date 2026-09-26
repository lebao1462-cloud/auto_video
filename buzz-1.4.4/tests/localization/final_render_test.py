import json
from pathlib import Path
import subprocess
import wave

import pytest

from buzz.localization.final_render import (
    FinalRenderError,
    FinalRenderOptions,
    FinalRenderResult,
    render_localized_mp4,
)


class FakeRunner:
    def __init__(self, create_output=True, fail=False):
        self.commands = []
        self.create_output = create_output
        self.fail = fail

    def __call__(self, command):
        self.commands.append(command)
        if self.fail:
            Path(command[-1]).write_bytes(b"partial")
            raise FinalRenderError("render failed")
        if self.create_output:
            Path(command[-1]).write_bytes(b"fake mp4")


def create_inputs(tmp_path):
    video = tmp_path / "source.mp4"
    audio = tmp_path / "localized.wav"
    subtitle = tmp_path / "viet.srt"
    video.write_bytes(b"fake video")
    audio.write_bytes(b"fake audio")
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nXin chào\n",
        encoding="utf-8",
    )
    return video, audio, subtitle


def test_soft_subtitle_command_copies_video_and_encodes_aac(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()

    result = render_localized_mp4(
        video,
        audio,
        tmp_path / "final.mp4",
        subtitle_file=subtitle,
        options=FinalRenderOptions(subtitle_mode="soft"),
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert "mov_text" in command
    assert "language=vie" in command
    assert result == FinalRenderResult(
        video_file=str(tmp_path / "final.mp4"),
        subtitle_mode="soft",
        audio_replaced=True,
        subtitle_included=True,
    )


def test_burn_subtitle_command_reencodes_video(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()

    render_localized_mp4(
        video,
        audio,
        tmp_path / "final.mp4",
        subtitle_file=subtitle,
        options=FinalRenderOptions(
            subtitle_mode="burn",
            video_codec="libx264",
            crf=20,
            preset="fast",
        ),
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "20"
    assert command[command.index("-preset") + 1] == "fast"
    vf = command[command.index("-vf") + 1]
    assert vf.startswith("subtitles='")
    assert "viet.srt" in vf


def test_no_subtitle_mode_does_not_require_subtitle(tmp_path):
    video, audio, _ = create_inputs(tmp_path)
    runner = FakeRunner()

    result = render_localized_mp4(
        video,
        audio,
        tmp_path / "final.mp4",
        options=FinalRenderOptions(subtitle_mode="none"),
        ffmpeg_runner=runner,
    )

    assert result.subtitle_included is False
    assert "-c:s" not in runner.commands[0]


@pytest.mark.parametrize("mode", ["soft", "burn"])
def test_subtitle_modes_require_subtitle_file(mode, tmp_path):
    video, audio, _ = create_inputs(tmp_path)

    with pytest.raises(FinalRenderError, match="requires a subtitle"):
        render_localized_mp4(
            video,
            audio,
            tmp_path / "final.mp4",
            options=FinalRenderOptions(subtitle_mode=mode),
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_missing_source_video(tmp_path):
    _, audio, subtitle = create_inputs(tmp_path)

    with pytest.raises(FinalRenderError, match="Source video"):
        render_localized_mp4(
            tmp_path / "missing.mp4",
            audio,
            tmp_path / "final.mp4",
            subtitle_file=subtitle,
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_missing_localized_audio(tmp_path):
    video, _, subtitle = create_inputs(tmp_path)

    with pytest.raises(FinalRenderError, match="Localized audio"):
        render_localized_mp4(
            video,
            tmp_path / "missing.wav",
            tmp_path / "final.mp4",
            subtitle_file=subtitle,
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_missing_subtitle(tmp_path):
    video, audio, _ = create_inputs(tmp_path)

    with pytest.raises(FinalRenderError, match="Subtitle file"):
        render_localized_mp4(
            video,
            audio,
            tmp_path / "final.mp4",
            subtitle_file=tmp_path / "missing.srt",
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_unsupported_subtitle_extension(tmp_path):
    video, audio, _ = create_inputs(tmp_path)
    subtitle = tmp_path / "viet.ass"
    subtitle.write_text("fake", encoding="utf-8")

    with pytest.raises(FinalRenderError, match="SRT or VTT"):
        render_localized_mp4(
            video,
            audio,
            tmp_path / "final.mp4",
            subtitle_file=subtitle,
            ffmpeg_runner=FakeRunner(),
        )


def test_rejects_non_mp4_output(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)

    with pytest.raises(FinalRenderError, match=".mp4"):
        render_localized_mp4(
            video,
            audio,
            tmp_path / "final.mkv",
            subtitle_file=subtitle,
            ffmpeg_runner=FakeRunner(),
        )


def test_never_overwrites_source_video(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)

    with pytest.raises(FinalRenderError, match="must not overwrite"):
        render_localized_mp4(
            video,
            audio,
            video,
            subtitle_file=subtitle,
            ffmpeg_runner=FakeRunner(),
        )

    assert video.read_bytes() == b"fake video"


def test_failure_removes_partial_output(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    output = tmp_path / "final.mp4"

    with pytest.raises(FinalRenderError, match="render failed"):
        render_localized_mp4(
            video,
            audio,
            output,
            subtitle_file=subtitle,
            ffmpeg_runner=FakeRunner(fail=True),
        )

    assert not output.exists()


def test_rejects_runner_that_does_not_create_output(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)

    with pytest.raises(FinalRenderError, match="did not create"):
        render_localized_mp4(
            video,
            audio,
            tmp_path / "final.mp4",
            subtitle_file=subtitle,
            ffmpeg_runner=FakeRunner(create_output=False),
        )


@pytest.mark.parametrize(
    "options",
    [
        lambda: FinalRenderOptions(subtitle_mode="invalid"),
        lambda: FinalRenderOptions(crf=-1),
        lambda: FinalRenderOptions(crf=52),
        lambda: FinalRenderOptions(video_codec=""),
        lambda: FinalRenderOptions(audio_codec=""),
        lambda: FinalRenderOptions(audio_bitrate=""),
    ],
)
def test_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        options()


def test_result_is_json_compatible(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)

    result = render_localized_mp4(
        video,
        audio,
        tmp_path / "final.mp4",
        subtitle_file=subtitle,
        ffmpeg_runner=FakeRunner(),
    )

    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def _make_wav(path: Path, duration=1.0, sample_rate=16000):
    frames = int(duration * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _run_raw_ffmpeg(command):
    subprocess.run(command, check=True, capture_output=True)


def test_real_ffmpeg_renders_mp4_with_soft_subtitle(tmp_path):
    video = tmp_path / "source.mp4"
    audio = tmp_path / "localized.wav"
    subtitle = tmp_path / "viet.srt"
    output = tmp_path / "final.mp4"

    _run_raw_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    _make_wav(audio, duration=1.0)
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:00,900\nXin chào Việt Nam\n",
        encoding="utf-8",
    )

    result = render_localized_mp4(
        video,
        audio,
        output,
        subtitle_file=subtitle,
        options=FinalRenderOptions(subtitle_mode="soft"),
    )

    assert Path(result.video_file).is_file()
    assert Path(result.video_file).stat().st_size > 0

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "default=noprint_wrappers=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "codec_type=video" in probe.stdout
    assert "codec_type=audio" in probe.stdout
    assert "codec_type=subtitle" in probe.stdout
    assert "codec_name=mov_text" in probe.stdout


def test_real_ffmpeg_renders_mp4_with_burned_subtitle(tmp_path):
    video = tmp_path / "source-burn.mp4"
    audio = tmp_path / "localized-burn.wav"
    subtitle = tmp_path / "viet-burn.srt"
    output = tmp_path / "final-burn.mp4"

    _run_raw_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    _make_wav(audio, duration=1.0)
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:00,900\nXin chào Việt Nam\n",
        encoding="utf-8",
    )

    result = render_localized_mp4(
        video,
        audio,
        output,
        subtitle_file=subtitle,
        options=FinalRenderOptions(subtitle_mode="burn", preset="ultrafast"),
    )

    assert Path(result.video_file).is_file()
    assert Path(result.video_file).stat().st_size > 0

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "default=noprint_wrappers=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "codec_type=video" in probe.stdout
    assert "codec_type=audio" in probe.stdout
    assert "codec_type=subtitle" not in probe.stdout
    assert result.subtitle_mode == "burn"
    assert result.subtitle_included is True
