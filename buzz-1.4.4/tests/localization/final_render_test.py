import json
from pathlib import Path
import subprocess
import wave

import pytest

from buzz.localization.final_render import (
    FinalRenderError,
    FinalRenderOptions,
    FinalRenderResult,
    blur_bilibili_watermark_video,
    build_bilibili_watermark_filter_graph,
    render_localized_mp4,
    refresh_dubbed_portrait_subtitles,
    subtitle_refresh_output_path,
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
    assert command[command.index("-threads") + 1] == "4"
    vf = command[command.index("-vf") + 1]
    assert vf.startswith("subtitles='")
    assert "viet.srt" in vf
    assert "force_style='WrapStyle=2'" in vf


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


def test_burn_subtitle_can_blur_original_subtitle_area(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()

    render_localized_mp4(
        video,
        audio,
        tmp_path / "final.mp4",
        subtitle_file=subtitle,
        options=FinalRenderOptions(
            subtitle_mode="burn",
            cover_original_subtitles=True,
        ),
        ffmpeg_runner=runner,
    )

    command = runner.commands[0]
    fc = command[command.index("-filter_complex") + 1]
    assert "[0:v]split=2[base][blur_src]" in fc
    assert "crop=iw:ih*0.1800:0:ih*0.8200" in fc
    assert "boxblur=luma_radius=20:luma_power=2" in fc
    assert "[base][blurred]overlay=0:H-h[covered]" in fc
    assert "color=black@0.22:t=fill" in fc
    assert "subtitles='" in fc
    assert "force_style='WrapStyle=2'" in fc
    assert command[command.index("-map") + 1] == "[vout]"

def test_cover_options_are_validated():
    with pytest.raises(ValueError, match="cover_height_ratio"):
        FinalRenderOptions(cover_height_ratio=0)
    with pytest.raises(ValueError, match="cover_height_ratio"):
        FinalRenderOptions(cover_height_ratio=1)
    with pytest.raises(ValueError, match="cover_blur_radius"):
        FinalRenderOptions(cover_blur_radius=0)
    with pytest.raises(ValueError, match="cover_dark_opacity"):
        FinalRenderOptions(cover_dark_opacity=-0.1)
    with pytest.raises(ValueError, match="cover_dark_opacity"):
        FinalRenderOptions(cover_dark_opacity=1.1)


def test_bilibili_watermark_graph_composes_with_bottom_cover_and_subtitles(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()
    render_localized_mp4(
        video, audio, tmp_path / "final.mp4", subtitle_file=subtitle,
        options=FinalRenderOptions(subtitle_mode="burn", cover_original_subtitles=True,
                                   blur_bilibili_watermark=True), ffmpeg_runner=runner,
    )
    graph = runner.commands[0][runner.commands[0].index("-filter_complex") + 1]
    assert "crop=iw*0.2400:ih*0.1500:iw*0.7600:ih*0.0000" in graph
    assert "boxblur=luma_radius=24:luma_power=2" in graph
    assert "drawbox=x=iw*0.7600:y=ih*0.0000:w=iw*0.2400:h=ih*0.1500:color=black@0.18:t=fill" in graph
    assert "crop=iw:ih*0.1800:0:ih*0.8200" in graph
    assert "subtitles='" in graph


def test_ben_review_logo_composes_after_watermark_cover_and_subtitles(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()
    render_localized_mp4(
        video, audio, tmp_path / "final.mp4", subtitle_file=subtitle,
        options=FinalRenderOptions(
            subtitle_mode="burn", cover_original_subtitles=True,
            blur_bilibili_watermark=True, overlay_ben_review_logo=True,
        ), ffmpeg_runner=runner,
    )
    command = runner.commands[0]
    graph = command[command.index("-filter_complex") + 1]
    assert command.count("-filter_complex") == 1
    assert "[2:v]format=rgba[ben_review_logo_rgba]" in graph
    assert "[ben_review_logo_rgba][subtitle_burned]scale2ref=w=oh*main_w/main_h:h=ih*0.115" in graph
    assert "overlay=x=W-w-W*0.01:y=H*0.01:shortest=1:repeatlast=0[vout]" in graph
    assert graph.index("watermark_covered") < graph.index("subtitles=") < graph.index("[ben_review_base][ben_review_logo]overlay")


@pytest.mark.parametrize("mode", ["soft", "none"])
def test_ben_review_logo_reencodes_and_uses_correct_inputs(tmp_path, mode):
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()
    render_localized_mp4(
        video, audio, tmp_path / "final.mp4",
        subtitle_file=subtitle if mode == "soft" else None,
        options=FinalRenderOptions(subtitle_mode=mode, overlay_ben_review_logo=True),
        ffmpeg_runner=runner,
    )
    command = runner.commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert "[ben_review_logo_rgba][0:v]scale2ref" in command[command.index("-filter_complex") + 1]
    if mode == "soft":
        assert "3:0" in command


def test_enabled_ben_review_logo_requires_existing_file(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    with pytest.raises(FinalRenderError, match="Ben review logo file does not exist"):
        render_localized_mp4(
            video, audio, tmp_path / "final.mp4", subtitle_file=subtitle,
            options=FinalRenderOptions(
                overlay_ben_review_logo=True,
                ben_review_logo_path=tmp_path / "missing.png",
            ), ffmpeg_runner=FakeRunner(),
        )


def test_real_ffmpeg_ben_review_logo_is_1080p_and_keeps_audio(tmp_path):
    video, audio, _ = create_inputs(tmp_path)
    output = tmp_path / "logo.mp4"
    _run_raw_ffmpeg([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ])
    _make_wav(audio)
    render_localized_mp4(
        video, audio, output,
        options=FinalRenderOptions(subtitle_mode="none", overlay_ben_review_logo=True, preset="ultrafast"),
    )
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
        "-of", "default=noprint_wrappers=1", str(output),
    ], check=True, capture_output=True, text=True)
    assert "width=1920" in probe.stdout and "height=1080" in probe.stdout
    assert "codec_type=audio" in probe.stdout
    frame = tmp_path / "logo.png"
    _run_raw_ffmpeg(["ffmpeg", "-y", "-i", str(output), "-frames:v", "1", str(frame)])
    from PIL import Image
    # The 11.5%-high logo occupies the upper-right; this pixel is inside it.
    assert Image.open(frame).convert("RGB").getpixel((1810, 70)) != (0, 0, 0)


@pytest.mark.parametrize("mode", ["soft", "none"])
def test_bilibili_watermark_reencodes_video_outside_burn_mode(tmp_path, mode):
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()
    render_localized_mp4(
        video, audio, tmp_path / "final.mp4",
        subtitle_file=subtitle if mode == "soft" else None,
        options=FinalRenderOptions(subtitle_mode=mode, blur_bilibili_watermark=True),
        ffmpeg_runner=runner,
    )
    command = runner.commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert "[vout]" in command


def test_bilibili_watermark_options_and_standalone_command(tmp_path):
    with pytest.raises(ValueError, match="fit within the frame width"):
        FinalRenderOptions(bilibili_watermark_x_ratio=.8, bilibili_watermark_width_ratio=.3)
    with pytest.raises(ValueError, match="bilibili_watermark_blur_radius"):
        FinalRenderOptions(bilibili_watermark_blur_radius=0)
    options = FinalRenderOptions(blur_bilibili_watermark=True)
    assert "[0:v]split=2" in build_bilibili_watermark_filter_graph(options)
    video, _, _ = create_inputs(tmp_path)
    runner = FakeRunner()
    blur_bilibili_watermark_video(video, tmp_path / "blurred.mp4", options=options, ffmpeg_runner=runner)
    assert "-c:a" in runner.commands[0]
    assert runner.commands[0][runner.commands[0].index("-c:a") + 1] == "copy"

def test_portrait_preset_overlays_vietnamese_at_chinese_subtitle_height(tmp_path):
    from buzz.localization.final_render import build_portrait_subtitle_filter_graph
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()
    options = FinalRenderOptions(
        subtitle_mode="burn", cover_original_subtitles=True,
        blur_bilibili_watermark=True, portrait_subtitle_overlay=True,
        overlay_ben_review_logo=True,
    )
    render_localized_mp4(
        video, audio, tmp_path / "portrait.mp4", subtitle_file=subtitle,
        options=options, ffmpeg_runner=runner,
    )
    cmd = runner.commands[0]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph == build_portrait_subtitle_filter_graph(
        graph.split("subtitles='", 1)[1].split("':force_style=", 1)[0],
        overlay_logo=True,
    )
    assert "-vf" not in cmd
    assert "portrait_top_blur" in graph and "portrait_bottom_blur" in graph
    assert "portrait_dialogue_blur" in graph
    assert "crop=iw*0.90:ih*0.06:iw*0.05:ih*0.655" in graph
    assert "overlay=x=W*0.05:y=H*0.655" in graph
    assert "drawbox=x=iw*0.05:y=ih*0.655:w=iw*0.90:h=ih*0.06" in graph
    assert "Alignment=5,MarginV=0,FontSize=32" in graph
    assert "ben_review_logo" in graph
    assert "ih*0.82" not in graph and "watermark_ready" not in graph
    assert graph.index("portrait_dialogue_blur") < graph.index("subtitles=")
    assert graph.index("subtitles=") < graph.index("[ben_review_base][ben_review_logo]overlay")
    assert cmd[cmd.index("-map") + 1] == "[vout]"
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_portrait_preset_requires_burn_and_preserves_landscape_default(tmp_path):
    with pytest.raises(ValueError, match="portrait_subtitle_overlay"):
        FinalRenderOptions(subtitle_mode="soft", portrait_subtitle_overlay=True)
    with pytest.raises(ValueError, match="portrait_subtitle_overlay"):
        FinalRenderOptions(subtitle_mode="none", portrait_subtitle_overlay=True)
    assert FinalRenderOptions().portrait_subtitle_overlay is False
    video, audio, subtitle = create_inputs(tmp_path)
    runner = FakeRunner()
    render_localized_mp4(
        video, audio, tmp_path / "landscape.mp4", subtitle_file=subtitle,
        options=FinalRenderOptions(subtitle_mode="burn", cover_original_subtitles=True),
        ffmpeg_runner=runner,
    )
    graph = runner.commands[0][runner.commands[0].index("-filter_complex") + 1]
    assert "portrait_top_blur" not in graph
    assert "portrait_dialogue_blur" not in graph
    assert "cover" in graph

def test_portrait_one_line_ass_preserves_entire_cues_and_fits_width(tmp_path):
    from buzz.localization.final_render import _portrait_one_line_ass
    source = tmp_path / "source.srt"
    short = "Xin chào!"
    long = "Tôi sẽ đi cùng anh. Nhưng chúng ta phải cẩn thận. Và vẫn còn một câu rất dài."
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n" + short
        + "\n\n2\n00:00:01,000 --> 00:00:03,000\n"
        + "Tôi sẽ đi cùng anh.\n"
        + "Nhưng chúng ta phải cẩn thận. Và vẫn còn một câu rất dài.\n",
        encoding="utf-8",
    )
    derived = _portrait_one_line_ass(source, tmp_path)
    try:
        text = derived.read_text(encoding="utf-8-sig")
        # Legacy full cues are split only in the temporary ASS, not the SRT.
        assert text.count("Dialogue: ") >= 2
        assert short in text
        assert all(word in text for word in long.split())
        assert "\\an5\\pos(540,1315)\\q2" in text  # centered x and y inside blur band
        assert "\\N" not in text
        lines = [line for line in text.splitlines() if line.startswith("Dialogue: ")]
        assert all("\\fs32\\fscx100" in line for line in lines)
        # Every render-only legacy split remains sequential (no overlap/gap).
        times = [line.split(",")[1:3] for line in lines]
        assert all(times[index][1] == times[index + 1][0] for index in range(len(times) - 1))
        assert "1:00:00" not in text  # no accidental hour offset
    finally:
        derived.unlink()


def test_portrait_render_temp_ass_is_deleted_after_runner(tmp_path):
    video, audio, subtitle = create_inputs(tmp_path)
    observed = []
    def runner(cmd):
        graph = cmd[cmd.index("-filter_complex") + 1]
        from pathlib import Path
        import re
        match = re.search(r"subtitles='([^']+\.ass)'", graph)
        assert match is not None
        derived_path = Path(match.group(1).replace(r"\:", ":"))
        assert derived_path.is_file()
        assert "Dialogue:" in derived_path.read_text(encoding="utf-8-sig")
        observed.append(derived_path)
        Path(cmd[-1]).write_bytes(b"fake mp4")
    render_localized_mp4(
        video, audio, tmp_path / "output.mp4", subtitle_file=subtitle,
        options=FinalRenderOptions(subtitle_mode="burn", portrait_subtitle_overlay=True),
        ffmpeg_runner=runner,
    )
    assert len(observed) == 1 and not observed[0].exists()


def test_subtitle_refresh_maps_original_audio_and_only_dialogue_band(tmp_path):
    video = tmp_path / "source.vi.mp4"
    subtitle = tmp_path / "source.vi.srt"
    output = subtitle_refresh_output_path(video)
    video.write_bytes(b"source")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao Viet Nam\n", encoding="utf-8")
    commands = []

    def probe(_):
        return subprocess.CompletedProcess([], 0, '{"streams":[{"width":1080,"height":1920}],"format":{"duration":"904.0"}}', "")

    def runner(command, *, cancel_event=None):
        commands.append(command)
        Path(command[-1]).write_bytes(b"refreshed")

    result = refresh_dubbed_portrait_subtitles(
        video, output, subtitle_file=subtitle, ffmpeg_runner=runner, ffprobe_runner=probe,
    )

    command = commands[0]
    graph = command[command.index("-filter_complex") + 1]
    assert result.video_file == str(output)
    assert command[command.index("-map") + 1] == "[vout]"
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "0:a:0"
    assert command[command.index("-c:a") + 1] == "copy"
    assert "crop=iw*0.90:ih*0.06:iw*0.05:ih*0.655" in graph
    assert "portrait_top" not in graph and "portrait_bottom" not in graph and "ben_review" not in graph
    assert output.is_file()
    assert not list(tmp_path.glob("portrait_one_line_*.ass"))


def test_subtitle_refresh_rejects_existing_output_and_cleans_partial_on_cancel(tmp_path):
    from buzz.localization.audio_mix import AudioMixingCancelled
    video = tmp_path / "source.vi.mp4"
    subtitle = tmp_path / "source.vi.srt"
    output = subtitle_refresh_output_path(video)
    video.write_bytes(b"source")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
    def probe(_):
        return subprocess.CompletedProcess([], 0, '{"streams":[{"width":1080,"height":1920}],"format":{"duration":"1.0"}}', "")
    def cancelled(command, *, cancel_event=None):
        Path(command[-1]).write_bytes(b"partial")
        raise AudioMixingCancelled("cancelled")
    with pytest.raises(AudioMixingCancelled):
        refresh_dubbed_portrait_subtitles(video, output, subtitle_file=subtitle, ffmpeg_runner=cancelled, ffprobe_runner=probe)
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial.mp4"))
    output.write_bytes(b"do not overwrite")
    with pytest.raises(FinalRenderError, match="already exists"):
        refresh_dubbed_portrait_subtitles(video, output, subtitle_file=subtitle, ffprobe_runner=probe)
