from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable

from buzz.localization.audio_mix import AudioMixingCancelled, _run_ffmpeg
from buzz.localization.subtitles import SubtitleGenerationError, split_portrait_cue


class FinalRenderError(RuntimeError):
    """Raised when the final localized MP4 cannot be rendered."""


@dataclass(frozen=True)
class FinalRenderOptions:
    subtitle_mode: str = "soft"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    crf: int = 18
    preset: str = "medium"
    cover_original_subtitles: bool = False
    portrait_subtitle_overlay: bool = False
    cover_height_ratio: float = 0.18
    cover_blur_radius: int = 20
    cover_dark_opacity: float = 0.22
    blur_bilibili_watermark: bool = False
    overlay_ben_review_logo: bool = False
    ben_review_logo_path: str | Path | None = None
    bilibili_watermark_x_ratio: float = 0.76
    bilibili_watermark_y_ratio: float = 0.0
    bilibili_watermark_width_ratio: float = 0.24
    bilibili_watermark_height_ratio: float = 0.15
    bilibili_watermark_blur_radius: int = 24
    bilibili_watermark_dark_opacity: float = 0.18

    def __post_init__(self):
        if self.subtitle_mode not in {"soft", "burn", "none"}:
            raise ValueError("subtitle_mode must be 'soft', 'burn', or 'none'")
        if self.portrait_subtitle_overlay and self.subtitle_mode != "burn":
            raise ValueError("portrait_subtitle_overlay requires subtitle_mode=burn")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        if not self.video_codec:
            raise ValueError("video_codec is required")
        if not self.audio_codec:
            raise ValueError("audio_codec is required")
        if not self.audio_bitrate:
            raise ValueError("audio_bitrate is required")
        if not 0 < self.cover_height_ratio < 1:
            raise ValueError("cover_height_ratio must be between 0 and 1")
        if self.cover_blur_radius < 1:
            raise ValueError("cover_blur_radius must be at least 1")
        if not 0 <= self.cover_dark_opacity <= 1:
            raise ValueError("cover_dark_opacity must be between 0 and 1")
        if not 0 <= self.bilibili_watermark_x_ratio < 1:
            raise ValueError("bilibili_watermark_x_ratio must be between 0 and 1")
        if not 0 <= self.bilibili_watermark_y_ratio < 1:
            raise ValueError("bilibili_watermark_y_ratio must be between 0 and 1")
        if not 0 < self.bilibili_watermark_width_ratio <= 1:
            raise ValueError("bilibili_watermark_width_ratio must be between 0 and 1")
        if not 0 < self.bilibili_watermark_height_ratio <= 1:
            raise ValueError("bilibili_watermark_height_ratio must be between 0 and 1")
        if self.bilibili_watermark_x_ratio + self.bilibili_watermark_width_ratio > 1:
            raise ValueError("bilibili watermark region must fit within the frame width")
        if self.bilibili_watermark_y_ratio + self.bilibili_watermark_height_ratio > 1:
            raise ValueError("bilibili watermark region must fit within the frame height")
        if self.bilibili_watermark_blur_radius < 1:
            raise ValueError("bilibili_watermark_blur_radius must be at least 1")
        if not 0 <= self.bilibili_watermark_dark_opacity <= 1:
            raise ValueError("bilibili_watermark_dark_opacity must be between 0 and 1")


@dataclass(frozen=True)
class FinalRenderResult:
    video_file: str
    subtitle_mode: str
    audio_replaced: bool
    subtitle_included: bool

    def to_dict(self) -> dict:
        return {
            "video_file": self.video_file,
            "subtitle_mode": self.subtitle_mode,
            "audio_replaced": self.audio_replaced,
            "subtitle_included": self.subtitle_included,
        }


def _escape_subtitle_filter_path(path: Path) -> str:
    escaped = str(path.resolve()).replace("\\", "/")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("[", "\\[")
    escaped = escaped.replace("]", "\\]")
    escaped = escaped.replace(",", "\\,")
    return escaped


def default_ben_review_logo_path() -> Path:
    """Return the bundled Ben review logo independently of the current directory."""
    return Path(__file__).resolve().parent.parent / "assets" / "branding" / "ben_review_logo.png"


def _ben_review_logo_path(options: FinalRenderOptions) -> Path:
    path = Path(options.ben_review_logo_path) if options.ben_review_logo_path else default_ben_review_logo_path()
    if not path.is_file():
        raise FinalRenderError(f"Ben review logo file does not exist: {path}")
    return path


def build_ben_review_logo_filter_graph(
    *, input_label: str, logo_input_label: str, output_label: str = "vout",
) -> str:
    """Scale the logo to 11.5% of video height and place it at the top-right."""
    return (
        f"[{logo_input_label}]format=rgba[ben_review_logo_rgba];"
        f"[ben_review_logo_rgba][{input_label}]"
        "scale2ref=w=oh*main_w/main_h:h=ih*0.115[ben_review_logo][ben_review_base];"
        "[ben_review_base][ben_review_logo]"
        f"overlay=x=W-w-W*0.01:y=H*0.01:shortest=1:repeatlast=0[{output_label}]"
    )


def build_bilibili_watermark_filter_graph(
    options: FinalRenderOptions,
    *,
    input_label: str = "0:v",
    output_label: str = "vout",
) -> str:
    """Return an FFmpeg graph that blurs and lightly darkens the Bilibili region."""
    if not isinstance(options, FinalRenderOptions):
        raise TypeError("options must be FinalRenderOptions")
    x = options.bilibili_watermark_x_ratio
    y = options.bilibili_watermark_y_ratio
    width = options.bilibili_watermark_width_ratio
    height = options.bilibili_watermark_height_ratio
    return (
        f"[{input_label}]split=2[watermark_base][watermark_src];"
        "[watermark_src]"
        f"crop=iw*{width:.4f}:ih*{height:.4f}:iw*{x:.4f}:ih*{y:.4f},"
        f"boxblur=luma_radius={options.bilibili_watermark_blur_radius}:"
        "luma_power=2[watermark_blurred];"
        f"[watermark_base][watermark_blurred]overlay=W-w*{(1 - x) / width:.4f}:H-h*{(1 - y) / height:.4f}[watermark_covered];"
        "[watermark_covered]"
        f"drawbox=x=iw*{x:.4f}:y=ih*{y:.4f}:"
        f"w=iw*{width:.4f}:h=ih*{height:.4f}:"
        f"color=black@{options.bilibili_watermark_dark_opacity:.2f}:t=fill"
        f"[{output_label}]"
    )



_PORTRAIT_CUE_TIMING = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def _format_srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _portrait_one_line_ass(subtitle_path: Path, output_directory: Path) -> Path:
    """Create render-only ASS subtitles with readable sequential Vietnamese lines.

    The original SRT/VTT stays untouched. Long legacy cues are split with the
    same Arial metric used for newly written portrait SRT files.
    """
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise FinalRenderError("Portrait single-line subtitles require Pillow") from exc

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if not font_path.is_file():
        raise FinalRenderError("Portrait one-line subtitle font is missing: Arial")
    source = subtitle_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    blocks = re.split(r"\n[ \t]*\n", source.strip())
    events = []
    ImageFont.truetype(str(font_path), 32)  # verify the renderer's metric font

    def timestamp(text: str) -> str:
        hours, minutes, rest = text.replace(",", ".").split(":")
        secs, millis = rest.split(".")
        hundredths = round(int(millis) / 10)
        total = ((int(hours) * 60 + int(minutes)) * 60 + int(secs)) * 100 + hundredths
        return f"{total // 360000}:{(total // 6000) % 60:02d}:{(total // 100) % 60:02d}.{total % 100:02d}"

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((i for i, line in enumerate(lines) if _PORTRAIT_CUE_TIMING.match(line)), None)
        if time_index is None:
            continue
        match = _PORTRAIT_CUE_TIMING.match(lines[time_index])
        assert match is not None
        cue = " ".join(" ".join(lines[time_index + 1:]).split())
        if not cue:
            continue
        def seconds(value: str) -> float:
            hours, minutes, rest = value.replace(",", ".").split(":")
            secs, millis = rest.split(".")
            return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000
        try:
            cue_parts = split_portrait_cue(cue, seconds(match.group("start")), seconds(match.group("end")))
        except SubtitleGenerationError as exc:
            raise FinalRenderError(str(exc)) from exc
        for part_start, part_end, part_text in cue_parts:
            # The shared splitter guarantees Arial 10 fits 140 PlayRes pixels.
            # Keep scale at 100: no compressed or overlapping-looking glyphs.
            display = part_text.replace("\\", "\\\\").replace("{", r"\{").replace("}", r"\}")
            display = f"{{\\an5\\pos(540,1315)\\q2\\fs32\\fscx100}}" + display
            events.append(
                f"Dialogue: 0,{timestamp(_format_srt_time(part_start))},"
                f"{timestamp(_format_srt_time(part_end))},Default,,0,0,0,,{display}"
            )
    if not events:
        raise FinalRenderError("Portrait single-line subtitles have no valid cues")
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,32,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,1,0,2,12,12,80,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    # Create an auxiliary ASS only for FFmpeg; remove it after rendering.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", suffix=".ass", prefix="portrait_one_line_",
        dir=output_directory, delete=False,
    ) as temporary:
        temporary.write(header + "\n".join(events) + "\n")
        return Path(temporary.name)


def build_portrait_subtitle_filter_graph(
    escaped_subtitle: str, *, overlay_logo: bool = False,
) -> str:
    """Blur portrait source labels/dialogue, then burn Vietnamese exactly over dialogue.

    For 9:16 shorts with source dialogue at 68-71% of frame height.
    Filter position and font size were verified on the user's 1080x1920 source.
    """
    out_label = "portrait_subtitle_burned" if overlay_logo else "vout"
    graph = (
        "[0:v]split=2[portrait_top_base][portrait_top_src];"
        "[portrait_top_src]crop=iw*0.37:ih*0.063:iw*0.005:ih*0.012,"
        "boxblur=luma_radius=22:luma_power=2[portrait_top_blur];"
        "[portrait_top_base][portrait_top_blur]"
        "overlay=x=W*0.005:y=H*0.012[portrait_top_done];"
        "[portrait_top_done]drawbox=x=iw*0.005:y=ih*0.012:"
        "w=iw*0.37:h=ih*0.063:color=black@0.15:t=fill[portrait_top_masked];"

        "[portrait_top_masked]split=2[portrait_bottom_base][portrait_bottom_src];"
        "[portrait_bottom_src]crop=iw*0.57:ih*0.067:iw*0.35:ih*0.93,"
        "boxblur=luma_radius=22:luma_power=2[portrait_bottom_blur];"
        "[portrait_bottom_base][portrait_bottom_blur]"
        "overlay=x=W*0.35:y=H*0.93[portrait_bottom_done];"
        "[portrait_bottom_done]drawbox=x=iw*0.35:y=ih*0.93:"
        "w=iw*0.57:h=ih*0.067:color=black@0.15:t=fill[portrait_bottom_masked];"

        "[portrait_bottom_masked]split=2[portrait_dialogue_base][portrait_dialogue_src];"
        "[portrait_dialogue_src]crop=iw*0.90:ih*0.06:iw*0.05:ih*0.655,"
        "boxblur=luma_radius=22:luma_power=2[portrait_dialogue_blur];"
        "[portrait_dialogue_base][portrait_dialogue_blur]"
        "overlay=x=W*0.05:y=H*0.655[portrait_dialogue_done];"
        "[portrait_dialogue_done]drawbox=x=iw*0.05:y=ih*0.655:"
        "w=iw*0.90:h=ih*0.06:color=black@0.16:t=fill,"
        f"subtitles='{escaped_subtitle}':force_style="
        f"'Alignment=5,MarginV=0,FontSize=32,Outline=2,Shadow=0,WrapStyle=2'"
        f"[{out_label}]"
    )
    if overlay_logo:
        graph += ";" + build_ben_review_logo_filter_graph(
            input_label=out_label, logo_input_label="2:v"
        )
    return graph


def subtitle_refresh_output_path(source_video_file: str | Path) -> Path:
    """Return the safe, never-in-place destination for a subtitle refresh."""
    source = Path(source_video_file)
    return source.with_name(f"{source.stem}.vietsub-fixed.mp4")


def build_portrait_subtitle_refresh_filter_graph(escaped_subtitle: str) -> str:
    """Replace only the already-burned portrait dialogue band.

    This intentionally does not apply the normal portrait top/bottom covers or
    Ben logo: those are already baked into the source part and repeating them
    would visibly double-brand it.
    """
    return (
        "[0:v]split=2[refresh_base][refresh_dialogue_src];"
        "[refresh_dialogue_src]crop=iw*0.90:ih*0.06:iw*0.05:ih*0.655,"
        "boxblur=luma_radius=22:luma_power=2[refresh_dialogue_blur];"
        "[refresh_base][refresh_dialogue_blur]"
        "overlay=x=W*0.05:y=H*0.655[refresh_dialogue_done];"
        "[refresh_dialogue_done]drawbox=x=iw*0.05:y=ih*0.655:"
        "w=iw*0.90:h=ih*0.06:color=black@0.16:t=fill,"
        f"subtitles='{escaped_subtitle}':force_style="
        "'Alignment=5,MarginV=0,FontSize=32,Outline=2,Shadow=0,WrapStyle=2'"
        "[vout]"
    )


def _probe_refresh_video(
    media_file: Path,
    *,
    ffprobe_runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> tuple[int, int, float]:
    runner = ffprobe_runner or (lambda command: subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    ))
    completed = runner([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", str(media_file),
    ])
    if completed.returncode != 0:
        raise FinalRenderError(f"ffprobe could not inspect video: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        return int(stream["width"]), int(stream["height"]), float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FinalRenderError("ffprobe did not return valid video dimensions and duration") from exc


def _probe_refresh_audio(
    media_file: Path,
    *,
    ffprobe_runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> None:
    runner = ffprobe_runner or (lambda command: subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    ))
    completed = runner([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=codec_type", "-of", "json", str(media_file),
    ])
    try:
        if completed.returncode != 0 or not json.loads(completed.stdout).get("streams"):
            raise FinalRenderError("Subtitle refresh output has no audio stream")
    except json.JSONDecodeError as exc:
        raise FinalRenderError("ffprobe did not return valid audio stream data") from exc


def refresh_dubbed_portrait_subtitles(
    source_video_file: str | Path,
    output_file: str | Path,
    *,
    subtitle_file: str | Path | None = None,
    cancel_event=None,
    preview_duration: float | None = None,
    ffmpeg_runner: Callable[..., None] = _run_ffmpeg,
    ffprobe_runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> FinalRenderResult:
    """Refresh burned Vietnamese portrait subtitles without touching dubbed audio.

    ``preview_duration`` makes a safe beginning-of-part review clip through
    the same application renderer; it never changes the source or full output.
    """
    source, output = Path(source_video_file), Path(output_file)
    subtitle = Path(subtitle_file) if subtitle_file else source.with_suffix(".srt")
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise FinalRenderError("Subtitle refresh source must be an existing MP4")
    if not subtitle.is_file() or subtitle.suffix.lower() not in {".srt", ".vtt"}:
        raise FinalRenderError("Subtitle refresh requires an existing adjacent or selected SRT/VTT")
    if output.suffix.lower() != ".mp4":
        raise FinalRenderError("Subtitle refresh output must use the .mp4 extension")
    if source.resolve() == output.resolve():
        raise FinalRenderError("Subtitle refresh output must not overwrite the source video")
    if output.exists():
        raise FinalRenderError(f"Subtitle refresh output already exists: {output}")
    width, height, source_duration = _probe_refresh_video(source, ffprobe_runner=ffprobe_runner)
    if (width, height) != (1080, 1920):
        raise FinalRenderError(
            f"Subtitle refresh requires a 1080x1920 portrait part, got {width}x{height}"
        )
    if source_duration <= 0:
        raise FinalRenderError("Subtitle refresh source has no positive duration")
    if preview_duration is not None and (preview_duration <= 0 or preview_duration > source_duration):
        raise FinalRenderError("Subtitle refresh preview duration must fit within the source")
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    if partial.exists():
        # A process interrupted before FFmpeg writes its header can leave an
        # empty sentinel.  It is never a usable video, so clear only that
        # exact zero-byte temporary name; retain any material partial for the
        # user to inspect instead of overwriting it.
        if partial.stat().st_size == 0:
            partial.unlink()
        else:
            raise FinalRenderError(f"Subtitle refresh partial output already exists: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_ass = _portrait_one_line_ass(subtitle, output.parent)
    command = [
        "ffmpeg", "-y", "-nostdin", "-i", str(source), "-filter_complex",
        build_portrait_subtitle_refresh_filter_graph(_escape_subtitle_filter_path(temporary_ass)),
        "-map", "[vout]", "-map", "0:a:0", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-threads", "4", "-c:a", "copy", "-map_metadata", "0",
        "-movflags", "+faststart",
    ]
    if preview_duration is not None:
        command.extend(["-t", f"{preview_duration:.3f}"])
    command.append(str(partial))
    try:
        ffmpeg_runner(command, cancel_event=cancel_event)
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise FinalRenderError("FFmpeg did not create a non-empty subtitle refresh MP4")
        refreshed_width, refreshed_height, refreshed_duration = _probe_refresh_video(
            partial, ffprobe_runner=ffprobe_runner
        )
        _probe_refresh_audio(partial, ffprobe_runner=ffprobe_runner)
        if (refreshed_width, refreshed_height) != (width, height):
            raise FinalRenderError("Subtitle refresh changed the video dimensions")
        expected_duration = preview_duration if preview_duration is not None else source_duration
        if abs(refreshed_duration - expected_duration) > 0.25:
            raise FinalRenderError("Subtitle refresh changed the video duration")
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        temporary_ass.unlink(missing_ok=True)
    return FinalRenderResult(str(output), "burn", False, True)


def blur_bilibili_watermark_video(
    source_video_file: str | Path,
    output_file: str | Path,
    *,
    options: FinalRenderOptions | None = None,
    ffmpeg_runner: Callable[[list[str]], None] = _run_ffmpeg,
) -> None:
    """Apply the reusable Bilibili-region blur while stream-copying audio."""
    render_options = FinalRenderOptions() if options is None else options
    source_path, output_path = Path(source_video_file), Path(output_file)
    if not source_path.is_file():
        raise FinalRenderError(f"Source video file does not exist: {source_video_file}")
    if output_path.suffix.lower() != ".mp4":
        raise FinalRenderError("Watermark-blur output file must use the .mp4 extension")
    try:
        if source_path.resolve() == output_path.resolve():
            raise FinalRenderError("Watermark-blur output must not overwrite the source video")
    except OSError:
        pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_bilibili_watermark_filter_graph(render_options)
    ffmpeg_runner(["ffmpeg", "-y", "-nostdin", "-i", str(source_path),
                   "-filter_complex", graph, "-map", "[vout]", "-map", "0:a?",
                   "-c:v", render_options.video_codec, "-preset", render_options.preset,
                   "-crf", str(render_options.crf), "-c:a", "copy", "-movflags",
                   "+faststart", str(output_path)])


def render_localized_mp4(
    source_video_file: str | Path,
    localized_audio_file: str | Path,
    output_file: str | Path,
    *,
    subtitle_file: str | Path | None = None,
    options: FinalRenderOptions | None = None,
    ffmpeg_runner: Callable[[list[str]], None] = _run_ffmpeg,
) -> FinalRenderResult:
    """Render the final localized MP4 from source video, localized audio, and subtitles."""
    render_options = FinalRenderOptions() if options is None else options
    if not isinstance(render_options, FinalRenderOptions):
        raise TypeError("options must be FinalRenderOptions")

    source_path = Path(source_video_file)
    audio_path = Path(localized_audio_file)
    output_path = Path(output_file)

    if not source_path.is_file():
        raise FinalRenderError(f"Source video file does not exist: {source_video_file}")
    if not audio_path.is_file():
        raise FinalRenderError(
            f"Localized audio file does not exist: {localized_audio_file}"
        )
    if output_path.suffix.lower() != ".mp4":
        raise FinalRenderError("Final output file must use the .mp4 extension")

    try:
        if source_path.resolve() == output_path.resolve():
            raise FinalRenderError("Final output must not overwrite the source video")
    except OSError:
        pass

    subtitle_path = None
    if subtitle_file is not None:
        subtitle_path = Path(subtitle_file)
        if not subtitle_path.is_file():
            raise FinalRenderError(f"Subtitle file does not exist: {subtitle_file}")
        if subtitle_path.suffix.lower() not in {".srt", ".vtt"}:
            raise FinalRenderError("Subtitle file must be SRT or VTT")
    elif render_options.subtitle_mode != "none":
        raise FinalRenderError(
            f"subtitle_mode '{render_options.subtitle_mode}' requires a subtitle file"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(source_path),
        "-i",
        str(audio_path),
    ]
    logo_path = _ben_review_logo_path(render_options) if render_options.overlay_ben_review_logo else None
    if logo_path is not None:
        # Loop the still logo for the source video duration; the mapped video
        # stream controls output duration, not this auxiliary input.
        command.extend(["-loop", "1", "-i", str(logo_path)])

    subtitle_included = False
    portrait_ass_path: Path | None = None

    if render_options.subtitle_mode == "soft":
        assert subtitle_path is not None
        command.extend(["-i", str(subtitle_path)])
        video_map = "0:v:0"
        if render_options.blur_bilibili_watermark or logo_path is not None:
            graph = ""
            source_label = "0:v"
            if render_options.blur_bilibili_watermark:
                graph = build_bilibili_watermark_filter_graph(
                    render_options, output_label="watermark_ready"
                )
                source_label = "watermark_ready"
            if logo_path is not None:
                if graph:
                    graph += ";"
                graph += build_ben_review_logo_filter_graph(
                    input_label=source_label, logo_input_label="2:v"
                )
            else:
                graph = build_bilibili_watermark_filter_graph(render_options)
            command.extend(["-filter_complex", graph])
            video_map = "[vout]"
        subtitle_input = "3:0" if logo_path is not None else "2:0"
        command.extend(
            [
                "-map",
                video_map,
                "-map",
                "1:a:0",
                "-map",
                subtitle_input,
                "-c:v",
                render_options.video_codec if (render_options.blur_bilibili_watermark or logo_path is not None) else "copy",
                "-c:a",
                render_options.audio_codec,
                "-b:a",
                render_options.audio_bitrate,
                "-c:s",
                "mov_text",
                "-metadata:s:s:0",
                "language=vie",
                "-disposition:s:0",
                "default",
            ]
        )
        if render_options.blur_bilibili_watermark or logo_path is not None:
            command.extend(["-preset", render_options.preset, "-crf", str(render_options.crf), "-threads", "4"])
        subtitle_included = True
    elif render_options.subtitle_mode == "burn":
        assert subtitle_path is not None
        escaped_subtitle = _escape_subtitle_filter_path(subtitle_path)
        video_filter = (
            f"subtitles='{escaped_subtitle}':force_style='WrapStyle=2'"
        )
        video_map = "0:v:0"
        if render_options.portrait_subtitle_overlay:
            portrait_ass_path = _portrait_one_line_ass(subtitle_path, output_path.parent)
            escaped_subtitle = _escape_subtitle_filter_path(portrait_ass_path)
            filter_complex = build_portrait_subtitle_filter_graph(
                escaped_subtitle, overlay_logo=logo_path is not None,
            )
            command.extend(["-filter_complex", filter_complex])
            video_map = "[vout]"
            video_filter = None
        elif render_options.cover_original_subtitles:
            cover_start = 1.0 - render_options.cover_height_ratio
            source_label = "0:v"
            watermark_prefix = ""
            if render_options.blur_bilibili_watermark:
                watermark_prefix = build_bilibili_watermark_filter_graph(
                    render_options, output_label="watermark_ready"
                ) + ";"
                source_label = "watermark_ready"
            subtitle_output = "subtitle_burned" if logo_path is not None else "vout"
            filter_complex = (
                watermark_prefix +
                f"[{source_label}]split=2[base][blur_src];"
                "[blur_src]"
                f"crop=iw:ih*{render_options.cover_height_ratio:.4f}:"
                f"0:ih*{cover_start:.4f},"
                f"boxblur=luma_radius={render_options.cover_blur_radius}:"
                "luma_power=2[blurred];"
                "[base][blurred]overlay=0:H-h[covered];"
                "[covered]"
                f"drawbox=x=0:y=ih*{cover_start:.4f}:"
                f"w=iw:h=ih*{render_options.cover_height_ratio:.4f}:"
                f"color=black@{render_options.cover_dark_opacity:.2f}:t=fill,"
                f"subtitles='{escaped_subtitle}':force_style='WrapStyle=2'[{subtitle_output}]"
            )
            if logo_path is not None:
                filter_complex += ";" + build_ben_review_logo_filter_graph(
                    input_label="subtitle_burned", logo_input_label="2:v"
                )
            command.extend(["-filter_complex", filter_complex])
            video_map = "[vout]"
            video_filter = None
        elif render_options.blur_bilibili_watermark or logo_path is not None:
            watermark_prefix = ""
            source_label = "0:v"
            if render_options.blur_bilibili_watermark:
                watermark_prefix = build_bilibili_watermark_filter_graph(
                    render_options, output_label="watermark_ready"
                ) + ";"
                source_label = "watermark_ready"
            subtitle_label = "subtitle_burned"
            if logo_path is not None:
                filter_complex = watermark_prefix + f"[{source_label}]" + video_filter + f"[{subtitle_label}];" + build_ben_review_logo_filter_graph(input_label=subtitle_label, logo_input_label="2:v")
            else:
                filter_complex = watermark_prefix + f"[{source_label}]" + video_filter + "[vout]"
            command.extend(["-filter_complex", filter_complex])
            video_map = "[vout]"
            video_filter = None
        command.extend(
            [
                "-map",
                video_map,
                "-map",
                "1:a:0",
            ]
        )
        if video_filter is not None:
            command.extend(["-vf", video_filter])
        command.extend(
            [
                "-c:v",
                render_options.video_codec,
                "-preset",
                render_options.preset,
                "-crf",
                str(render_options.crf),
                "-threads",
                "4",
                "-c:a",
                render_options.audio_codec,
                "-b:a",
                render_options.audio_bitrate,
            ]
        )
        subtitle_included = True
    else:
        if render_options.blur_bilibili_watermark or logo_path is not None:
            graph = ""
            source_label = "0:v"
            if render_options.blur_bilibili_watermark:
                graph = build_bilibili_watermark_filter_graph(render_options, output_label="watermark_ready")
                source_label = "watermark_ready"
            if logo_path is not None:
                if graph:
                    graph += ";"
                graph += build_ben_review_logo_filter_graph(input_label=source_label, logo_input_label="2:v")
            else:
                graph = build_bilibili_watermark_filter_graph(render_options)
            command.extend(["-filter_complex", graph])
        command.extend(
            [
                "-map",
                "[vout]" if (render_options.blur_bilibili_watermark or logo_path is not None) else "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                render_options.video_codec if (render_options.blur_bilibili_watermark or logo_path is not None) else "copy",
                "-c:a",
                render_options.audio_codec,
                "-b:a",
                render_options.audio_bitrate,
            ]
        )
        if render_options.blur_bilibili_watermark or logo_path is not None:
            command.extend(["-preset", render_options.preset, "-crf", str(render_options.crf), "-threads", "4"])

    if render_options.subtitle_mode != "soft":
        command.append("-shortest")
    command.extend(
        [
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    try:
        ffmpeg_runner(command)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        if portrait_ass_path is not None:
            portrait_ass_path.unlink(missing_ok=True)

    if not output_path.is_file():
        raise FinalRenderError("FFmpeg did not create the final MP4 file")
    if output_path.stat().st_size <= 0:
        output_path.unlink(missing_ok=True)
        raise FinalRenderError("FFmpeg created an empty final MP4 file")

    return FinalRenderResult(
        video_file=str(output_path),
        subtitle_mode=render_options.subtitle_mode,
        audio_replaced=True,
        subtitle_included=subtitle_included,
    )
