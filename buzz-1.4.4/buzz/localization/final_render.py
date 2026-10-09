from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from buzz.localization.audio_mix import _run_ffmpeg


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
        if render_options.cover_original_subtitles:
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
