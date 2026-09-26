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

    subtitle_included = False

    if render_options.subtitle_mode == "soft":
        assert subtitle_path is not None
        command.extend(["-i", str(subtitle_path)])
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "2:0",
                "-c:v",
                "copy",
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
        subtitle_included = True
    elif render_options.subtitle_mode == "burn":
        assert subtitle_path is not None
        escaped_subtitle = _escape_subtitle_filter_path(subtitle_path)
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                f"subtitles='{escaped_subtitle}'",
                "-c:v",
                render_options.video_codec,
                "-preset",
                render_options.preset,
                "-crf",
                str(render_options.crf),
                "-c:a",
                render_options.audio_codec,
                "-b:a",
                render_options.audio_bitrate,
            ]
        )
        subtitle_included = True
    else:
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                render_options.audio_codec,
                "-b:a",
                render_options.audio_bitrate,
            ]
        )

    command.extend(
        [
            "-shortest",
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
