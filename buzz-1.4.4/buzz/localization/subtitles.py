from dataclasses import dataclass
from pathlib import Path
import textwrap

from buzz.localization.timing import TimedLocalizationTranscript
from buzz.transcriber.file_transcriber import to_timestamp


class SubtitleGenerationError(ValueError):
    """Raised when localization subtitle data is invalid."""


@dataclass(frozen=True)
class SubtitleOptions:
    max_line_length: int | None = 42
    max_lines: int = 2

    def __post_init__(self):
        if self.max_line_length is not None and self.max_line_length <= 0:
            raise ValueError("max_line_length must be positive or None")
        if self.max_lines <= 0:
            raise ValueError("max_lines must be positive")


@dataclass(frozen=True)
class SubtitleResult:
    subtitle_file: str
    format: str
    cue_count: int

    def to_dict(self) -> dict:
        return {
            "subtitle_file": self.subtitle_file,
            "format": self.format,
            "cue_count": self.cue_count,
        }


def _format_seconds(seconds: float, separator: str) -> str:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        raise SubtitleGenerationError("Subtitle timestamp must be numeric")
    if seconds < 0:
        raise SubtitleGenerationError("Subtitle timestamp must be non-negative")
    return to_timestamp(round(float(seconds) * 1000), ms_separator=separator)


def _wrap_text(text: str, options: SubtitleOptions) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        raise SubtitleGenerationError("Subtitle text cannot be empty")

    if options.max_line_length is None:
        return normalized

    lines = textwrap.wrap(
        normalized,
        width=options.max_line_length,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) <= options.max_lines:
        return "\n".join(lines)

    head = lines[: options.max_lines - 1]
    tail = " ".join(lines[options.max_lines - 1 :])
    return "\n".join([*head, tail])


def _validate_transcript(transcript: TimedLocalizationTranscript) -> None:
    if not isinstance(transcript, TimedLocalizationTranscript):
        raise TypeError("A Phase 4 TimedLocalizationTranscript is required")
    if transcript.target_language != "vi":
        raise SubtitleGenerationError("Phase 6 target language must be 'vi'")
    if not transcript.segments:
        raise SubtitleGenerationError("Phase 6 cannot generate an empty subtitle file")

    previous_end = None
    for segment in transcript.segments:
        if segment.end <= segment.start:
            raise SubtitleGenerationError("Subtitle interval must have end > start")
        if previous_end is not None and segment.start < previous_end:
            raise SubtitleGenerationError("Subtitle intervals must not overlap")
        if not isinstance(segment.translated_text, str) or not segment.translated_text.strip():
            raise SubtitleGenerationError("Subtitle text cannot be empty")
        previous_end = segment.end


def render_srt(
    transcript: TimedLocalizationTranscript,
    options: SubtitleOptions | None = None,
) -> str:
    _validate_transcript(transcript)
    subtitle_options = SubtitleOptions() if options is None else options
    if not isinstance(subtitle_options, SubtitleOptions):
        raise TypeError("options must be SubtitleOptions")

    cues = []
    for index, segment in enumerate(transcript.segments, start=1):
        start = _format_seconds(segment.start, ",")
        end = _format_seconds(segment.end, ",")
        text = _wrap_text(segment.translated_text, subtitle_options)
        cues.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(cues) + "\n"


def render_vtt(
    transcript: TimedLocalizationTranscript,
    options: SubtitleOptions | None = None,
) -> str:
    _validate_transcript(transcript)
    subtitle_options = SubtitleOptions() if options is None else options
    if not isinstance(subtitle_options, SubtitleOptions):
        raise TypeError("options must be SubtitleOptions")

    cues = ["WEBVTT", ""]
    for segment in transcript.segments:
        start = _format_seconds(segment.start, ".")
        end = _format_seconds(segment.end, ".")
        text = _wrap_text(segment.translated_text, subtitle_options)
        cues.extend([f"{start} --> {end}", text, ""])
    return "\n".join(cues)


def write_subtitles(
    transcript: TimedLocalizationTranscript,
    output_file: str | Path,
    *,
    format: str | None = None,
    options: SubtitleOptions | None = None,
) -> SubtitleResult:
    _validate_transcript(transcript)

    output_path = Path(output_file)
    subtitle_format = (format or output_path.suffix.lstrip(".")).lower()
    if subtitle_format not in {"srt", "vtt"}:
        raise SubtitleGenerationError("Subtitle format must be 'srt' or 'vtt'")

    if output_path.suffix.lower() != f".{subtitle_format}":
        raise SubtitleGenerationError(
            f"Output file extension must be .{subtitle_format}"
        )

    content = (
        render_srt(transcript, options)
        if subtitle_format == "srt"
        else render_vtt(transcript, options)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8", newline="\n")

    return SubtitleResult(
        subtitle_file=str(output_path),
        format=subtitle_format,
        cue_count=len(transcript.segments),
    )
