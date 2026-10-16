from dataclasses import dataclass
from pathlib import Path
import textwrap
import unicodedata

from buzz.localization.timing import TimedLocalizationTranscript
from buzz.transcriber.file_transcriber import to_timestamp


class SubtitleGenerationError(ValueError):
    """Raised when localization subtitle data is invalid."""


@dataclass(frozen=True)
class SubtitleOptions:
    max_line_length: int | None = 42
    max_lines: int = 2
    # Portrait burn-in uses a narrow, single-line dialogue band.  Long cues
    # are shown as consecutive one-line cues instead of squeezing glyphs.
    portrait_sequential_cues: bool = False

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

    words = normalized.split()
    line_count = min(options.max_lines, len(words))
    word_lengths = [len(word) for word in words]
    prefix_lengths = [0]
    for length in word_lengths:
        prefix_lengths.append(prefix_lengths[-1] + length)

    def line_length(start: int, end: int) -> int:
        return prefix_lengths[end] - prefix_lengths[start] + (end - start - 1)

    # Partition all words across the permitted lines, minimizing the longest
    # line. This preserves every word without leaving an oversized final line.
    costs: list[list[int | None]] = [
        [None] * (len(words) + 1) for _ in range(line_count + 1)
    ]
    splits: list[list[int | None]] = [
        [None] * (len(words) + 1) for _ in range(line_count + 1)
    ]
    costs[0][0] = 0
    for current_line in range(1, line_count + 1):
        for end in range(current_line, len(words) + 1):
            for start in range(current_line - 1, end):
                previous_cost = costs[current_line - 1][start]
                if previous_cost is None:
                    continue
                cost = max(previous_cost, line_length(start, end))
                if costs[current_line][end] is None or cost < costs[current_line][end]:
                    costs[current_line][end] = cost
                    splits[current_line][end] = start

    balanced_lines = []
    end = len(words)
    for current_line in range(line_count, 0, -1):
        start = splits[current_line][end]
        assert start is not None
        balanced_lines.append(" ".join(words[start:end]))
        end = start
    return "\n".join(reversed(balanced_lines))


def _portrait_font():
    try:
        from PIL import ImageFont
    except ImportError as exc:  # pragma: no cover - reported by the renderer too
        raise SubtitleGenerationError("Portrait subtitles require Pillow") from exc
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if not font_path.is_file():
        raise SubtitleGenerationError("Portrait subtitle font is missing: Arial")
    return ImageFont.truetype(str(font_path), 32)


def split_portrait_cue(
    text: str, start: float, end: float, *, max_width: float = 680,
) -> list[tuple[float, float, str]]:
    """Return sequential, readable single-line portrait cues.

    Words are never discarded or duplicated.  A pathological unbreakable token
    is split only between Unicode grapheme-like clusters (base plus combining
    marks), which keeps Vietnamese diacritics attached to their base letter.
    """
    normalized = " ".join(text.split())
    if not normalized:
        raise SubtitleGenerationError("Subtitle text cannot be empty")
    if end <= start:
        raise SubtitleGenerationError("Subtitle interval must have end > start")
    font = _portrait_font()

    # (text, insert_space_before). Normal words preserve word boundaries;
    # fragments of an exceptional over-wide word do not gain a space.
    atoms: list[tuple[str, bool]] = []
    for word_index, word in enumerate(normalized.split()):
        prefix_space = word_index > 0
        if font.getlength(word) <= max_width:
            atoms.append((word, prefix_space))
            continue
        clusters: list[str] = []
        for character in word:
            if clusters and unicodedata.combining(character):
                clusters[-1] += character
            else:
                clusters.append(character)
        fragment = ""
        first_fragment = True
        for cluster in clusters:
            if fragment and font.getlength(fragment + cluster) > max_width:
                atoms.append((fragment, prefix_space if first_fragment else False))
                fragment = cluster
                first_fragment = False
            else:
                fragment += cluster
        if fragment:
            atoms.append((fragment, prefix_space if first_fragment else False))

    def joined(left: int, right: int) -> str:
        return "".join((" " if needs_space and index > left else "") + value
                       for index, (value, needs_space) in enumerate(atoms[left:right], left))

    # Greedily establishes the minimum count required by the real Arial metric.
    groups: list[tuple[int, int]] = []
    group_start = 0
    for index in range(1, len(atoms) + 1):
        if font.getlength(joined(group_start, index)) > max_width:
            groups.append((group_start, index - 1))
            group_start = index - 1
    groups.append((group_start, len(atoms)))
    count = len(groups)
    duration = end - start
    if count == 1:
        # A dense ~2s sentence can technically fit the whole portrait band
        # but is still hard to read as one moving line. Show two turns of 1s.
        if (1.7 <= duration <= 2.3 and len(atoms) >= 2
                and font.getlength(normalized) >= 500):
            midpoint = len(atoms) // 2
            groups = [(0, midpoint), (midpoint, len(atoms))]
            count = 2
        else:
            return [(start, end, normalized)]
    if count == 2:
        # Balance the real rendered glyph widths at word boundaries.
        possibilities = []
        for boundary in range(1, len(atoms)):
            left, right = joined(0, boundary), joined(boundary, len(atoms))
            left_width, right_width = font.getlength(left), font.getlength(right)
            if left_width <= max_width and right_width <= max_width:
                possibilities.append((abs(left_width-right_width),boundary))
        if possibilities:
            midpoint = min(possibilities)[1]
            groups = [(0, midpoint), (midpoint, len(atoms))]

    # ``groups`` already is the minimum number of real-font-width lines.  Do
    # not run a cubic repartitioning pass: real parts contain hundreds of
    # words, and the old 140px setting made this path unnecessarily slow.
    text_groups = [joined(left, right) for left, right in groups]
    visual_widths = [font.getlength(item) for item in text_groups]
    duration = end - start
    if count == 2 and (
        1.7 <= duration <= 2.3
        or max(visual_widths) / min(visual_widths) <= 1.6
    ):
        # Two seconds: first short line ~one second, then the second ~one second.
        shares = [0.5, 0.5]
    else:
        total = sum(visual_widths)
        shares = [width / total for width in visual_widths]
    minimum = min(0.7, (duration / count) * 0.95)
    available = duration - count * minimum
    durations = [minimum + available * share for share in shares]
    milliseconds = [max(1, round(value * 1000)) for value in durations]
    difference = round(duration * 1000) - sum(milliseconds)
    milliseconds[-1] += difference
    if milliseconds[-1] < 1:
        deficit = 1 - milliseconds[-1]
        milliseconds[-1] = 1
        for index in range(len(milliseconds) - 2, -1, -1):
            take = min(deficit, milliseconds[index] - 1)
            milliseconds[index] -= take
            deficit -= take
            if not deficit:
                break
    cursor = round(start * 1000)
    result = []
    for index, (cue_text, cue_duration) in enumerate(zip(text_groups, milliseconds)):
        next_cursor = round(end * 1000) if index == count - 1 else cursor + cue_duration
        result.append((cursor / 1000, next_cursor / 1000, cue_text))
        cursor = next_cursor
    return result


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
    for segment in transcript.segments:
        entries = (split_portrait_cue(segment.translated_text, segment.start, segment.end)
                   if subtitle_options.portrait_sequential_cues else
                   [(segment.start, segment.end, _wrap_text(segment.translated_text, subtitle_options))])
        for start_seconds, end_seconds, text in entries:
            cues.append(f"{len(cues) + 1}\n{_format_seconds(start_seconds, ',')} --> "
                        f"{_format_seconds(end_seconds, ',')}\n{text}")
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
        entries = (split_portrait_cue(segment.translated_text, segment.start, segment.end)
                   if subtitle_options.portrait_sequential_cues else
                   [(segment.start, segment.end, _wrap_text(segment.translated_text, subtitle_options))])
        for start_seconds, end_seconds, text in entries:
            cues.extend([f"{_format_seconds(start_seconds, '.')} --> "
                         f"{_format_seconds(end_seconds, '.')}", text, ""])
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

    cue_count = content.count(" --> ")
    return SubtitleResult(
        subtitle_file=str(output_path),
        format=subtitle_format,
        cue_count=cue_count,
    )
