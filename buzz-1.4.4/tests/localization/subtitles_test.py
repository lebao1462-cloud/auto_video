import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from buzz.localization.subtitles import (
    SubtitleGenerationError,
    SubtitleOptions,
    SubtitleResult,
    render_srt,
    render_vtt,
    write_subtitles,
)
from buzz.localization.timing import (
    TimedLocalizationSegment,
    TimedLocalizationTranscript,
    TimingPolicy,
)


def make_segment(
    start=0.0,
    end=2.5,
    text="Xin chào Việt Nam",
):
    return TimedLocalizationSegment(
        start=start,
        end=end,
        source_text="Hello Vietnam",
        translated_text=text,
        audio_file="speech.wav",
        audio_duration=end - start,
        provider="fake",
        voice="vi-test",
        provider_metadata={},
        slot_duration=end - start,
        playback_rate=1.0,
        adjusted_audio_duration=end - start,
        leading_padding=0.0,
        trailing_padding=0.0,
        timing_action="none",
    )


def make_transcript(segments=None, target_language="vi"):
    return TimedLocalizationTranscript(
        source_file="input.mp4",
        source_language="en",
        target_language=target_language,
        timing_policy=TimingPolicy(),
        segments=tuple(
            segments if segments is not None else [make_segment()]
        ),
    )


def test_renders_srt_with_vietnamese_unicode():
    content = render_srt(make_transcript())

    assert content == (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "Xin chào Việt Nam\n"
    )


def test_renders_vtt_header_and_decimal_separator():
    content = render_vtt(make_transcript())

    assert content.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:02.500" in content
    assert "Xin chào Việt Nam" in content


def test_preserves_segment_order_and_count():
    transcript = make_transcript(
        [
            make_segment(0.0, 1.0, "Một"),
            make_segment(1.0, 2.25, "Hai"),
            make_segment(2.25, 4.0, "Ba"),
        ]
    )

    content = render_srt(transcript)

    assert content.index("Một") < content.index("Hai") < content.index("Ba")
    assert content.count(" --> ") == 3


def test_formats_hours_minutes_seconds_and_milliseconds():
    transcript = make_transcript(
        [make_segment(3661.234, 3662.009, "Kiểm tra")]
    )

    content = render_srt(transcript)

    assert "01:01:01,234 --> 01:01:02,009" in content


def test_rounds_fractional_seconds_to_nearest_millisecond():
    transcript = make_transcript(
        [make_segment(0.0006, 1.9996, "Làm tròn")]
    )

    content = render_srt(transcript)

    assert "00:00:00,001 --> 00:00:02,000" in content


def test_wraps_long_text_without_breaking_unicode_words():
    text = (
        "Đây là một câu tiếng Việt tương đối dài để kiểm tra "
        "việc tự động xuống dòng phụ đề"
    )
    content = render_srt(
        make_transcript([make_segment(text=text)]),
        SubtitleOptions(max_line_length=30, max_lines=2),
    )

    subtitle_text = content.splitlines()[2:]
    assert len(subtitle_text) == 2
    assert "Việt" in content
    assert all(len(line) > 0 for line in subtitle_text)


def test_can_disable_line_wrapping():
    text = "Đây là một câu dài nhưng không được tự động xuống dòng"
    content = render_srt(
        make_transcript([make_segment(text=text)]),
        SubtitleOptions(max_line_length=None),
    )

    assert text in content


def test_normalizes_existing_whitespace_and_newlines():
    transcript = make_transcript(
        [make_segment(text="Xin chào\n\n   mọi người")]
    )

    content = render_srt(transcript)

    assert "Xin chào mọi người" in content


@pytest.mark.parametrize("text", ["", "   ", None])
def test_rejects_empty_translated_text(text):
    with pytest.raises(SubtitleGenerationError, match="text cannot be empty"):
        render_srt(make_transcript([make_segment(text=text)]))


@pytest.mark.parametrize(
    "start,end",
    [
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_rejects_invalid_intervals(start, end):
    with pytest.raises(SubtitleGenerationError, match="end > start"):
        render_srt(make_transcript([make_segment(start, end)]))


def test_rejects_overlapping_segments():
    transcript = make_transcript(
        [
            make_segment(0.0, 2.0, "Một"),
            make_segment(1.9, 3.0, "Hai"),
        ]
    )

    with pytest.raises(SubtitleGenerationError, match="must not overlap"):
        render_srt(transcript)


def test_rejects_empty_transcript():
    with pytest.raises(SubtitleGenerationError, match="empty subtitle"):
        render_srt(make_transcript([]))


def test_rejects_non_vietnamese_target_language():
    with pytest.raises(SubtitleGenerationError, match="target language"):
        render_srt(make_transcript(target_language="en"))


@pytest.mark.parametrize(
    "options",
    [
        lambda: SubtitleOptions(max_line_length=0),
        lambda: SubtitleOptions(max_line_length=-1),
        lambda: SubtitleOptions(max_lines=0),
    ],
)
def test_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        options()


def test_writes_utf8_srt_file(tmp_path):
    path = tmp_path / "viet.srt"

    result = write_subtitles(make_transcript(), path)

    assert result == SubtitleResult(
        subtitle_file=str(path),
        format="srt",
        cue_count=1,
    )
    assert path.read_text(encoding="utf-8") == render_srt(make_transcript())


def test_writes_vtt_file(tmp_path):
    path = tmp_path / "viet.vtt"

    result = write_subtitles(make_transcript(), path)

    assert result.format == "vtt"
    assert result.cue_count == 1
    assert path.read_text(encoding="utf-8").startswith("WEBVTT\n\n")


def test_explicit_format_must_match_extension(tmp_path):
    with pytest.raises(SubtitleGenerationError, match="extension"):
        write_subtitles(
            make_transcript(),
            tmp_path / "viet.srt",
            format="vtt",
        )


def test_rejects_unsupported_format(tmp_path):
    with pytest.raises(SubtitleGenerationError, match="format"):
        write_subtitles(
            make_transcript(),
            tmp_path / "viet.ass",
        )


def test_result_is_json_compatible(tmp_path):
    result = write_subtitles(
        make_transcript(),
        tmp_path / "viet.srt",
    )

    encoded = json.dumps(result.to_dict(), ensure_ascii=False)
    assert json.loads(encoded) == result.to_dict()


def test_does_not_mutate_timed_transcript(tmp_path):
    transcript = make_transcript()
    original = transcript.to_dict()

    write_subtitles(transcript, tmp_path / "viet.srt")

    assert transcript.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        transcript.source_file = "other.mp4"
