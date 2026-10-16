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
    split_portrait_cue,
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


def test_rebalances_very_long_vietnamese_text_across_two_lines():
    text = (
        "\u0110\u00e2y l\u00e0 m\u1ed9t c\u00e2u ph\u1ee5 \u0111\u1ec1 ti\u1ebfng Vi\u1ec7t r\u1ea5t d\u00e0i c\u1ea7n \u0111\u01b0\u1ee3c c\u00e2n b\u1eb1ng \u0111\u1ec1u "
        "gi\u1eefa hai d\u00f2ng \u0111\u1ec3 kh\u00f4ng t\u1ea1o th\u00e0nh d\u00f2ng cu\u1ed1i qu\u00e1 d\u00e0i"
    )
    content = render_srt(
        make_transcript([make_segment(text=text)]),
        SubtitleOptions(max_line_length=30, max_lines=2),
    )

    subtitle_text = content.splitlines()[2:]
    assert 1 <= len(subtitle_text) <= 2
    assert " ".join(subtitle_text) == text
    assert max(map(len, subtitle_text)) - min(map(len, subtitle_text)) <= 8


def test_can_disable_line_wrapping():
    text = "Đây là một câu dài nhưng không được tự động xuống dòng"
    content = render_srt(
        make_transcript([make_segment(text=text)]),
        SubtitleOptions(max_line_length=None),
    )

    assert text in content


def test_portrait_two_second_cue_is_two_readable_sequential_lines():
    text = "\u0110\u00e2y l\u00e0 c\u00e2u \u0111\u1ea7u \u0111\u1ee7 d\u00e0i. \u0110\u00e2y l\u00e0 c\u00e2u sau \u0111\u1ee7 d\u00e0i."
    cues = split_portrait_cue(text, 0.0, 2.0)

    assert 1 <= len(cues) <= 2
    assert cues[0][0] == 0.0 and cues[-1][1] == 2.0
    assert " ".join(cue[2] for cue in cues) == text
    assert all("\n" not in cue[2] for cue in cues)


def test_portrait_default_width_uses_dialogue_mask_not_legacy_narrow_column():
    from PIL import ImageFont
    text = "Day la mot dong phu de tieng Viet du dai de can duoc hien thi ro rang tren man hinh doc."
    cues = split_portrait_cue(text, 0.0, 3.0)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 10)

    assert " ".join(cue[2] for cue in cues) == text
    assert all(font.getlength(cue[2]) <= 320 for cue in cues)
    assert len(cues) < len(split_portrait_cue(text, 0.0, 3.0, max_width=140))


def test_actual_part1_srt_uses_readable_320px_portrait_lines_when_available():
    from PIL import ImageFont
    source = Path("D:/AutoVideoOutput/new/1/1_localization_work/part_0001/source.vi.srt")
    if not source.is_file():
        pytest.skip("User part_0001 subtitle file is not mounted in this test environment")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 10)
    blocks = source.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip().split("\n\n")
    original_cues = 0
    refreshed_cues = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = [value.replace(",", ".") for value in lines[1].split(" --> ")]
        def seconds(value):
            h, m, s = value.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        text = " ".join(lines[2:])
        original_cues += 1
        refreshed_cues.extend(split_portrait_cue(text, seconds(start), seconds(end)))
    assert original_cues == 146
    assert all(font.getlength(text) <= 320 for _, _, text in refreshed_cues)
    assert len(refreshed_cues) < 450


def test_portrait_long_vietnamese_cue_becomes_multiple_ordered_cues():
    text = " ".join(["N\u1ed9i dung ti\u1ebfng Vi\u1ec7t r\u00f5 r\u00e0ng, d\u1ec5 \u0111\u1ecdc."] * 6)
    cues = split_portrait_cue(text, 3.0, 7.0)

    assert len(cues) >= 3
    assert " ".join(cue[2] for cue in cues) == text
    assert cues[0][0] == 3.0 and cues[-1][1] == 7.0
    assert all(end > start for start, end, _ in cues)
    assert all(cues[index][1] == cues[index + 1][0] for index in range(len(cues) - 1))


def test_portrait_short_and_subsecond_cues_are_safe():
    assert split_portrait_cue("Xin ch\u00e0o!", 0.0, 0.7) == [(0.0, 0.7, "Xin ch\u00e0o!")]
    # A long 0.7s cue still gets positive, ordered render-only windows.
    cues = split_portrait_cue(" ".join(["Vi\u1ec7t Nam r\u1ea5t \u0111\u1eb9p."] * 4), 0.0, 0.7)
    assert all(end > start for start, end, _ in cues)


def test_portrait_option_writes_split_srt_and_correct_cue_count(tmp_path):
    text = "\u0110\u00e2y l\u00e0 c\u00e2u \u0111\u1ea7u \u0111\u1ee7 d\u00e0i. \u0110\u00e2y l\u00e0 c\u00e2u sau \u0111\u1ee7 d\u00e0i."
    path = tmp_path / "portrait.srt"
    result = write_subtitles(
        make_transcript([make_segment(0.0, 2.0, text)]), path,
        options=SubtitleOptions(max_line_length=None, max_lines=1, portrait_sequential_cues=True),
    )
    content = path.read_text(encoding="utf-8")

    assert result.cue_count == 2
    assert "00:00:00,000 --> 00:00:01,000" in content
    assert "00:00:01,000 --> 00:00:02,000" in content
    assert "\n\n" in content and "\n" in content


def test_nonportrait_subtitle_options_keep_one_original_cue():
    text = "\u0110\u00e2y l\u00e0 c\u00e2u \u0111\u1ea7u \u0111\u1ee7 d\u00e0i. \u0110\u00e2y l\u00e0 c\u00e2u sau \u0111\u1ee7 d\u00e0i."
    content = render_srt(make_transcript([make_segment(0.0, 2.0, text)]), SubtitleOptions(max_line_length=None))
    assert content.count(" --> ") == 1


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



def test_portrait_long_two_second_text_balances_two_one_second_cues():
    text = "Đây là một câu tiếng Việt rất dài phải chia thành hai lượt rõ ràng."
    parts = split_portrait_cue(text, 8.0, 10.0)
    assert len(parts) == 2
    assert [(start,end) for start,end,_ in parts] == [(8.0,9.0),(9.0,10.0)]
    assert " ".join(content for _,_,content in parts) == text


def test_portrait_single_line_still_keeps_short_dialogue():
    assert split_portrait_cue("Đồng ý!", 4.0, 6.0) == [(4.0,6.0,"Đồng ý!")]
