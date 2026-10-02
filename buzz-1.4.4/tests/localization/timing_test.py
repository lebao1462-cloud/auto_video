import json
from dataclasses import FrozenInstanceError

import pytest

from buzz.localization.timing import (
    TimedLocalizationSegment,
    TimingPolicy,
    TimingSynchronizationError,
    synchronize_for_localization,
)
from buzz.localization.tts import (
    SynthesizedLocalizationSegment,
    SynthesizedLocalizationTranscript,
)


def make_segment(
    start=0.0,
    end=2.0,
    audio_duration=2.0,
    source_text="Hello",
    translated_text="Xin chào",
):
    return SynthesizedLocalizationSegment(
        start=start,
        end=end,
        source_text=source_text,
        translated_text=translated_text,
        audio_file="segment.wav",
        audio_duration=audio_duration,
        provider="fake",
        voice="vi-test",
        provider_metadata={"format": "wav"},
    )


def make_transcript(segments=None):
    return SynthesizedLocalizationTranscript(
        source_file="input.mp4",
        source_language="en",
        target_language="vi",
        segments=tuple(segments if segments is not None else [make_segment()]),
    )


def test_exact_duration_requires_no_adjustment():
    result = synchronize_for_localization(make_transcript())

    segment = result.segments[0]
    assert segment.timing_action == "none"
    assert segment.playback_rate == pytest.approx(1.0)
    assert segment.adjusted_audio_duration == pytest.approx(2.0)
    assert segment.trailing_padding == pytest.approx(0.0)


def test_slightly_short_audio_is_slowed_within_policy():
    result = synchronize_for_localization(
        make_transcript([make_segment(end=2.0, audio_duration=1.9)])
    )

    segment = result.segments[0]
    assert segment.timing_action == "slow_down"
    assert segment.playback_rate == pytest.approx(0.95)
    assert segment.adjusted_audio_duration == pytest.approx(2.0)
    assert segment.trailing_padding == pytest.approx(0.0)


def test_much_shorter_audio_uses_silence_padding():
    result = synchronize_for_localization(
        make_transcript([make_segment(end=3.0, audio_duration=1.0)])
    )

    segment = result.segments[0]
    assert segment.timing_action == "pad_silence"
    assert segment.playback_rate == pytest.approx(1.0)
    assert segment.adjusted_audio_duration == pytest.approx(1.0)
    assert segment.trailing_padding == pytest.approx(2.0)
    assert segment.leading_padding == pytest.approx(0.0)


def test_slightly_long_audio_is_sped_up_within_policy():
    result = synchronize_for_localization(
        make_transcript([make_segment(end=2.0, audio_duration=2.4)])
    )

    segment = result.segments[0]
    assert segment.timing_action == "speed_up"
    assert segment.playback_rate == pytest.approx(1.2)
    assert segment.adjusted_audio_duration == pytest.approx(2.0)
    assert segment.trailing_padding == pytest.approx(0.0)


def test_much_longer_audio_is_rejected_instead_of_overcompressed():
    with pytest.raises(TimingSynchronizationError, match="above maximum"):
        synchronize_for_localization(
            make_transcript([make_segment(end=2.0, audio_duration=3.2)])
        )


def test_tiny_rate_overrun_is_clamped_to_configured_maximum():
    result = synchronize_for_localization(
        make_transcript([make_segment(end=1.0, audio_duration=2.006)]),
        TimingPolicy(max_playback_rate=2.0),
    )

    segment = result.segments[0]
    assert segment.timing_action == "speed_up"
    assert segment.playback_rate == pytest.approx(2.0)
    assert segment.adjusted_audio_duration == pytest.approx(1.0)


def test_clearly_excessive_rate_is_rejected_with_tolerance_enabled():
    with pytest.raises(TimingSynchronizationError, match="2.100.*above maximum 2.000"):
        synchronize_for_localization(
            make_transcript([make_segment(end=1.0, audio_duration=2.1)]),
            TimingPolicy(max_playback_rate=2.0),
        )


def test_custom_policy_controls_allowed_speed_change():
    transcript = make_transcript([make_segment(end=2.0, audio_duration=2.6)])

    result = synchronize_for_localization(
        transcript,
        TimingPolicy(max_playback_rate=1.35),
    )

    assert result.segments[0].playback_rate == pytest.approx(1.3)
    assert result.segments[0].timing_action == "speed_up"


def test_adjacent_segments_do_not_overlap():
    transcript = make_transcript(
        [
            make_segment(start=0.0, end=2.0, audio_duration=1.0),
            make_segment(start=2.0, end=4.0, audio_duration=2.2),
        ]
    )

    result = synchronize_for_localization(transcript)

    assert result.segments[0].end == result.segments[1].start
    assert result.segments[0].trailing_padding == pytest.approx(1.0)
    assert result.segments[1].playback_rate == pytest.approx(1.1)


def test_rejects_overlapping_source_segments():
    transcript = make_transcript(
        [
            make_segment(start=0.0, end=2.0),
            make_segment(start=1.9, end=3.0),
        ]
    )

    with pytest.raises(TimingSynchronizationError, match="overlapping"):
        synchronize_for_localization(transcript)


@pytest.mark.parametrize(
    "start,end",
    [
        (-0.1, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_rejects_invalid_segment_timing(start, end):
    with pytest.raises(TimingSynchronizationError, match="end > start"):
        synchronize_for_localization(
            make_transcript([make_segment(start=start, end=end)])
        )


@pytest.mark.parametrize("duration", [0, -1, None, "1.0", True])
def test_rejects_invalid_audio_duration(duration):
    with pytest.raises(TimingSynchronizationError, match="Audio duration"):
        synchronize_for_localization(
            make_transcript([make_segment(audio_duration=duration)])
        )


def test_rejects_empty_transcript():
    with pytest.raises(TimingSynchronizationError, match="empty transcript"):
        synchronize_for_localization(make_transcript([]))


@pytest.mark.parametrize(
    "policy",
    [
        lambda: TimingPolicy(min_playback_rate=0),
        lambda: TimingPolicy(min_playback_rate=1.1),
        lambda: TimingPolicy(max_playback_rate=0.9),
    ],
)
def test_rejects_invalid_policy(policy):
    with pytest.raises(ValueError):
        policy()


def test_output_is_json_compatible():
    result = synchronize_for_localization(
        make_transcript([make_segment(audio_duration=1.0)])
    )

    encoded = json.dumps(result.to_dict(), ensure_ascii=False)

    assert json.loads(encoded) == result.to_dict()
    assert result.to_dict()["segments"][0]["timing_action"] == "pad_silence"


def test_preserves_phase3_metadata():
    source = make_segment(
        start=1.0,
        end=3.0,
        audio_duration=2.2,
        source_text="Good morning",
        translated_text="Chào buổi sáng",
    )

    result = synchronize_for_localization(make_transcript([source]))
    timed = result.segments[0]

    assert timed.source_text == source.source_text
    assert timed.translated_text == source.translated_text
    assert timed.audio_file == source.audio_file
    assert timed.provider == source.provider
    assert timed.voice == source.voice
    assert timed.provider_metadata == source.provider_metadata
    assert timed.start == source.start
    assert timed.end == source.end


def test_does_not_mutate_phase3_input():
    transcript = make_transcript([make_segment(audio_duration=1.0)])
    original = transcript.to_dict()

    synchronize_for_localization(transcript)

    assert transcript.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        transcript.source_file = "other.mp4"


def test_returns_expected_timed_segment_type():
    result = synchronize_for_localization(make_transcript())

    assert isinstance(result.segments[0], TimedLocalizationSegment)
