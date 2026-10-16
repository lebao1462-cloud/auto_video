from pathlib import Path
import shutil

import pytest

from buzz.localization.audio_mix import _run_ffmpeg, probe_media_duration
from buzz.localization.still_frame import (
    StillFrameExtensionEvent, StillFrameExtensionOptions,
    StillFrameExtensionError, plan_still_frame_extensions,
    plan_terminal_still_frame_extension, reconcile_still_frame_extensions,
    render_still_frame_video,
)
from buzz.localization.timing import (
    TimedLocalizationSegment, TimedLocalizationTranscript, TimingPolicy,
    synchronize_for_localization,
)
from buzz.localization.tts import SynthesizedLocalizationSegment, SynthesizedLocalizationTranscript


def _transcript():
    return SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(0, 1, "one", "mot", "a.wav", 1.35, "test", None),
            SynthesizedLocalizationSegment(1, 2, "two", "hai", "b.wav", 1.0, "test", None),
        ),
    )


def test_plan_rounds_up_and_shifts_later_timestamps():
    plan = plan_still_frame_extensions(
        _transcript(), TimingPolicy(), media_end=2,
        options=StillFrameExtensionOptions(fps=30, max_seconds_per_event=.5),
    )
    assert len(plan.events) == 1
    assert plan.events[0].duration == pytest.approx(7 / 30)
    assert plan.transcript.segments[0].end == pytest.approx(1 + 7 / 30)
    assert plan.transcript.segments[1].start == pytest.approx(1 + 7 / 30)
    timed = synchronize_for_localization(
        plan.transcript, TimingPolicy(), media_end=2 + plan.total_duration,
    )
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_plan_refuses_unsafe_event():
    with pytest.raises(StillFrameExtensionError, match="per event"):
        plan_still_frame_extensions(
            _transcript(), TimingPolicy(), media_end=2,
            options=StillFrameExtensionOptions(max_seconds_per_event=.1),
        )


def test_default_plan_accepts_realistic_1100ms_and_1633ms_internal_freezes():
    policy = TimingPolicy()
    cases = ((2.31, 1.1), (2.8966666667, 49 / 30))
    for audio_duration, expected_extension in cases:
        transcript = SynthesizedLocalizationTranscript(
            "source.mp4", "en", "vi", (
                SynthesizedLocalizationSegment(
                    0, 1, "source", "translation", "segment.wav", audio_duration, "test", None,
                ),
            ),
        )
        plan = plan_still_frame_extensions(
            transcript, policy, media_end=1,
            options=StillFrameExtensionOptions(fps=30),
        )
        assert plan.events[0].duration == pytest.approx(expected_extension)
        timed = synchronize_for_localization(
            plan.transcript, policy, media_end=1 + plan.total_duration,
        )
        assert timed.segments[0].playback_rate <= 1.10


def test_default_plan_refuses_event_above_two_second_cap():
    transcript = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(0, 1, "source", "translation", "segment.wav", 3.31, "test", None),
        ),
    )
    with pytest.raises(StillFrameExtensionError, match="2.000s per event"):
        plan_still_frame_extensions(
            transcript, TimingPolicy(), media_end=1,
            options=StillFrameExtensionOptions(fps=30),
        )


def test_plan_refuses_total_cap_across_internal_events():
    transcript = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(0, 1, "a", "a", "a.wav", 2.42, "test", None),
            SynthesizedLocalizationSegment(1, 2, "b", "b", "b.wav", 2.42, "test", None),
            SynthesizedLocalizationSegment(2, 3, "c", "c", "c.wav", 0.1, "test", None),
        ),
    )
    with pytest.raises(StillFrameExtensionError, match="would total"):
        plan_still_frame_extensions(
            transcript, TimingPolicy(), media_end=4,
            options=StillFrameExtensionOptions(fps=30, max_seconds_total=2),
        )


def _one_second_segments(count: int) -> SynthesizedLocalizationTranscript:
    return SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", tuple(
            SynthesizedLocalizationSegment(
                index, index + 1, f"source {index}", f"translation {index}",
                f"segment-{index}.wav", 3.3, "test", None,
            )
            for index in range(count)
        ),
    )


def test_default_total_budget_accepts_26_seconds_across_bounded_events():
    policy = TimingPolicy()
    plan = plan_still_frame_extensions(
        _one_second_segments(13), policy, media_end=13,
        options=StillFrameExtensionOptions(fps=30),
    )
    assert plan.total_duration == pytest.approx(26.0)
    assert all(event.duration == pytest.approx(2.0) for event in plan.events)
    timed = synchronize_for_localization(
        plan.transcript, policy, media_end=13 + plan.total_duration,
    )
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_default_total_budget_refuses_more_than_45_seconds():
    with pytest.raises(StillFrameExtensionError, match="would total 46.000s; limit is 45.000s"):
        plan_still_frame_extensions(
            _one_second_segments(23), TimingPolicy(), media_end=23,
            options=StillFrameExtensionOptions(fps=30),
        )


def test_long_default_part_adapts_only_after_the_normal_45_second_limit():
    plan = plan_still_frame_extensions(
        _one_second_segments(58), TimingPolicy(), media_end=900,
        options=StillFrameExtensionOptions(fps=30),
    )
    assert plan.total_duration == pytest.approx(114.0)
    assert max(event.duration for event in plan.events) == pytest.approx(2.0)
    timed = synchronize_for_localization(
        plan.transcript, TimingPolicy(), media_end=900 + plan.total_duration,
    )
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_long_default_part_adapts_event_cap_but_explicit_bounds_remain_strict():
    transcript = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", tuple(
            SynthesizedLocalizationSegment(
                index, index + 1, "a", "a", f"{index}.wav",
                # Segment 275 mirrors Part 0006: at a 1.10 ceiling, this
                # needs precisely 3.301 seconds of internal frozen video.
                4.7311 if index == 275 else 1.0, "test", None,
            )
            for index in range(277)
        ),
    )
    plan = plan_still_frame_extensions(
        transcript, TimingPolicy(), media_end=900,
        options=StillFrameExtensionOptions(fps=1000),
    )
    assert plan.events[0].segment_index == 275
    assert plan.events[0].duration == pytest.approx(3.301)
    with pytest.raises(StillFrameExtensionError, match="2.000s per event"):
        plan_still_frame_extensions(
            transcript, TimingPolicy(), media_end=900,
            options=StillFrameExtensionOptions(fps=1000, max_seconds_total=18),
        )
    with pytest.raises(StillFrameExtensionError, match="0.500s per event"):
        plan_still_frame_extensions(
            transcript, TimingPolicy(), media_end=900,
            options=StillFrameExtensionOptions(fps=1000, max_seconds_per_event=.5),
        )
    too_long = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(0, 1, "a", "a", "a.wav", 5.61, "test", None),
            SynthesizedLocalizationSegment(1, 2, "b", "b", "b.wav", 1.0, "test", None),
        ),
    )
    with pytest.raises(StillFrameExtensionError, match="4.000s per event"):
        plan_still_frame_extensions(
            too_long, TimingPolicy(), media_end=900,
            options=StillFrameExtensionOptions(fps=1000),
        )


def test_custom_total_budget_remains_an_explicit_override():
    with pytest.raises(StillFrameExtensionError, match="would total 20.000s; limit is 18.000s"):
        plan_still_frame_extensions(
            _one_second_segments(13), TimingPolicy(), media_end=13,
            options=StillFrameExtensionOptions(fps=30, max_seconds_total=18),
        )


def test_default_total_budget_reserves_terminal_correction_before_internal_freezes():
    policy = TimingPolicy()
    # This mirrors the Part 0002 bounds: 102 internal freezes totalling
    # 31.033s plus a 2.633s terminal correction.  The terminal plan is made
    # first, as the workflow does, and its duration is removed from the
    # ordinary planner's shared 45-second budget.
    internal_segments = tuple(
        SynthesizedLocalizationSegment(
            index + 797, index + 798, f"source {index}", f"translation {index}",
            f"segment-{index}.wav",
            1.43 if index < 101 else 2.9736666667,
            "test", None,
        )
        for index in range(102)
    )
    terminal = SynthesizedLocalizationSegment(
        899.97, 902.64, "last source", "last translation", "last.wav",
        2.0, "test", None,
    )
    transcript = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (*internal_segments, terminal),
    )
    media_end = 900.021334
    options = StillFrameExtensionOptions(fps=30)
    terminal_plan = plan_terminal_still_frame_extension(
        transcript, policy, media_end=media_end, options=options,
    )
    assert terminal_plan.total_duration == pytest.approx(79 / 30)

    internal_plan = plan_still_frame_extensions(
        terminal_plan.transcript, policy,
        media_end=media_end + terminal_plan.total_duration,
        options=StillFrameExtensionOptions(
            fps=30,
            max_seconds_total=options.max_seconds_total - terminal_plan.total_duration,
        ),
    )
    assert len(internal_plan.events) == 102
    assert internal_plan.total_duration == pytest.approx(31.0333333333)
    assert terminal_plan.total_duration + internal_plan.total_duration == pytest.approx(33.6666666667)
    timed = synchronize_for_localization(
        internal_plan.transcript, policy,
        media_end=media_end + terminal_plan.total_duration + internal_plan.total_duration,
    )
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_terminal_plan_preserves_final_text_and_fits_real_portrait_tail():
    transcript = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(
                899.97, 902.20, "last source", "dÃ²ng cuá»‘i", "last.wav",
                4.56, "test", None,
            ),
        ),
    )
    plan = plan_terminal_still_frame_extension(
        transcript, TimingPolicy(), media_end=900.021334,
        options=StillFrameExtensionOptions(fps=30, max_terminal_seconds=5),
    )
    assert plan.events[0].at == pytest.approx(900.021334)
    assert plan.events[0].duration == pytest.approx(4.1, abs=1e-6)
    assert plan.transcript.segments[-1].translated_text == "dÃ²ng cuá»‘i"
    timed = synchronize_for_localization(
        plan.transcript, TimingPolicy(), media_end=900.021334 + plan.total_duration,
    )
    assert timed.segments[-1].end <= 900.021334 + plan.total_duration
    assert max(segment.playback_rate for segment in timed.segments) <= 1.10


def test_terminal_plan_refuses_oversized_or_disabled_correction():
    transcript = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(9.9, 12, "a", "b", "a.wav", 4.56, "test", None),
        ),
    )
    with pytest.raises(StillFrameExtensionError, match="Terminal correction.*limit"):
        plan_terminal_still_frame_extension(
            transcript, TimingPolicy(), media_end=10,
            options=StillFrameExtensionOptions(max_terminal_seconds=.5),
        )
    with pytest.raises(StillFrameExtensionError, match="disabled"):
        plan_terminal_still_frame_extension(
            transcript, TimingPolicy(), media_end=10,
            options=StillFrameExtensionOptions(enabled=False),
        )


def test_reconciliation_repairs_part_0004_under_budget_checkpoint_plan():
    """A resumed Part 0004 plan must retain its tail without new provider calls."""
    source_end = 900.021334
    timed = TimedLocalizationTranscript(
        "source.mp4", "en", "vi", TimingPolicy(), (
            TimedLocalizationSegment(
                start=900.94, end=903.455303, source_text="last source",
                translated_text="last translation", audio_file="last.wav",
                audio_duration=2.72, provider="test", voice="voice",
                slot_duration=2.72,
            ),
        ),
        video_extensions=(StillFrameExtensionEvent(800, 3.267272, 0),),
    )

    reconciled = reconcile_still_frame_extensions(
        timed, media_end=source_end,
        options=StillFrameExtensionOptions(fps=30),
    )

    assert len(reconciled.video_extensions) == 2
    assert reconciled.video_extensions[-1].at == pytest.approx(source_end)
    assert reconciled.video_extensions[-1].duration == pytest.approx(0.4)
    rendered_end = source_end + sum(event.duration for event in reconciled.video_extensions)
    speech_end = max(
        reconciled.segments[-1].end,
        reconciled.segments[-1].start + reconciled.segments[-1].slot_duration,
    )
    assert rendered_end >= speech_end


def test_reconciliation_expands_one_terminal_event_without_duplication():
    timed = TimedLocalizationTranscript(
        "source.mp4", "en", "vi", TimingPolicy(), (
            TimedLocalizationSegment(
                10, 12, "source", "translation", "last.wav", 2.72, "test", "voice",
                slot_duration=2.72,
            ),
        ),
        video_extensions=(StillFrameExtensionEvent(10, 1.0, 0),),
    )
    reconciled = reconcile_still_frame_extensions(
        timed, media_end=10, options=StillFrameExtensionOptions(fps=30),
    )
    assert len(reconciled.video_extensions) == 1
    assert reconciled.video_extensions[0].duration == pytest.approx(2.7333333333)


def test_long_default_reconciliation_uses_the_same_15_percent_total_bound():
    timed = TimedLocalizationTranscript(
        "source.mp4", "en", "vi", TimingPolicy(), (
            TimedLocalizationSegment(
                1008, 1010.3, "source", "translation", "last.wav", 2.3,
                "test", "voice", slot_duration=2.3,
            ),
        ),
        video_extensions=(StillFrameExtensionEvent(800, 110.0, 0),),
    )
    reconciled = reconcile_still_frame_extensions(
        timed, media_end=900, options=StillFrameExtensionOptions(fps=30),
    )
    assert sum(event.duration for event in reconciled.video_extensions) == pytest.approx(110.3)
    assert reconciled.video_extensions[-1].at == 900


def test_reconciliation_preserves_standard_no_overrun_plan():
    timed = TimedLocalizationTranscript(
        "source.mp4", "en", "vi", TimingPolicy(), (
            TimedLocalizationSegment(
                0, 1, "source", "translation", "last.wav", 1, "test", "voice",
                slot_duration=1,
            ),
        ),
    )
    assert reconcile_still_frame_extensions(
        timed, media_end=1, options=StillFrameExtensionOptions(fps=30),
    ) is timed


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_real_ffmpeg_selective_freezes_extend_duration(tmp_path: Path):
    source, output = tmp_path / "source.mp4", tmp_path / "retimed.mp4"
    _run_ffmpeg([
        "ffmpeg", "-y", "-nostdin", "-f", "lavfi", "-i",
        "testsrc=size=64x64:rate=30", "-t", "2", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(source),
    ])
    render_still_frame_video(source, output, (
        StillFrameExtensionEvent(0.7, 0.2, 0),
        StillFrameExtensionEvent(1.5, 0.3, 1),
    ))
    assert probe_media_duration(output) == pytest.approx(2.5, abs=.12)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_real_ffmpeg_terminal_freeze_has_no_empty_trailing_concat_input(tmp_path: Path):
    source, output = tmp_path / "source.mp4", tmp_path / "retimed.mp4"
    _run_ffmpeg([
        "ffmpeg", "-y", "-nostdin", "-f", "lavfi", "-i",
        "testsrc=size=64x64:rate=30", "-t", "2", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(source),
    ])
    source_duration = probe_media_duration(source)
    render_still_frame_video(source, output, (
        StillFrameExtensionEvent(source_duration, 0.4, 0),
    ))
    assert probe_media_duration(output) == pytest.approx(source_duration + .4, abs=.12)

def test_two_extensions_use_original_video_coordinates_and_preserve_speech_order():
    source = SynthesizedLocalizationTranscript(
        "source.mp4", "en", "vi", (
            SynthesizedLocalizationSegment(0, 1, "a", "mot", "a.wav", 1.35, "test", None),
            SynthesizedLocalizationSegment(1, 2, "b", "hai", "b.wav", 1.35, "test", None),
            SynthesizedLocalizationSegment(2, 3, "c", "ba", "c.wav", 0.8, "test", None),
        ),
    )
    policy = TimingPolicy()
    plan = plan_still_frame_extensions(
        source, policy, media_end=3,
        options=StillFrameExtensionOptions(fps=30, max_seconds_per_event=.5),
    )
    assert [event.at for event in plan.events] == pytest.approx([1, 2])
    assert [event.duration for event in plan.events] == pytest.approx([7 / 30, 7 / 30])
    assert plan.transcript.segments[2].start == pytest.approx(2 + 14 / 30)
    assert plan.transcript.segments[0].end == plan.transcript.segments[1].start
    assert plan.transcript.segments[1].end == plan.transcript.segments[2].start
    timed = synchronize_for_localization(
        plan.transcript, policy, media_end=3 + plan.total_duration,
    )
    assert all(segment.playback_rate <= policy.max_playback_rate for segment in timed.segments)
