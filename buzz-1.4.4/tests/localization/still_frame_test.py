from pathlib import Path
import shutil

import pytest

from buzz.localization.audio_mix import _run_ffmpeg, probe_media_duration
from buzz.localization.still_frame import (
    StillFrameExtensionEvent, StillFrameExtensionOptions,
    StillFrameExtensionError, plan_still_frame_extensions, render_still_frame_video,
)
from buzz.localization.timing import TimingPolicy, synchronize_for_localization
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
