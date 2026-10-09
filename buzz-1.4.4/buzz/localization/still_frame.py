"""Experimental, bounded still-frame extension for impossible speech timing.

This module deliberately operates after all text/TTS recovery has failed.  It
does not alter speech, cached TTS, or the source transcript; it inserts only
short frozen video intervals and shifts the affected presentation timeline.
"""
from dataclasses import dataclass, replace
import math
from pathlib import Path
import subprocess
from typing import Callable

from buzz.localization.audio_mix import _ffmpeg_env, _run_ffmpeg
from buzz.localization.timing import TimingPolicy
from buzz.localization.tts import SynthesizedLocalizationTranscript


class StillFrameExtensionError(ValueError):
    """A requested extension is unsafe or exceeds its explicit bound."""


@dataclass(frozen=True)
class StillFrameExtensionOptions:
    enabled: bool = True
    max_seconds_per_event: float = 0.5
    max_seconds_total: float = 18.0
    fps: float = 30.0

    def __post_init__(self):
        if self.max_seconds_per_event <= 0 or self.max_seconds_total <= 0:
            raise ValueError("still-frame extension limits must be positive")
        if self.fps <= 0 or not math.isfinite(self.fps):
            raise ValueError("still-frame extension fps must be positive")


@dataclass(frozen=True)
class StillFrameExtensionEvent:
    """An insertion in the retimed presentation timeline, in seconds."""
    at: float
    duration: float
    segment_index: int

    def to_dict(self) -> dict:
        return {"at": self.at, "duration": self.duration,
                "segment_index": self.segment_index}


@dataclass(frozen=True)
class StillFrameExtensionPlan:
    transcript: SynthesizedLocalizationTranscript
    events: tuple[StillFrameExtensionEvent, ...]

    @property
    def total_duration(self) -> float:
        return sum(event.duration for event in self.events)


def probe_video_fps(source: str | Path) -> float:
    """Read the actual stream frame rate used to round extension duration."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", str(source),
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", env=_ffmpeg_env())
    value = result.stdout.strip()
    try:
        numerator, denominator = value.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as exc:
        raise StillFrameExtensionError("FFprobe could not determine source video fps") from exc
    if result.returncode or not math.isfinite(fps) or fps <= 0:
        raise StillFrameExtensionError("FFprobe could not determine source video fps")
    return fps


def plan_still_frame_extensions(
    transcript: SynthesizedLocalizationTranscript, policy: TimingPolicy,
    *, media_end: float | None, options: StillFrameExtensionOptions,
) -> StillFrameExtensionPlan:
    """Return a shifted transcript that fits at ``policy.max_playback_rate``.

    Insertions occur at the original end of an overflowing utterance.  Later
    source events (and subtitle timestamps) move by the accumulated insertion.
    Durations are rounded *up* to video frames so no TTS tail is cut.
    """
    if not options.enabled:
        raise StillFrameExtensionError("Still-frame extension is disabled")
    accumulated = 0.0
    events: list[StillFrameExtensionEvent] = []
    shifted = []
    segments = transcript.segments
    for index, segment in enumerate(segments):
        next_start = (float(segments[index + 1].start) if index + 1 < len(segments)
                      else (media_end if media_end is not None else float(segment.end)))
        available = next_start - float(segment.start)
        needed = max(0.0, float(segment.audio_duration) / policy.max_playback_rate - available)
        extension = 0.0
        if needed > 1e-9:
            extension = math.ceil(needed * options.fps - 1e-9) / options.fps
            if extension > options.max_seconds_per_event + 1e-9:
                raise StillFrameExtensionError(
                    f"Segment {index} needs {extension:.3f}s still-frame extension; "
                    f"limit is {options.max_seconds_per_event:.3f}s per event"
                )
            if accumulated + extension > options.max_seconds_total + 1e-9:
                raise StillFrameExtensionError(
                    f"Still-frame extension would total {accumulated + extension:.3f}s; "
                    f"limit is {options.max_seconds_total:.3f}s"
                )
            events.append(StillFrameExtensionEvent(
                # Event coordinates are on the ORIGINAL source video.  The
                # cumulative offset belongs only to the shifted transcript.
                at=float(segment.end), duration=extension,
                segment_index=index,
            ))
        # The utterance's presentation end includes the freeze at its end.
        # Use microsecond precision consistently. Floating-point accumulation
        # can otherwise make two originally adjacent boundaries differ by 1e-13
        # and falsely trigger the strict transcript overlap validator.
        shifted.append(replace(
            segment,
            start=round(float(segment.start) + accumulated, 6),
            end=round(float(segment.end) + accumulated + extension, 6),
        ))
        accumulated += extension
    return StillFrameExtensionPlan(replace(transcript, segments=tuple(shifted)), tuple(events))


def render_still_frame_video(
    source: str | Path, destination: str | Path,
    events: tuple[StillFrameExtensionEvent, ...], *,
    ffmpeg_runner: Callable[[list[str]], None] = _run_ffmpeg,
) -> Path:
    """Render source video without audio, selectively freezing its last frame.

    A filter graph is passed as one argument (never through a shell), which is
    important for Windows paths and subtitle-like punctuation.
    """
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise StillFrameExtensionError(f"Source video does not exist: {source}")
    if not events:
        return source
    previous = 0.0
    filters, labels = [], []
    for index, event in enumerate(events):
        if event.at <= previous or event.duration <= 0:
            raise StillFrameExtensionError("Still-frame events must be ordered and non-overlapping")
        label = f"v{index}"
        filters.append(
            f"[0:v]trim=start={previous:.6f}:end={event.at:.6f},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={event.duration:.6f}[{label}]"
        )
        labels.append(f"[{label}]")
        previous = event.at
    label = f"v{len(events)}"
    filters.append(f"[0:v]trim=start={previous:.6f},setpts=PTS-STARTPTS[{label}]")
    labels.append(f"[{label}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[vout]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_runner([
        "ffmpeg", "-y", "-nostdin", "-i", str(source), "-filter_complex",
        ";".join(filters), "-map", "[vout]", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-threads", "4",
        "-movflags", "+faststart", str(destination),
    ])
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise StillFrameExtensionError("FFmpeg did not create the retimed still-frame video")
    return destination
