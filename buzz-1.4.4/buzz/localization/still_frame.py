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

from buzz.localization.audio_mix import _ffmpeg_env, _run_ffmpeg, probe_media_duration
from buzz.localization.timing import TimingPolicy
from buzz.localization.tts import SynthesizedLocalizationTranscript


class StillFrameExtensionError(ValueError):
    """A requested extension is unsafe or exceeds its explicit bound."""


@dataclass(frozen=True)
class StillFrameExtensionOptions:
    enabled: bool = True
    # Natural TTS occasionally needs a longer internal pause than the original
    # experimental half-second cap.  Keep this bounded: timing must still fit
    # at the hard playback ceiling and all freezes share the total cap below.
    max_seconds_per_event: float = 2.0
    # Fifteen-minute chunks can contain hundreds of short TTS utterances.  A
    # 45 seconds (5% of a standard 15-minute part) accommodates bounded
    # recovery distributed across such a chunk (including a terminal
    # correction) while preserving
    # explicit caller overrides and the per-event limits below.
    max_seconds_total: float = 45.0
    # A final ASR timestamp can land just past the container end.  This is a
    # distinct case from an internal pause: it freezes only the source's last
    # frame and is deliberately bounded independently of the per-event cap.
    max_terminal_seconds: float = 5.0
    fps: float = 30.0

    def __post_init__(self):
        if (self.max_seconds_per_event <= 0 or self.max_seconds_total <= 0
                or self.max_terminal_seconds <= 0):
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


_ADAPTIVE_MIN_MEDIA_SECONDS = 300.0
_ADAPTIVE_MAX_MEDIA_FRACTION = 0.15
# A natural utterance can occasionally need a multi-second internal hold.  On
# standard 900-second parts, allow that bounded recovery without changing the
# persisted 2-second default or any explicit operator override.
_ADAPTIVE_MAX_SECONDS_PER_EVENT = 4.0


def effective_still_frame_extension_options(
    options: StillFrameExtensionOptions, *, media_duration: float | None,
) -> StillFrameExtensionOptions:
    """Return a runtime-only bound for dense, long default-configured parts.

    The dataclass defaults deliberately remain the persisted/user-visible
    settings.  Only the exact default total and event limits opt into this
    fallback, so an operator's smaller bound is never silently widened.
    """
    if (media_duration is None or not isinstance(media_duration, (int, float))
            or isinstance(media_duration, bool) or not math.isfinite(media_duration)
            or media_duration < _ADAPTIVE_MIN_MEDIA_SECONDS
            or options.max_seconds_total != 45.0
            or options.max_seconds_per_event != 2.0):
        return options
    return replace(
        options,
        max_seconds_total=max(options.max_seconds_total,
                              float(media_duration) * _ADAPTIVE_MAX_MEDIA_FRACTION),
        max_seconds_per_event=_ADAPTIVE_MAX_SECONDS_PER_EVENT,
    )


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
    options = effective_still_frame_extension_options(
        options, media_duration=media_end,
    )
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


def plan_terminal_still_frame_extension(
    transcript: SynthesizedLocalizationTranscript, policy: TimingPolicy,
    *, media_end: float, options: StillFrameExtensionOptions,
) -> StillFrameExtensionPlan:
    """Extend a clip's last frame for a terminal ASR/container disagreement.

    The final subtitle and speech are retained.  Unlike ordinary insertions,
    the freeze is anchored at the actual source endpoint and its bound covers
    both the ASR overrun and the audio needed at the hard rate ceiling.
    """
    options = effective_still_frame_extension_options(
        options, media_duration=media_end,
    )
    if not options.enabled:
        raise StillFrameExtensionError("Still-frame extension is disabled")
    if not transcript.segments:
        raise StillFrameExtensionError("Cannot extend an empty transcript")
    final_index = len(transcript.segments) - 1
    final = transcript.segments[final_index]
    required_end = max(
        float(final.end),
        float(final.start) + float(final.audio_duration) / policy.max_playback_rate,
    )
    needed = required_end - float(media_end)
    if needed <= 1e-9:
        raise StillFrameExtensionError("Terminal still-frame extension is not needed")
    extension = math.ceil(needed * options.fps - 1e-9) / options.fps
    if extension > options.max_terminal_seconds + 1e-9:
        raise StillFrameExtensionError(
            f"Terminal correction needs {extension:.3f}s still-frame extension; "
            f"limit is {options.max_terminal_seconds:.3f}s"
        )
    if extension > options.max_seconds_total + 1e-9:
        raise StillFrameExtensionError(
            f"Terminal correction needs {extension:.3f}s still-frame extension; "
            f"total limit is {options.max_seconds_total:.3f}s"
        )
    extended_end = round(float(media_end) + extension, 6)
    shifted = list(transcript.segments)
    shifted[final_index] = replace(final, end=extended_end)
    return StillFrameExtensionPlan(
        replace(transcript, segments=tuple(shifted)),
        (StillFrameExtensionEvent(float(media_end), extension, final_index),),
    )


def reconcile_still_frame_extensions(
    transcript, *, media_end: float, options: StillFrameExtensionOptions,
):
    """Ensure a saved timing plan has enough video for its final speech slot.

    Timing recovery may resume from a checkpoint after an earlier still-frame
    plan was made.  At the render boundary, the timed slots are authoritative:
    reconcile them with the original source duration plus every planned
    insertion.  Any correction is a single, frame-rounded terminal freeze at
    the original source endpoint, never a second event at that coordinate.
    """
    if not transcript.segments:
        return transcript
    if not isinstance(media_end, (int, float)) or isinstance(media_end, bool):
        raise StillFrameExtensionError("Source media end must be numeric")
    media_end = float(media_end)
    if not math.isfinite(media_end) or media_end < 0:
        raise StillFrameExtensionError("Source media end must be non-negative and finite")

    options = effective_still_frame_extension_options(
        options, media_duration=media_end,
    )
    events = tuple(transcript.video_extensions)
    speech_end = max(
        max(float(segment.end), float(segment.start) + float(segment.slot_duration))
        for segment in transcript.segments
    )
    rendered_end = media_end + sum(float(event.duration) for event in events)
    missing = speech_end - rendered_end
    if missing <= 1e-6:
        return transcript
    if not options.enabled:
        raise StillFrameExtensionError(
            "Still-frame extension is disabled but the rendered video would cut speech"
        )

    extension = math.ceil(missing * options.fps - 1e-9) / options.fps
    terminal_indices = [
        index for index, event in enumerate(events)
        if abs(float(event.at) - media_end) <= 1e-6
    ]
    if len(terminal_indices) > 1:
        raise StillFrameExtensionError(
            "Still-frame plan has duplicate terminal events at the source endpoint"
        )
    terminal_duration = (
        float(events[terminal_indices[0]].duration) if terminal_indices else 0.0
    )
    if terminal_duration + extension > options.max_terminal_seconds + 1e-9:
        raise StillFrameExtensionError(
            "Terminal reconciliation needs "
            f"{terminal_duration + extension:.3f}s still-frame extension; "
            f"limit is {options.max_terminal_seconds:.3f}s"
        )
    if sum(float(event.duration) for event in events) + extension > options.max_seconds_total + 1e-9:
        raise StillFrameExtensionError(
            "Still-frame reconciliation would total "
            f"{sum(float(event.duration) for event in events) + extension:.3f}s; "
            f"limit is {options.max_seconds_total:.3f}s"
        )

    if terminal_indices:
        index = terminal_indices[0]
        revised = list(events)
        revised[index] = replace(revised[index], duration=terminal_duration + extension)
    else:
        revised = [*events, StillFrameExtensionEvent(
            media_end, extension, len(transcript.segments) - 1,
        )]
    revised.sort(key=lambda event: event.at)
    return replace(transcript, video_extensions=tuple(revised))


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
    # Do not feed concat an empty trailing trim when a terminal correction is
    # anchored at the exact source endpoint.  Some FFmpeg builds reject that
    # otherwise-valid graph instead of treating the final input as zero frames.
    source_duration = probe_media_duration(source)
    if previous < source_duration - 1e-6:
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
