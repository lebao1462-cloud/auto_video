from dataclasses import dataclass, field
from typing import Mapping

from buzz.localization.tts import (
    SynthesizedLocalizationSegment,
    SynthesizedLocalizationTranscript,
)


class TimingSynchronizationError(ValueError):
    """Raised when speech cannot fit the source timeline within the timing policy."""


@dataclass(frozen=True)
class TimingPolicy:
    """Limits for natural speech-rate adjustment during localization timing."""

    min_playback_rate: float = 0.90
    max_playback_rate: float = 1.25

    def __post_init__(self):
        if not 0 < self.min_playback_rate <= 1:
            raise ValueError("min_playback_rate must be > 0 and <= 1")
        if self.max_playback_rate < 1:
            raise ValueError("max_playback_rate must be >= 1")


@dataclass(frozen=True)
class TimedLocalizationSegment:
    start: float
    end: float
    source_text: str
    translated_text: str
    audio_file: str
    audio_duration: float
    provider: str
    voice: str | None
    provider_metadata: Mapping[str, object] = field(default_factory=dict)
    slot_duration: float = 0.0
    playback_rate: float = 1.0
    adjusted_audio_duration: float = 0.0
    leading_padding: float = 0.0
    trailing_padding: float = 0.0
    timing_action: str = "none"

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "audio_file": self.audio_file,
            "audio_duration": self.audio_duration,
            "provider": self.provider,
            "voice": self.voice,
            "provider_metadata": dict(self.provider_metadata),
            "slot_duration": self.slot_duration,
            "playback_rate": self.playback_rate,
            "adjusted_audio_duration": self.adjusted_audio_duration,
            "leading_padding": self.leading_padding,
            "trailing_padding": self.trailing_padding,
            "timing_action": self.timing_action,
        }


@dataclass(frozen=True)
class TimedLocalizationTranscript:
    source_file: str
    source_language: str
    target_language: str
    timing_policy: TimingPolicy
    segments: tuple[TimedLocalizationSegment, ...]

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "timing_policy": {
                "min_playback_rate": self.timing_policy.min_playback_rate,
                "max_playback_rate": self.timing_policy.max_playback_rate,
            },
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _validate_segment(
    segment: SynthesizedLocalizationSegment,
    previous_end: float | None,
) -> float:
    if not isinstance(segment.start, (int, float)) or isinstance(segment.start, bool):
        raise TimingSynchronizationError("Segment start must be numeric")
    if not isinstance(segment.end, (int, float)) or isinstance(segment.end, bool):
        raise TimingSynchronizationError("Segment end must be numeric")
    if segment.start < 0 or segment.end <= segment.start:
        raise TimingSynchronizationError("Segment timing must have end > start >= 0")
    if previous_end is not None and segment.start < previous_end:
        raise TimingSynchronizationError("Source timeline contains overlapping segments")
    if (
        not isinstance(segment.audio_duration, (int, float))
        or isinstance(segment.audio_duration, bool)
        or segment.audio_duration <= 0
    ):
        raise TimingSynchronizationError("Audio duration must be positive")
    return float(segment.end)


def synchronize_for_localization(
    transcript: SynthesizedLocalizationTranscript,
    policy: TimingPolicy | None = None,
) -> TimedLocalizationTranscript:
    """Create a deterministic timing plan without modifying audio waveforms."""
    if not isinstance(transcript, SynthesizedLocalizationTranscript):
        raise TypeError("A Phase 3 SynthesizedLocalizationTranscript is required")
    if not transcript.segments:
        raise TimingSynchronizationError("Phase 4 cannot synchronize an empty transcript")

    timing_policy = TimingPolicy() if policy is None else policy
    if not isinstance(timing_policy, TimingPolicy):
        raise TypeError("policy must be a TimingPolicy")

    previous_end = None
    timed_segments = []

    for index, segment in enumerate(transcript.segments):
        previous_end = _validate_segment(segment, previous_end)

        slot_duration = float(segment.end - segment.start)
        audio_duration = float(segment.audio_duration)
        required_rate = audio_duration / slot_duration

        if abs(required_rate - 1.0) < 1e-9:
            playback_rate = 1.0
            adjusted_duration = slot_duration
            trailing_padding = 0.0
            action = "none"
        elif required_rate < 1.0:
            if required_rate >= timing_policy.min_playback_rate:
                playback_rate = required_rate
                adjusted_duration = slot_duration
                trailing_padding = 0.0
                action = "slow_down"
            else:
                playback_rate = 1.0
                adjusted_duration = audio_duration
                trailing_padding = slot_duration - adjusted_duration
                action = "pad_silence"
        else:
            if required_rate > timing_policy.max_playback_rate:
                raise TimingSynchronizationError(
                    "Segment "
                    f"{index} requires playback rate {required_rate:.3f}, "
                    f"above maximum {timing_policy.max_playback_rate:.3f}"
                )
            playback_rate = required_rate
            adjusted_duration = slot_duration
            trailing_padding = 0.0
            action = "speed_up"

        timed_segments.append(
            TimedLocalizationSegment(
                start=float(segment.start),
                end=float(segment.end),
                source_text=segment.source_text,
                translated_text=segment.translated_text,
                audio_file=segment.audio_file,
                audio_duration=audio_duration,
                provider=segment.provider,
                voice=segment.voice,
                provider_metadata=dict(segment.provider_metadata),
                slot_duration=slot_duration,
                playback_rate=float(playback_rate),
                adjusted_audio_duration=float(adjusted_duration),
                leading_padding=0.0,
                trailing_padding=float(max(0.0, trailing_padding)),
                timing_action=action,
            )
        )

    return TimedLocalizationTranscript(
        source_file=transcript.source_file,
        source_language=transcript.source_language,
        target_language=transcript.target_language,
        timing_policy=timing_policy,
        segments=tuple(timed_segments),
    )
