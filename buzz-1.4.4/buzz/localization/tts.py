from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

from buzz.localization.translation import TranslatedLocalizationTranscript


@dataclass(frozen=True)
class TTSRequest:
    """Provider-neutral request for one translated localization segment."""

    text: str
    language: str
    output_file_stem: str
    voice: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSResult:
    """Audio asset and metadata returned by a TTS provider."""

    audio_file: str
    audio_duration: float
    provider: str
    voice: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class TTSProvider(Protocol):
    """Provider-neutral interface for synthesizing one speech asset."""

    def synthesize(self, request: TTSRequest) -> TTSResult:
        ...


@dataclass(frozen=True)
class SynthesizedLocalizationSegment:
    start: float
    end: float
    source_text: str
    translated_text: str
    audio_file: str
    audio_duration: float
    provider: str
    voice: str | None = None
    provider_metadata: Mapping[str, object] = field(default_factory=dict)

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
        }


@dataclass(frozen=True)
class SynthesizedLocalizationTranscript:
    source_language: str
    target_language: str
    source_file: str
    segments: tuple[SynthesizedLocalizationSegment, ...]

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def synthesize_for_localization(
    transcript: TranslatedLocalizationTranscript,
    provider: TTSProvider,
    output_directory: str | Path,
    voice: str | None = None,
    options: Mapping[str, object] | None = None,
) -> SynthesizedLocalizationTranscript:
    """Synthesize one Vietnamese audio asset for every translated segment."""
    if not isinstance(transcript, TranslatedLocalizationTranscript):
        raise TypeError("A Phase 2 TranslatedLocalizationTranscript is required")
    if not transcript.source_file:
        raise ValueError("Source file is required")
    if transcript.target_language != "vi":
        raise ValueError("Phase 3 target language must be 'vi'")
    if not transcript.segments:
        raise ValueError("Phase 3 cannot synthesize an empty transcript")
    if not isinstance(output_directory, (str, Path)) or not str(
        output_directory
    ).strip():
        raise ValueError("Output directory is required")

    for segment in transcript.segments:
        if (
            not isinstance(segment.translated_text, str)
            or not segment.translated_text.strip()
        ):
            raise ValueError("Phase 3 cannot synthesize an empty translated segment")

    asset_directory = Path(output_directory)
    asset_directory.mkdir(parents=True, exist_ok=True)
    provider_options = {} if options is None else dict(options)
    synthesized_segments = []

    for index, segment in enumerate(transcript.segments):
        result = provider.synthesize(
            TTSRequest(
                text=segment.translated_text,
                language=transcript.target_language,
                output_file_stem=str(asset_directory / f"segment-{index:06d}"),
                voice=voice,
                options=provider_options,
            )
        )
        if not isinstance(result, TTSResult):
            raise ValueError("TTS provider returned an invalid response")
        if not isinstance(result.audio_file, str) or not result.audio_file:
            raise ValueError("TTS provider returned an invalid audio file")
        if not Path(result.audio_file).is_file():
            raise ValueError("TTS provider audio file does not exist")
        if (
            not isinstance(result.audio_duration, (int, float))
            or isinstance(result.audio_duration, bool)
            or result.audio_duration <= 0
        ):
            raise ValueError("TTS provider returned an invalid audio duration")
        if not isinstance(result.provider, str) or not result.provider.strip():
            raise ValueError("TTS provider returned invalid provider metadata")

        synthesized_segments.append(
            SynthesizedLocalizationSegment(
                start=segment.start,
                end=segment.end,
                source_text=segment.source_text,
                translated_text=segment.translated_text,
                audio_file=result.audio_file,
                audio_duration=float(result.audio_duration),
                provider=result.provider,
                voice=result.voice,
                provider_metadata=dict(result.metadata),
            )
        )

    return SynthesizedLocalizationTranscript(
        source_file=transcript.source_file,
        source_language=transcript.source_language,
        target_language=transcript.target_language,
        segments=tuple(synthesized_segments),
    )
