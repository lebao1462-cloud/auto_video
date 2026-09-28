from dataclasses import dataclass
from typing import Protocol

from buzz.localization.transcript import (
    LocalizationTranscript,
    SUPPORTED_SOURCE_LANGUAGES,
)

SUPPORTED_TARGET_LANGUAGES = {"vi"}


class TranslationProvider(Protocol):
    """Provider-neutral interface for translating localization text."""

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        ...


@dataclass(frozen=True)
class TranslatedLocalizationSegment:
    start: float
    end: float
    source_text: str
    translated_text: str

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
        }


@dataclass(frozen=True)
class TranslatedLocalizationTranscript:
    source_language: str
    target_language: str
    source_file: str
    segments: tuple[TranslatedLocalizationSegment, ...]

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def translate_for_localization(
    transcript: LocalizationTranscript,
    provider: TranslationProvider,
    target_language: str = "vi",
) -> TranslatedLocalizationTranscript:
    """Translate a Phase 1 transcript while preserving its source data and timing."""
    if not isinstance(transcript, LocalizationTranscript):
        raise TypeError("A Phase 1 LocalizationTranscript is required")
    if not transcript.source_file:
        raise ValueError("Source file is required")
    if transcript.source_language not in SUPPORTED_SOURCE_LANGUAGES:
        raise ValueError("Phase 2 source language must be 'en' or 'zh'")
    if target_language not in SUPPORTED_TARGET_LANGUAGES:
        raise ValueError("Phase 2 target language must be 'vi'")
    if not transcript.segments:
        raise ValueError("Phase 2 cannot translate an empty transcript")

    for segment in transcript.segments:
        if not isinstance(segment.text, str) or not segment.text.strip():
            raise ValueError("Phase 2 cannot translate an empty source segment")

    translated_segments = []
    if callable(getattr(provider, "translate_segments", None)):
        batch_size = 30
        for start in range(0, len(transcript.segments), batch_size):
            batch = transcript.segments[start:start + batch_size]
            context = transcript.segments[max(0, start - 5):start]
            items = provider.translate_segments(
                batch, transcript.source_language, target_language, context=context
            )
            if len(items) != len(batch):
                raise ValueError("Translation provider returned a different segment count")
            for segment, item in zip(batch, items):
                corrected, translated = item
                if not isinstance(corrected, str) or not corrected.strip() or not isinstance(translated, str) or not translated.strip():
                    raise ValueError("Translation provider returned an invalid response")
                translated_segments.append(TranslatedLocalizationSegment(
                    start=segment.start, end=segment.end,
                    source_text=corrected.strip(), translated_text=translated.strip(),
                ))
        return TranslatedLocalizationTranscript(
            source_file=transcript.source_file, source_language=transcript.source_language,
            target_language=target_language, segments=tuple(translated_segments),
        )

    for segment in transcript.segments:
        translated_text = provider.translate(
            segment.text,
            source_language=transcript.source_language,
            target_language=target_language,
        )
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise ValueError("Translation provider returned an invalid response")
        translated_segments.append(
            TranslatedLocalizationSegment(
                start=segment.start,
                end=segment.end,
                source_text=segment.text,
                translated_text=translated_text,
            )
        )

    return TranslatedLocalizationTranscript(
        source_file=transcript.source_file,
        source_language=transcript.source_language,
        target_language=target_language,
        segments=tuple(translated_segments),
    )