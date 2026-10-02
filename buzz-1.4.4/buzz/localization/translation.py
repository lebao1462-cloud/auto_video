from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
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


class ConciseTranslationProvider(Protocol):
    """Optional capability for making one translated dubbing line fit its slot."""

    def shorten_translation(
        self, *, source_text: str, translated_text: str,
        source_language: str, target_language: str, required_rate: float,
        max_playback_rate: float,
    ) -> str:
        ...


class BatchConciseTranslationProvider(Protocol):
    """Optional capability for shortening several dubbing lines in one request."""

    def shorten_translations(
        self, *, segments, source_language: str, target_language: str,
    ) -> list[str]:
        """Return non-empty shortened text in the same order as ``segments``."""
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


@dataclass(frozen=True)
class _IdentifiedSegment:
    """A source segment with an ID that survives retries and batch splits."""

    start: float
    end: float
    text: str
    segment_id: str


def _segment_id(index, segment) -> str:
    value = json.dumps(
        [index, segment.start, segment.end, segment.text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"segment-{index:06d}-{hashlib.sha256(value).hexdigest()[:12]}"


def _provider_identity(provider) -> dict:
    return {
        "provider": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "model": getattr(provider, "model", None),
    }


def _checkpoint_fingerprint(transcript, provider, target_language) -> str:
    payload = {
        "source": transcript.to_dict(),
        "target_language": target_language,
        **_provider_identity(provider),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_checkpoint(path: Path, fingerprint: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return {}
        items = payload.get("segments")
        return items if isinstance(items, dict) else {}
    except (OSError, TypeError, json.JSONDecodeError):
        return {}


def _write_checkpoint(path: Path, fingerprint: str, items: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(
                {"fingerprint": fingerprint, "segments": items}, handle,
                ensure_ascii=False, sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def translate_for_localization(
    transcript: LocalizationTranscript,
    provider: TranslationProvider,
    target_language: str = "vi",
    checkpoint_path: str | Path | None = None,
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
        identified = tuple(
            _IdentifiedSegment(
                segment.start, segment.end, segment.text, _segment_id(index, segment)
            )
            for index, segment in enumerate(transcript.segments)
        )
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        fingerprint = _checkpoint_fingerprint(transcript, provider, target_language)
        completed = _read_checkpoint(checkpoint, fingerprint) if checkpoint else {}

        def translate_batch(batch, context):
            pending = [item for item in batch if item.segment_id not in completed]
            if not pending:
                return
            last_error = None
            for _ in range(2):
                try:
                    items = provider.translate_segments(
                        pending, transcript.source_language, target_language,
                        context=context,
                    )
                    if len(items) != len(pending):
                        raise ValueError(
                            "Translation provider returned a different segment count"
                        )
                    validated = {}
                    for segment, item in zip(pending, items):
                        corrected, translated = item
                        if (not isinstance(corrected, str) or not corrected.strip()
                                or not isinstance(translated, str)
                                or not translated.strip()):
                            raise ValueError(
                                "Translation provider returned an invalid response"
                            )
                        validated[segment.segment_id] = {
                            "source_text": corrected.strip(),
                            "translated_text": translated.strip(),
                        }
                    completed.update(validated)
                    if checkpoint:
                        _write_checkpoint(checkpoint, fingerprint, completed)
                    return
                except Exception as exc:
                    last_error = exc
            if len(pending) == 1:
                raise last_error
            midpoint = len(pending) // 2
            translate_batch(pending[:midpoint], context)
            translate_batch(pending[midpoint:], context)

        batch_size = 30
        for start in range(0, len(identified), batch_size):
            batch = identified[start:start + batch_size]
            context = identified[max(0, start - 5):start]
            translate_batch(batch, context)
        for segment in identified:
            item = completed[segment.segment_id]
            translated_segments.append(TranslatedLocalizationSegment(
                start=segment.start, end=segment.end,
                source_text=item["source_text"],
                translated_text=item["translated_text"],
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
