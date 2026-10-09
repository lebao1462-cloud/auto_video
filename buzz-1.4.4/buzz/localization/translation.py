from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
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


class RegisterReviewTranslationProvider(Protocol):
    """Optional continuity-editing pass for completed Vietnamese translations."""

    def review_translation_register(
        self, segments, source_language: str, target_language: str, *, context=(),
    ) -> list[tuple[str, str]]:
        """Return unchanged stable IDs paired with reviewed Vietnamese text."""
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


@dataclass(frozen=True)
class _TranslationContextSegment:
    """Completed source/translation context, never an item to translate."""

    start: float
    end: float
    text: str
    segment_id: str
    translated_text: str


@dataclass(frozen=True)
class _ReviewSegment:
    start: float
    end: float
    text: str
    segment_id: str
    translated_text: str


def _segment_id(index, segment) -> str:
    value = json.dumps(
        [index, segment.start, segment.end, segment.text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"segment-{index:06d}-{hashlib.sha256(value).hexdigest()[:12]}"


def _provider_identity(provider) -> dict:
    identity = {
        "provider": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "model": getattr(provider, "model", None),
    }
    policy_version = getattr(provider, "translation_policy_version", None)
    if policy_version is not None:
        identity["translation_policy_version"] = policy_version
    return identity


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


def _is_speakable_translated_text(value) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and any(character.isalnum() for character in value)
    )


_REGISTER_MIXES = (
    ("tao", "người"), ("tao", "cậu"), ("ta", "mày"),
    ("tôi", "người"), ("tôi", "mày"),
)


def _valid_reviewed_text(value) -> bool:
    """Reject deterministic, unsafe register mistakes without rewriting model text."""
    if not _is_speakable_translated_text(value):
        return False
    words = set(re.findall(r"[^\W\d_]+", value.casefold(), flags=re.UNICODE))
    if "hố" in words and "cha" in words:
        return False
    return not any(left in words and right in words for left, right in _REGISTER_MIXES)


def _review_checkpoint_fingerprint(raw_fingerprint: str, raw_items: dict, provider) -> str:
    """Bind review reuse to the exact raw output, without provider secrets."""
    payload = {
        "raw_translation_fingerprint": raw_fingerprint,
        "raw_segments": raw_items,
        "review_policy_version": getattr(
            provider, "translation_policy_version", None
        ),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _review_checkpoint_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("translation-register-checkpoint.json")


def _read_checkpoint(path: Path, fingerprint: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return {}
        items = payload.get("segments")
        if not isinstance(items, dict):
            return {}
        return {
            segment_id: item
            for segment_id, item in items.items()
            if isinstance(item, dict)
            and isinstance(item.get("source_text"), str)
            and item["source_text"].strip()
            and _is_speakable_translated_text(item.get("translated_text"))
        }
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


def _review_translation_register(
    transcript, provider, target_language: str, raw_fingerprint: str,
    translated_segments: list[TranslatedLocalizationSegment], checkpoint: Path | None,
) -> list[TranslatedLocalizationSegment]:
    review = getattr(provider, "review_translation_register", None)
    if not callable(review):
        return translated_segments
    items = tuple(_ReviewSegment(
        segment.start, segment.end, segment.source_text,
        _segment_id(index, transcript.segments[index]), segment.translated_text,
    ) for index, segment in enumerate(translated_segments))
    raw_items = {item.segment_id: {
        "source_text": item.text, "translated_text": item.translated_text,
    } for item in items}
    fingerprint = _review_checkpoint_fingerprint(raw_fingerprint, raw_items, provider)
    review_path = _review_checkpoint_path(checkpoint) if checkpoint else None
    completed = _read_checkpoint(review_path, fingerprint) if review_path else {}
    completed = {
        identifier: value for identifier, value in completed.items()
        if identifier in raw_items and value["source_text"] == raw_items[identifier]["source_text"]
        and _valid_reviewed_text(value["translated_text"])
    }
    positions = {item.segment_id: index for index, item in enumerate(items)}

    def context_for(batch):
        first = positions[batch[0].segment_id]
        return tuple(_TranslationContextSegment(
            item.start, item.end, item.text, item.segment_id,
            completed[item.segment_id]["translated_text"],
        ) for item in items[max(0, first - 12):first]
            if item.segment_id in completed)

    def review_batch(batch):
        pending = [item for item in batch if item.segment_id not in completed]
        if not pending:
            return
        last_error = None
        for _ in range(2):
            try:
                response = review(
                    pending, transcript.source_language, target_language,
                    context=context_for(pending),
                )
                if not isinstance(response, (list, tuple)) or len(response) != len(pending):
                    raise ValueError("Translation provider returned a different review segment count")
                expected = {item.segment_id for item in pending}
                validated = {}
                for result in response:
                    if not isinstance(result, (list, tuple)) or len(result) != 2:
                        raise ValueError("Translation provider returned an invalid review response")
                    identifier, translated = result
                    if identifier in validated or identifier not in expected or not _valid_reviewed_text(translated):
                        raise ValueError("Translation provider returned an invalid reviewed translation")
                    validated[identifier] = {
                        "source_text": raw_items[identifier]["source_text"],
                        "translated_text": translated.strip(),
                    }
                if set(validated) != expected:
                    raise ValueError("Translation provider returned missing review IDs")
                completed.update(validated)
                if review_path:
                    _write_checkpoint(review_path, fingerprint, completed)
                return
            except Exception as exc:
                last_error = exc
        if len(pending) == 1:
            raise last_error
        midpoint = len(pending) // 2
        review_batch(pending[:midpoint])
        review_batch(pending[midpoint:])

    for start in range(0, len(items), 120):
        review_batch(items[start:start + 120])
    return [TranslatedLocalizationSegment(
        item.start, item.end, raw_items[item.segment_id]["source_text"],
        completed[item.segment_id]["translated_text"],
    ) for item in items]


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
    raw_fingerprint = _checkpoint_fingerprint(transcript, provider, target_language)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if callable(getattr(provider, "translate_segments", None)):
        identified = tuple(
            _IdentifiedSegment(
                segment.start, segment.end, segment.text, _segment_id(index, segment)
            )
            for index, segment in enumerate(transcript.segments)
        )
        fingerprint = raw_fingerprint
        completed = _read_checkpoint(checkpoint, fingerprint) if checkpoint else {}

        positions = {item.segment_id: index for index, item in enumerate(identified)}

        def completed_context(batch):
            """Use recent completed Vietnamese lines as scene/register memory."""
            first = positions[batch[0].segment_id]
            context = []
            for item in identified[max(0, first - 8):first]:
                completed_item = completed.get(item.segment_id)
                if completed_item is None:
                    continue
                context.append(_TranslationContextSegment(
                    start=item.start,
                    end=item.end,
                    text=completed_item["source_text"],
                    segment_id=item.segment_id,
                    translated_text=completed_item["translated_text"],
                ))
            return tuple(context)

        def translate_batch(batch):
            pending = [item for item in batch if item.segment_id not in completed]
            if not pending:
                return
            last_error = None
            for _ in range(2):
                try:
                    items = provider.translate_segments(
                        pending, transcript.source_language, target_language,
                        context=completed_context(pending),
                    )
                    if len(items) != len(pending):
                        raise ValueError(
                            "Translation provider returned a different segment count"
                        )
                    validated = {}
                    for segment, item in zip(pending, items):
                        corrected, translated = item
                        if (not isinstance(corrected, str) or not corrected.strip()
                                or not _is_speakable_translated_text(translated)):
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
                translate = getattr(provider, "translate", None)
                if callable(translate):
                    try:
                        translated = translate(
                            pending[0].text,
                            source_language=transcript.source_language,
                            target_language=target_language,
                        )
                        if _is_speakable_translated_text(translated):
                            completed[pending[0].segment_id] = {
                                "source_text": pending[0].text,
                                "translated_text": translated.strip(),
                            }
                            if checkpoint:
                                _write_checkpoint(checkpoint, fingerprint, completed)
                            return
                    except Exception:
                        pass
                raise last_error
            midpoint = len(pending) // 2
            translate_batch(pending[:midpoint])
            translate_batch(pending[midpoint:])

        batch_size = 30
        for start in range(0, len(identified), batch_size):
            batch = identified[start:start + batch_size]
            translate_batch(batch)
        for segment in identified:
            item = completed[segment.segment_id]
            translated_segments.append(TranslatedLocalizationSegment(
                start=segment.start, end=segment.end,
                source_text=item["source_text"],
                translated_text=item["translated_text"],
            ))
        translated_segments = _review_translation_register(
            transcript, provider, target_language, raw_fingerprint, translated_segments, checkpoint,
        )
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
        if not _is_speakable_translated_text(translated_text):
            raise ValueError("Translation provider returned an invalid response")
        translated_segments.append(
            TranslatedLocalizationSegment(
                start=segment.start,
                end=segment.end,
                source_text=segment.text,
                translated_text=translated_text,
            )
        )

    translated_segments = _review_translation_register(
        transcript, provider, target_language, raw_fingerprint, translated_segments, checkpoint,
    )
    return TranslatedLocalizationTranscript(
        source_file=transcript.source_file,
        source_language=transcript.source_language,
        target_language=target_language,
        segments=tuple(translated_segments),
    )
