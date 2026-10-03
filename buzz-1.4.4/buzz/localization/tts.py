from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, ContextManager, Mapping, Protocol

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


_CHECKPOINT_VERSION = 1


def _json_identity(value):
    """Make provider settings deterministic without requiring JSON-only options."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_identity(item) for item in value]
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _provider_identity(provider: TTSProvider):
    identity = {
        "class": f"{type(provider).__module__}.{type(provider).__qualname__}",
    }
    cache_identity = getattr(provider, "tts_cache_identity", None)
    if callable(cache_identity):
        identity["settings"] = _json_identity(cache_identity())
    return identity


def _request_fingerprint(request: TTSRequest, provider: TTSProvider) -> str:
    payload = {
        "version": _CHECKPOINT_VERSION,
        "provider": _provider_identity(provider),
        "text": request.text,
        "language": request.language,
        "voice": request.voice,
        "options": _json_identity(request.options),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audio_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(request: TTSRequest) -> Path:
    stem = Path(request.output_file_stem)
    return stem.with_name(f"{stem.name}.localization-tts.json")


def _valid_result(result: object) -> TTSResult:
    if not isinstance(result, TTSResult):
        raise ValueError("TTS provider returned an invalid response")
    if not isinstance(result.audio_file, str) or not result.audio_file:
        raise ValueError("TTS provider returned an invalid audio file")
    audio_path = Path(result.audio_file)
    if not audio_path.is_file():
        raise ValueError("TTS provider audio file does not exist")
    if audio_path.stat().st_size <= 0:
        raise ValueError("TTS provider audio file is empty")
    if (
        not isinstance(result.audio_duration, (int, float))
        or isinstance(result.audio_duration, bool)
        or result.audio_duration <= 0
    ):
        raise ValueError("TTS provider returned an invalid audio duration")
    if not isinstance(result.provider, str) or not result.provider.strip():
        raise ValueError("TTS provider returned invalid provider metadata")
    return result


def _read_checkpoint(path: Path, fingerprint: str) -> TTSResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("version") != _CHECKPOINT_VERSION
                or payload.get("fingerprint") != fingerprint):
            return None
        audio_path = Path(payload["audio_file"])
        if (not audio_path.is_file() or audio_path.stat().st_size <= 0
                or audio_path.stat().st_size != payload["audio_size"]
                or _audio_digest(audio_path) != payload["audio_sha256"]):
            return None
        return _valid_result(TTSResult(
            audio_file=str(audio_path),
            audio_duration=payload["audio_duration"],
            provider=payload["provider"],
            voice=payload.get("voice"),
            metadata=payload.get("metadata", {}),
        ))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _write_checkpoint(path: Path, fingerprint: str, result: TTSResult) -> None:
    audio_path = Path(result.audio_file)
    payload = {
        "version": _CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "audio_file": str(audio_path),
        "audio_size": audio_path.stat().st_size,
        "audio_sha256": _audio_digest(audio_path),
        "audio_duration": float(result.audio_duration),
        "provider": result.provider,
        "voice": result.voice,
        "metadata": dict(result.metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def synthesize_segment(request: TTSRequest, provider: TTSProvider) -> TTSResult:
    """Synthesize or reuse one fully validated, persistent segment checkpoint."""
    checkpoint = _checkpoint_path(request)
    fingerprint = _request_fingerprint(request, provider)
    cached = _read_checkpoint(checkpoint, fingerprint)
    if cached is not None:
        return cached

    # A stale manifest cannot claim completion if replacement synthesis fails.
    checkpoint.unlink(missing_ok=True)
    result = _valid_result(provider.synthesize(request))
    _write_checkpoint(checkpoint, fingerprint, result)
    return result


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
    segment_context: Callable[[int, int], ContextManager[None]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SynthesizedLocalizationTranscript:
    """Synthesize one Vietnamese audio asset for every translated segment.

    ``segment_context`` is provider-neutral and wraps one segment only, so a
    caller can share global capacity between jobs without parallelizing a
    single job's segment requests.
    """
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

    segment_count = len(transcript.segments)
    for index, segment in enumerate(transcript.segments):
        context = (segment_context(index, segment_count)
                   if segment_context is not None else nullcontext())
        with context:
            result = synthesize_segment(
                TTSRequest(
                    text=segment.translated_text,
                    language=transcript.target_language,
                    output_file_stem=str(asset_directory / f"segment-{index:06d}"),
                    voice=voice,
                    options=provider_options,
                ),
                provider,
            )

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
        if progress_callback is not None:
            progress_callback(index + 1, segment_count)

    return SynthesizedLocalizationTranscript(
        source_file=transcript.source_file,
        source_language=transcript.source_language,
        target_language=transcript.target_language,
        segments=tuple(synthesized_segments),
    )
