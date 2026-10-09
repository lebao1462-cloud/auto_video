"""Sequential long-video processing with bounded temporary audio workspaces."""

from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
import hashlib
import json
import math
import os
import re
import threading
from typing import Callable

from buzz.localization.audio_mix import AudioMixingCancelled, AudioMixingError, _run_ffmpeg, probe_media_duration
from buzz.localization.final_render import FinalRenderResult
from buzz.localization.resources import ResourceCancelled
from buzz.localization.subtitles import SubtitleResult
from buzz.localization.transcript import (
    LocalizationSegment, LocalizationTranscript, LocalizationTranscriptionCancelled,
)
from buzz.localization.workflow import (
    LocalizationCancelled, LocalizationProgress, LocalizationStage,
    LocalizationWorkflowOptions, LocalizationWorkflowResult,
)

_TIME = re.compile(r"^(\d+:\d{2}:\d{2}[,.]\d{3}) --> (\d+:\d{2}:\d{2}[,.]\d{3})$")
_PART_MANIFEST = "localization-part-complete.json"
_PART_MANIFEST_VERSION = 1
_ASR_CHECKPOINT = "localization-asr-checkpoint.json"
_ASR_CHECKPOINT_VERSION = 1
_SECRET_NAMES = frozenset({
    "api_key", "apikey", "access_token", "openai_access_token", "password",
    "secret", "token",
})
_NON_OUTPUT_PROVIDER_FIELDS = frozenset({
    "rate_limit_fallback_models",
})


def _stable_value(value):
    """Return deterministic JSON data without object reprs or secret fields."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _stable_value(value.value)
    if isinstance(value, Path):
        return str(value.resolve())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _stable_value(getattr(value, item.name))
            for item in fields(value)
            if item.name.lower() not in _SECRET_NAMES
        }
    if isinstance(value, dict):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _SECRET_NAMES
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_stable_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _provider_identity(provider, *, tts: bool = False) -> dict:
    identity = {
        "type": f"{type(provider).__module__}.{type(provider).__qualname__}",
    }
    cache_identity = getattr(provider, "tts_cache_identity", None) if tts else None
    if callable(cache_identity):
        identity["settings"] = _stable_value(cache_identity())
    elif is_dataclass(provider):
        identity["settings"] = {
            # Gemini timeout affects request transport, not translated content.
            # Keep the historical 60-second identity so completed video parts
            # from older builds remain reusable after raising actual timeout.
            item.name: (
                60.0 if (
                    item.name == "timeout"
                    and type(provider).__name__ == "GeminiTranslationProvider"
                    and type(provider).__module__ == "buzz.localization.providers"
                ) else _stable_value(getattr(provider, item.name))
            )
            for item in fields(provider)
            if item.name.lower() not in _SECRET_NAMES
            and item.name not in _NON_OUTPUT_PROVIDER_FIELDS
        }
    else:
        configuration_names = {
            "model", "model_name", "base_url", "default_voice", "voice",
            "batch_size", "max_input_length", "max_new_tokens",
        }
        public = {
            name: value for name, value in vars(provider).items()
            if name in configuration_names and name.lower() not in _SECRET_NAMES
            and isinstance(value, (type(None), bool, int, float, str, tuple, list, dict))
        }
        if public:
            identity["settings"] = _stable_value(public)
    return identity


def _part_fingerprint(
    source: Path, index: int, start: float, chunk_seconds: float,
    transcription_options, model_path: str, translation_provider, tts_provider,
    options: LocalizationWorkflowOptions, asr_provider: str,
) -> str:
    source_stat = source.stat()
    payload = {
        "version": _PART_MANIFEST_VERSION,
        "source": {
            "path": str(source.resolve()),
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        "chunk": {"index": index, "start": start, "duration": chunk_seconds},
        "asr": {
            "provider": asr_provider,
            "model_path": str(Path(model_path).resolve()) if model_path else "",
            "options": _stable_value(transcription_options),
        },
        "translation": _provider_identity(translation_provider),
        "tts": {
            "provider": _provider_identity(tts_provider, tts=True),
            "voice": options.tts_voice,
            "options": _stable_value(dict(options.tts_options)),
        },
        # Still-frame support is deliberately omitted from the historical part
        # identity so normal completed parts (including p01 Parts 1-9) remain
        # reusable.  It only changes the exceptional overflow render path.
        "workflow": {
            key: value for key, value in _stable_value(
                replace(options, output_directory="<output>")
            ).items()
            # Rendering changes invalidate a completed part, but the render
            # options stay out of the earlier ASR/translation/TTS identities.
            if key != "still_frame_extension"
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _record_matches(path: Path, record) -> bool:
    if not isinstance(record, dict) or record.get("name") != path.name:
        return False
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if record.get("size") != path.stat().st_size:
        return False
    return _file_record(path)["sha256"] == record.get("sha256")


def _write_manifest_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _duration_matches(path: Path, expected: float) -> bool:
    """Check a prepared/rendered clip with the chunk duration tolerance."""
    try:
        actual = probe_media_duration(path)
    except (OSError, ValueError, AudioMixingError):
        return False
    return actual > 0 and abs(actual - expected) <= max(0.25, expected * 0.02)


def _reusable_prepared_clip(clip: Path, source: Path, chunk_seconds: float) -> bool:
    """A source edit always wins over a leftover intermediate clip."""
    try:
        if not clip.is_file() or clip.stat().st_size <= 0:
            return False
        if source.stat().st_mtime_ns > clip.stat().st_mtime_ns:
            return False
    except OSError:
        return False
    return _duration_matches(clip, chunk_seconds)


def _asr_fingerprint(clip: Path, transcription_options, model_path: str, asr_provider: str) -> str:
    stat = clip.stat()
    payload = {
        "version": _ASR_CHECKPOINT_VERSION,
        "clip": {
            "path": str(clip.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        },
        "asr": {
            "provider": asr_provider,
            "model_path": str(Path(model_path).resolve()) if model_path else "",
            "options": _stable_value(transcription_options),
        },
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _transcript_from_checkpoint(path: Path, fingerprint: str, clip: Path) -> LocalizationTranscript | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("version") != _ASR_CHECKPOINT_VERSION
                or data.get("fingerprint") != fingerprint
                or data.get("source_language") not in {"en", "zh"}
                or not isinstance(data.get("segments"), list)):
            return None
        segments = []
        for item in data["segments"]:
            start, end, text = item["start"], item["end"], item["text"]
            if (not isinstance(start, (int, float)) or isinstance(start, bool)
                    or not isinstance(end, (int, float)) or isinstance(end, bool)
                    or not math.isfinite(start) or not math.isfinite(end) or end < start
                    or not isinstance(text, str)):
                return None
            segments.append(LocalizationSegment(float(start), float(end), text))
        return LocalizationTranscript(data["source_language"], str(clip), tuple(segments))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_asr_checkpoint(path: Path, fingerprint: str, transcript: LocalizationTranscript) -> None:
    _write_manifest_atomic(path, {
        "version": _ASR_CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "source_language": transcript.source_language,
        "segments": [segment.to_dict() for segment in transcript.segments],
    })


def _legacy_transcript_from_checkpoints(
    part_dir: Path, clip: Path, transcription_options, asr_provider: str, translation_provider,
) -> LocalizationTranscript | None:
    """Recover only complete old checkpoints whose translation input can be proven."""
    from buzz.localization.translation import _checkpoint_fingerprint

    try:
        legacy_workspace = part_dir / "source_localization_work"
        translation = json.loads((legacy_workspace / "translation-checkpoint.json").read_text(encoding="utf-8"))
        timing = json.loads((legacy_workspace / "timing-recovery-checkpoint.json").read_text(encoding="utf-8"))
        items, bounds, recovered = (
            translation["segments"], timing["segment_bounds"], timing["shortened_text"],
        )
        count = timing["segment_count"]
        if (not isinstance(items, dict) or not isinstance(bounds, list)
                or not isinstance(recovered, dict) or not isinstance(count, int)
                or count < 1 or len(items) != count or len(bounds) != count
                or any(
                    not isinstance(key, str) or not key.isdigit()
                    or int(key) < 0 or int(key) >= count
                    or not isinstance(value, str) or not value.strip()
                    for key, value in recovered.items()
                )):
            return None
        ordered = []
        for index in range(count):
            matching = [value for key, value in items.items()
                        if isinstance(key, str) and key.startswith(f"segment-{index:06d}-")]
            if len(matching) != 1 or not isinstance(matching[0], dict):
                return None
            start, end = bounds[index]
            text = matching[0].get("source_text")
            if (not isinstance(start, (int, float)) or isinstance(start, bool)
                    or not isinstance(end, (int, float)) or isinstance(end, bool)
                    or not math.isfinite(start) or not math.isfinite(end) or end < start
                    or not isinstance(text, str) or not text.strip()):
                return None
            ordered.append(LocalizationSegment(float(start), float(end), text.strip()))
        explicit = getattr(transcription_options, "language", None)
        languages = [explicit] if explicit in {"en", "zh"} else (
            ["zh"] if asr_provider in {"paraformer", "paraformer-zh"} else ["en", "zh"]
        )
        for language in languages:
            candidate = LocalizationTranscript(language, str(clip), tuple(ordered))
            if _checkpoint_fingerprint(candidate, translation_provider, "vi") == translation.get("fingerprint"):
                return candidate
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _load_completed_part(
    manifest_path: Path, fingerprint: str, part_video: Path, subtitle: Path,
    chunk_seconds: float,
) -> tuple[Path, Path | None, float, float] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("version") != _PART_MANIFEST_VERSION
                or manifest.get("fingerprint") != fingerprint):
            return None
        if not _record_matches(part_video, manifest.get("video")):
            return None
        actual_duration = probe_media_duration(part_video)
        extension = manifest.get("extension_seconds", 0.0)
        if (not isinstance(extension, (int, float)) or isinstance(extension, bool)
                or extension < 0):
            return None
        # New manifests make extension explicit.  Old manifests retain their
        # historical strict zero-extension expectation.
        expected_duration = chunk_seconds + float(extension)
        if actual_duration <= 0 or abs(actual_duration - expected_duration) > 0.25:
            return None
        subtitle_record = manifest.get("subtitle")
        if subtitle_record is not None and not _record_matches(subtitle, subtitle_record):
            return None
        return part_video, subtitle if subtitle_record is not None else None, actual_duration, float(extension)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _shift_timestamp(value: str, seconds: float) -> str:
    hours, minutes, rest = value.split(":")
    whole, millis = re.split("[,.]", rest)
    total = max(0, round(seconds * 1000)) + (
        ((int(hours) * 60 + int(minutes)) * 60 + int(whole)) * 1000
        + int(millis)
    )
    hours, remainder = divmod(total, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{',' if ',' in value else '.'}{millis:03}"


def _merge_subtitles(parts: list[tuple[Path, float]], destination: Path, fmt: str) -> int:
    cues = []
    for path, offset in parts:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        current = []
        for line in [*lines, ""]:
            if line.strip():
                current.append(line)
            elif current:
                if current == ["WEBVTT"]:
                    current = []
                    continue
                if current[0].isdigit():
                    current.pop(0)
                if not current or not _TIME.fullmatch(current[0]):
                    raise ValueError(f"Invalid subtitle cue in {path}")
                match = _TIME.fullmatch(current[0])
                start = _shift_timestamp(match.group(1), offset)
                end = _shift_timestamp(match.group(2), offset)
                cues.append((start, end, "\n".join(current[1:])))
                current = []
    if not cues:
        raise ValueError("No subtitle cues were generated")
    if fmt == "srt":
        content = "\n\n".join(
            f"{i}\n{start} --> {end}\n{text}"
            for i, (start, end, text) in enumerate(cues, 1)
        ) + "\n"
    else:
        content = "WEBVTT\n\n" + "\n\n".join(
            f"{start} --> {end}\n{text}" for start, end, text in cues
        ) + "\n"
    destination.write_text(content, encoding="utf-8", newline="\n")
    return len(cues)


def _concat_line(path: Path) -> str:
    # FFmpeg concat demuxer accepts forward slashes on Windows.
    return "file '" + str(path.resolve()).replace("\\", "/").replace("'", "'\\''") + "'\n"


def localize_chunked_video(
    source: Path,
    duration: float,
    transcription_options,
    model_path: str,
    translation_provider,
    tts_provider,
    options: LocalizationWorkflowOptions,
    progress_callback: Callable[[LocalizationProgress], None] | None,
    cancel_event: threading.Event | None,
    transcribe_func,
    asr_provider: str = "whisper",
) -> LocalizationWorkflowResult:
    from buzz.localization.workflow import _localize_single_video, cleanup_localization_workspace

    count = math.ceil(duration / options.chunk_duration_seconds)
    output = Path(options.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / f"{source.stem}_localization_work"
    workspace.mkdir(parents=True, exist_ok=True)
    pieces = []
    subtitles = []
    offset = 0.0
    total_extension = 0.0
    steps_per_piece = 9 if options.use_background_separation else 8
    total = count * steps_per_piece + 1

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise LocalizationCancelled("Localization was cancelled")

    for index in range(count):
        check_cancel()
        chunk_start = index * options.chunk_duration_seconds
        chunk_seconds = min(options.chunk_duration_seconds, duration - chunk_start)
        part_dir = workspace / f"part_{index + 1:04}"
        part_dir.mkdir(exist_ok=True)
        clip = part_dir / "source.mp4"
        part_video = part_dir / "source.vi.mp4"
        part_subtitle = part_dir / f"source.vi.{options.subtitle_format}"
        manifest_path = part_dir / _PART_MANIFEST
        fingerprint = _part_fingerprint(
            source, index, chunk_start, chunk_seconds, transcription_options,
            model_path, translation_provider, tts_provider, options, asr_provider,
        )
        resumed = _load_completed_part(
            manifest_path, fingerprint, part_video, part_subtitle, chunk_seconds,
        )
        if resumed is not None:
            part_video, resumed_subtitle, part_duration, extension_seconds = resumed
            pieces.append(part_video)
            if resumed_subtitle is not None:
                subtitles.append((resumed_subtitle, offset))
            offset += part_duration
            total_extension += extension_seconds
            if progress_callback:
                progress_callback(LocalizationProgress(
                    LocalizationStage.RENDER, (index + 1) * steps_per_piece,
                    total, f"Part {index + 1}/{count}: reused completed part"
                ))
            continue

        # A stale marker must not survive a failed reprocessing attempt.
        manifest_path.unlink(missing_ok=True)
        if _reusable_prepared_clip(clip, source, chunk_seconds):
            if progress_callback:
                progress_callback(LocalizationProgress(
                    LocalizationStage.TRANSCRIBE, index * steps_per_piece,
                    total, f"Part {index + 1}/{count}: reused prepared video"
                ))
        else:
            if progress_callback:
                progress_callback(LocalizationProgress(
                    LocalizationStage.TRANSCRIBE, index * steps_per_piece,
                    total, f"Part {index + 1}/{count}: preparing video"
                ))
            try:
                with options.resource_scheduler.acquire(
                    "prepare", cancel_event=cancel_event,
                    waiting=(lambda: progress_callback(LocalizationProgress(
                        LocalizationStage.TRANSCRIBE, index * steps_per_piece,
                        total, f"Part {index + 1}/{count}: waiting for prepare resource",
                    ))) if progress_callback else None,
                ):
                    _run_ffmpeg([
                        "ffmpeg", "-y", "-nostdin", "-ss",
                        str(chunk_start), "-i", str(source),
                        "-t", str(chunk_seconds), "-map", "0:v:0", "-map", "0:a:0",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-threads", "4", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
                        "-ac", "2", "-avoid_negative_ts", "make_zero", str(clip),
                    ], cancel_event=cancel_event)
            except ResourceCancelled as exc:
                raise LocalizationCancelled(str(exc)) from exc
            except AudioMixingCancelled as exc:
                clip.unlink(missing_ok=True)
                raise LocalizationCancelled(str(exc)) from exc
        if not clip.is_file():
            raise RuntimeError(f"Cannot create video part {index + 1}")

        def part_progress(progress: LocalizationProgress):
            if progress_callback and progress.stage != LocalizationStage.COMPLETED:
                progress_callback(LocalizationProgress(
                    progress.stage, index * steps_per_piece + progress.completed_steps,
                    total, f"Part {index + 1}/{count}: {progress.message}"
                ))

        asr_checkpoint = part_dir / _ASR_CHECKPOINT
        asr_fingerprint = _asr_fingerprint(clip, transcription_options, model_path, asr_provider)
        transcript = _transcript_from_checkpoint(asr_checkpoint, asr_fingerprint, clip)
        if transcript is not None:
            if progress_callback:
                progress_callback(LocalizationProgress(
                    LocalizationStage.TRANSCRIBE, index * steps_per_piece,
                    total, f"Part {index + 1}/{count}: reused ASR checkpoint"
                ))
        else:
            transcript = _legacy_transcript_from_checkpoints(
                part_dir, clip, transcription_options, asr_provider, translation_provider,
            )
            if transcript is not None:
                _write_asr_checkpoint(asr_checkpoint, asr_fingerprint, transcript)
                if progress_callback:
                    progress_callback(LocalizationProgress(
                        LocalizationStage.TRANSCRIBE, index * steps_per_piece,
                        total, f"Part {index + 1}/{count}: reconstructed ASR checkpoint"
                    ))
            else:
                transcribe_kwargs = {"asr_provider": asr_provider} if asr_provider != "whisper" else {}
                if cancel_event is not None and asr_provider in {"paraformer", "paraformer-zh"}:
                    transcribe_kwargs["cancel_event"] = cancel_event
                try:
                    with options.resource_scheduler.acquire(
                        "asr", cancel_event=cancel_event,
                        waiting=(lambda: progress_callback(LocalizationProgress(
                            LocalizationStage.TRANSCRIBE, index * steps_per_piece,
                            total, f"Part {index + 1}/{count}: waiting for ASR resource",
                        ))) if progress_callback else None,
                    ):
                        transcript = transcribe_func(
                            str(clip), transcription_options, model_path, **transcribe_kwargs,
                        )
                except ResourceCancelled as exc:
                    raise LocalizationCancelled(str(exc)) from exc
                except LocalizationTranscriptionCancelled as exc:
                    raise LocalizationCancelled(str(exc)) from exc
                _write_asr_checkpoint(asr_checkpoint, asr_fingerprint, transcript)
        check_cancel()
        if transcript.segments:
            result = _localize_single_video(
                str(clip), transcription_options, model_path,
                translation_provider, tts_provider,
                replace(options, output_directory=str(part_dir),
                        chunk_duration_seconds=0),
                progress_callback=part_progress, cancel_event=cancel_event,
                transcribe_func=lambda *args, **kwargs: transcript, asr_provider=asr_provider,
            )
            part_video = Path(result.final_video.video_file)
            part_subtitle = Path(result.subtitle.subtitle_file)
            subtitles.append((part_subtitle, offset))
            cleanup_localization_workspace(result)
        else:
            # A speech-free part needs the same video/audio stream layout for concat.
            _run_ffmpeg([
                "ffmpeg", "-y", "-nostdin", "-i", str(clip), "-f", "lavfi",
                "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0",
                "-map", "1:a:0", "-t", str(probe_media_duration(clip)),
                "-c:v", "copy", "-c:a", options.render_options.audio_codec,
                "-b:a", options.render_options.audio_bitrate, str(part_video),
            ])
            if progress_callback:
                progress_callback(LocalizationProgress(
                    LocalizationStage.RENDER, (index + 1) * steps_per_piece,
                    total, f"Part {index + 1}/{count}: no speech; kept video with silence"
                ))
        check_cancel()
        part_duration = probe_media_duration(part_video)
        extension_seconds = max(0.0, part_duration - chunk_seconds)
        maximum_extension = min(
            options.still_frame_extension.max_seconds_total, chunk_seconds * 0.02,
        )
        if extension_seconds > maximum_extension + 0.25:
            raise RuntimeError(
                f"Rendered part {index + 1} grew by {extension_seconds:.2f}s, "
                f"above its explicit {maximum_extension:.2f}s extension bound"
            )
        if part_duration <= 0 or abs(part_duration - (chunk_seconds + extension_seconds)) > 0.25:
            raise RuntimeError(
                f"Rendered part {index + 1} duration {part_duration:.2f}s differs "
                f"from expected {chunk_seconds:.2f}s"
            )
        subtitle_record = None
        if transcript.segments:
            if not part_subtitle.is_file() or part_subtitle.stat().st_size <= 0:
                raise RuntimeError(
                    f"Rendered part {index + 1} subtitle is missing or empty"
                )
            subtitle_record = _file_record(part_subtitle)
        _write_manifest_atomic(manifest_path, {
            "version": _PART_MANIFEST_VERSION,
            "fingerprint": fingerprint,
            "video": _file_record(part_video),
            "subtitle": subtitle_record,
            # Explicitly record bounded timeline growth; resume validation must
            # never treat it as generic duration tolerance.
            "extension_seconds": round(extension_seconds, 3),
        })
        pieces.append(part_video)
        # The concat demuxer offsets each part by its rendered MP4 duration.
        offset += part_duration
        total_extension += extension_seconds
        clip.unlink(missing_ok=True)

    check_cancel()
    playlist = workspace / "parts.ffconcat"
    playlist.write_text("ffconcat version 1.0\n" + "".join(
        _concat_line(piece) for piece in pieces
    ), encoding="utf-8")
    temporary_video = workspace / "joined.mp4"
    subtitled_video = workspace / "joined_subtitled.mp4"
    temporary_subtitle = workspace / f"joined.{options.subtitle_format}"
    final_video = output / f"{source.stem}.vi.mp4"
    final_subtitle = output / f"{source.stem}.vi.{options.subtitle_format}"
    final_audio = workspace / "localized_audio.m4a"
    try:
        cues = _merge_subtitles(subtitles, temporary_subtitle, options.subtitle_format)
        _run_ffmpeg([
            "ffmpeg", "-y", "-nostdin", "-safe", "0", "-f", "concat",
            "-i", str(playlist), "-map", "0:v:0", "-map", "0:a:0",
            "-c", "copy", "-movflags", "+faststart", str(temporary_video),
        ])
        actual_duration = probe_media_duration(temporary_video)
        expected_duration = duration + total_extension
        if abs(actual_duration - expected_duration) > max(0.5, count * 0.12):
            raise RuntimeError(
                f"Joined video duration {actual_duration:.2f}s differs from "
                f"expected {expected_duration:.2f}s (source {duration:.2f}s plus "
                f"{total_extension:.2f}s planned extension)"
            )
        if options.render_options.subtitle_mode == "soft":
            # Remux the single global subtitle track. Concatenated mov_text
            # packets otherwise gain an AAC priming offset at each join.
            _run_ffmpeg([
                "ffmpeg", "-y", "-nostdin", "-i", str(temporary_video),
                "-i", str(temporary_subtitle), "-map", "0:v:0",
                "-map", "0:a:0", "-map", "1:0", "-c:v", "copy",
                "-c:a", "copy", "-c:s", "mov_text",
                "-metadata:s:s:0", "language=vie",
                "-disposition:s:0", "default",
                "-movflags", "+faststart", str(subtitled_video),
            ])
        assembled_video = (
            subtitled_video if options.render_options.subtitle_mode == "soft"
            else temporary_video
        )
        _run_ffmpeg([
            "ffmpeg", "-y", "-nostdin", "-i", str(assembled_video),
            "-map", "0:a:0", "-c:a", "copy", str(final_audio),
        ])
        check_cancel()
        assembled_video.replace(final_video)
        temporary_subtitle.replace(final_subtitle)
    finally:
        temporary_video.unlink(missing_ok=True)
        subtitled_video.unlink(missing_ok=True)
        temporary_subtitle.unlink(missing_ok=True)

    playlist.unlink(missing_ok=True)
    if progress_callback:
        progress_callback(LocalizationProgress(
            LocalizationStage.COMPLETED, total, total, "Localization completed"
        ))
    return LocalizationWorkflowResult(
        final_video=FinalRenderResult(
            video_file=str(final_video), subtitle_mode=options.render_options.subtitle_mode,
            audio_replaced=True, subtitle_included=options.render_options.subtitle_mode != "none",
        ),
        subtitle=SubtitleResult(str(final_subtitle), options.subtitle_format, cues),
        localized_audio_file=str(final_audio),
        workspace=str(workspace),
    )
