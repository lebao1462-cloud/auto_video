from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
from typing import Callable, Mapping
from contextlib import contextmanager

from buzz.localization.audio_mix import (
    AudioMixingOptions,
    extract_audio_track,
    mix_localized_audio,
    probe_media_duration,
    separate_background_with_demucs,
)
from buzz.localization.final_render import FinalRenderOptions, FinalRenderResult, render_localized_mp4
from buzz.localization.resources import DEFAULT_RESOURCE_SCHEDULER, ResourceCancelled
from buzz.localization.subtitles import SubtitleOptions, SubtitleResult, write_subtitles
from buzz.localization.timing import (
    TimingPolicy,
    TimingSynchronizationError,
    _MAX_PLAYBACK_RATE_RELATIVE_TOLERANCE,
    synchronize_for_localization,
)
from buzz.localization.transcript import LocalizationTranscript, transcribe_for_localization
from buzz.localization.translation import (
    TranslatedLocalizationSegment,
    TranslatedLocalizationTranscript,
    TranslationProvider,
    translate_for_localization,
)
from buzz.localization.tts import (
    SynthesizedLocalizationTranscript,
    TTSProvider,
    TTSRequest,
    TTSResult,
    synthesize_segment,
    synthesize_for_localization,
)
from buzz.transcriber.transcriber import TranscriptionOptions

logger = logging.getLogger(__name__)


class LocalizationCancelled(RuntimeError):
    pass


class LocalizationStage(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    TTS = "tts"
    TIMING = "timing"
    EXTRACT_AUDIO = "extract_audio"
    SEPARATE_BACKGROUND = "separate_background"
    MIX_AUDIO = "mix_audio"
    SUBTITLES = "subtitles"
    RENDER = "render"
    COMPLETED = "completed"


@dataclass(frozen=True)
class LocalizationProgress:
    stage: LocalizationStage
    completed_steps: float
    total_steps: int
    message: str

    @property
    def fraction(self) -> float:
        return self.completed_steps / self.total_steps if self.total_steps else 0.0


@dataclass(frozen=True)
class LocalizationWorkflowOptions:
    output_directory: str
    chunk_duration_seconds: int = 900
    use_background_separation: bool = False
    subtitle_format: str = "srt"
    tts_voice: str | None = None
    tts_options: Mapping[str, object] = field(default_factory=dict)
    timing_policy: TimingPolicy = field(default_factory=TimingPolicy)
    timing_recovery_attempts: int = 4
    audio_options: AudioMixingOptions = field(default_factory=AudioMixingOptions)
    subtitle_options: SubtitleOptions = field(default_factory=SubtitleOptions)
    render_options: FinalRenderOptions = field(default_factory=FinalRenderOptions)
    resource_scheduler: object = DEFAULT_RESOURCE_SCHEDULER

    def __post_init__(self):
        if self.subtitle_format not in {"srt", "vtt"}:
            raise ValueError("subtitle_format must be 'srt' or 'vtt'")
        if not self.output_directory:
            raise ValueError("output_directory is required")
        if self.chunk_duration_seconds < 0:
            raise ValueError("chunk_duration_seconds must be non-negative")
        if (isinstance(self.timing_recovery_attempts, bool)
                or self.timing_recovery_attempts < 0):
            raise ValueError("timing_recovery_attempts must be non-negative")


@dataclass(frozen=True)
class LocalizationWorkflowResult:
    final_video: FinalRenderResult
    subtitle: SubtitleResult
    localized_audio_file: str
    workspace: str

    def to_dict(self) -> dict:
        return {
            "final_video": self.final_video.to_dict(),
            "subtitle": self.subtitle.to_dict(),
            "localized_audio_file": self.localized_audio_file,
            "workspace": self.workspace,
        }


def _coalesce_trailing_tiny_translation(
    transcript: TranslatedLocalizationTranscript, media_end: float,
) -> TranslatedLocalizationTranscript:
    """Merge a tiny final utterance into its predecessor when it ends a clip."""
    if len(transcript.segments) < 2:
        return transcript
    previous, final = transcript.segments[-2:]
    final_duration = float(final.end) - float(final.start)
    trailing_media = float(media_end) - float(final.end)
    gap = float(final.start) - float(previous.end)
    combined_span = float(final.end) - float(previous.start)
    if not (
        final_duration < 1.0
        and 0 <= trailing_media <= 0.25
        and 0 <= gap <= 0.40
        and combined_span <= 10.0
    ):
        return transcript

    source_separator = "" if transcript.source_language == "zh" else " "
    merged = TranslatedLocalizationSegment(
        start=previous.start,
        end=final.end,
        source_text=previous.source_text.rstrip() + source_separator + final.source_text.lstrip(),
        translated_text=previous.translated_text.rstrip() + " " + final.translated_text.lstrip(),
    )
    return replace(transcript, segments=(*transcript.segments[:-2], merged))


def _checkpoint_value(value):
    """Make deterministic, non-secret option data suitable for a checkpoint."""
    if isinstance(value, Mapping):
        return {
            str(key): _checkpoint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not any(word in str(key).lower() for word in ("key", "secret", "token", "password"))
        }
    if isinstance(value, (list, tuple)):
        return [_checkpoint_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _timing_recovery_fingerprint(
    translated: TranslatedLocalizationTranscript,
    options: LocalizationWorkflowOptions,
    media_end: float | None,
) -> str:
    """Fingerprint baseline text and every non-secret setting that changes TTS fit."""
    payload = {
        "segments": [segment.to_dict() for segment in translated.segments],
        "source_file": translated.source_file,
        "source_language": translated.source_language,
        "target_language": translated.target_language,
        "media_end": media_end,
        "tts_voice": options.tts_voice,
        "tts_options": _checkpoint_value(options.tts_options),
        "timing_policy": {
            "min_playback_rate": options.timing_policy.min_playback_rate,
            "max_playback_rate": options.timing_policy.max_playback_rate,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_speakable_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and any(char.isalnum() for char in value)


def _valid_timing_recovery_entries(entries: object, segment_count: int) -> bool:
    if not isinstance(entries, dict):
        return False
    for raw_index, text in entries.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return False
        if (not isinstance(raw_index, str) or str(index) != raw_index
                or not 0 <= index < segment_count or not _is_speakable_text(text)):
            return False
    return True


def _load_timing_recovery_checkpoint(
    path: Path, fingerprint: str, translated: TranslatedLocalizationTranscript,
) -> TranslatedLocalizationTranscript:
    """Apply only a complete, matching checkpoint; malformed files are disposable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["shortened_text"]
        expected_bounds = [[segment.start, segment.end] for segment in translated.segments]
        if (data.get("fingerprint") != fingerprint or data.get("segment_count") != len(translated.segments)
                or data.get("segment_bounds") != expected_bounds
                or not _valid_timing_recovery_entries(entries, len(translated.segments))):
            return translated
        segments = list(translated.segments)
        for raw_index, text in entries.items():
            index = int(raw_index)
            segments[index] = replace(segments[index], translated_text=text.strip())
        return replace(translated, segments=tuple(segments))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.info("Ignoring invalid timing-recovery checkpoint: %s", path)
        return translated


def _persist_timing_recovery_checkpoint(
    path: Path, fingerprint: str, segments, index: int, shortened_text: str,
) -> None:
    """Atomically save each successful recovery so an interrupted run can resume."""
    entries = {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing.get("fingerprint") == fingerprint
                and existing.get("segment_count") == len(segments)
                and existing.get("segment_bounds") == [
                    [segment.start, segment.end] for segment in segments
                ]
                and _valid_timing_recovery_entries(existing.get("shortened_text"), len(segments))):
            entries = existing["shortened_text"]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    entries[str(index)] = shortened_text
    payload = {
        "fingerprint": fingerprint,
        "segment_count": len(segments),
        "segment_bounds": [[segment.start, segment.end] for segment in segments],
        "shortened_text": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _synchronize_with_translation_recovery(
    synthesized: SynthesizedLocalizationTranscript,
    translation_provider: TranslationProvider,
    tts_provider: TTSProvider,
    options: LocalizationWorkflowOptions,
    *,
    media_end: float | None = None,
    cancel_event: threading.Event | None = None,
    waiting_callback: Callable[[str], None] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_fingerprint: str | None = None,
):
    current = synthesized
    attempts_by_segment: dict[int, int] = {}

    def overflows():
        """Find every rate overflow using the synchronizer's natural-gap rules."""
        failures = []
        for index, segment in enumerate(current.segments):
            next_start = (
                float(current.segments[index + 1].start)
                if index + 1 < len(current.segments)
                else media_end if media_end is not None else float(segment.end)
            )
            available_duration = next_start - float(segment.start)
            required_rate = float(segment.audio_duration) / available_duration
            tolerated_max_rate = options.timing_policy.max_playback_rate * (
                1.0 + _MAX_PLAYBACK_RATE_RELATIVE_TOLERANCE
            )
            if required_rate > tolerated_max_rate:
                failures.append((index, required_rate))
        return failures

    def regenerate(segment, shortened: str):
        with options.resource_scheduler.acquire(
            "tts", cancel_event=cancel_event,
            waiting=(lambda: waiting_callback("TTS")) if waiting_callback else None,
        ):
            result = synthesize_segment(TTSRequest(
                text=shortened,
                language=synthesized.target_language,
                output_file_stem=str(Path(segment.audio_file).with_suffix("")),
                voice=options.tts_voice,
                options=dict(options.tts_options),
            ), tts_provider)
        if (not isinstance(result, TTSResult)
                or not isinstance(result.audio_file, str)
                or not result.audio_file
                or not Path(result.audio_file).is_file()
                or not isinstance(result.audio_duration, (int, float))
                or isinstance(result.audio_duration, bool)
                or result.audio_duration <= 0
                or not isinstance(result.provider, str)
                or not result.provider.strip()):
            raise ValueError("TTS provider returned an invalid recovery response")
        replacement = replace(
            segment, translated_text=shortened, audio_file=result.audio_file,
            audio_duration=float(result.audio_duration), provider=result.provider,
            voice=result.voice, provider_metadata=dict(result.metadata),
        )
        return replacement

    while True:
        try:
            return synchronize_for_localization(
                current, options.timing_policy, media_end=media_end
            )
        except TimingSynchronizationError as exc:
            failures = overflows() if exc.segment_index is not None else []
            index = exc.segment_index
            shorten = getattr(translation_provider, "shorten_translation", None)
            batch_shorten = getattr(translation_provider, "shorten_translations", None)
            attempts = attempts_by_segment.get(index, 0) if index is not None else 0
            if (not failures or (not callable(batch_shorten) and not callable(shorten))
                    or any(attempts_by_segment.get(item_index, 0) >= options.timing_recovery_attempts
                           for item_index, _ in failures)):
                detail = (
                    f" after {attempts} concise-translation retries"
                    if (callable(batch_shorten) or callable(shorten)) and index is not None else
                    "; the active translation provider cannot shorten individual segments"
                )
                raise TimingSynchronizationError(
                    f"{exc}{detail}. Segment audio was not cut and the hard playback-rate "
                    "ceiling remains enforced.",
                    segment_index=index,
                    required_rate=exc.required_rate,
                    max_playback_rate=exc.max_playback_rate,
                ) from exc

            for start in range(0, len(failures), 30):
                batch = failures[start:start + 30]
                if callable(batch_shorten):
                    requests = [
                        {"id": f"segment-{item_index:06d}",
                         "source_text": current.segments[item_index].source_text,
                         "translated_text": current.segments[item_index].translated_text,
                         "required_rate": required_rate,
                         "max_playback_rate": options.timing_policy.max_playback_rate}
                        for item_index, required_rate in batch
                    ]
                    with options.resource_scheduler.acquire(
                        "gemini", cancel_event=cancel_event,
                        waiting=(lambda: waiting_callback("Gemini")) if waiting_callback else None,
                    ):
                        shortened_items = batch_shorten(
                            segments=requests, source_language=current.source_language,
                            target_language=current.target_language,
                        )
                    if not isinstance(shortened_items, (list, tuple)) or len(shortened_items) != len(batch):
                        raise ValueError("Translation provider returned an invalid shortened batch")
                else:
                    with options.resource_scheduler.acquire(
                        "gemini", cancel_event=cancel_event,
                        waiting=(lambda: waiting_callback("Gemini")) if waiting_callback else None,
                    ):
                        shortened_items = [shorten(
                            source_text=current.segments[item_index].source_text,
                            translated_text=current.segments[item_index].translated_text,
                            source_language=current.source_language,
                            target_language=current.target_language,
                            required_rate=required_rate,
                            max_playback_rate=options.timing_policy.max_playback_rate,
                        ) for item_index, required_rate in batch]
                jobs = []
                for (item_index, _), shortened in zip(batch, shortened_items):
                    if not isinstance(shortened, str) or not shortened.strip():
                        raise ValueError("Translation provider returned an invalid shortened segment")
                    jobs.append((item_index, current.segments[item_index], shortened.strip()))

                # Do not queue later Edge requests: failure must stop this batch now.
                for item_index, segment, shortened in jobs:
                    logger.info("Timing-recovery TTS starting for segment %d", item_index)
                    try:
                        replacement = regenerate(segment, shortened)
                    except Exception:
                        logger.info("Timing-recovery TTS failed for segment %d", item_index, exc_info=True)
                        raise
                    segments = list(current.segments)
                    segments[item_index] = replacement
                    current = replace(current, segments=tuple(segments))
                    attempts_by_segment[item_index] = attempts_by_segment.get(item_index, 0) + 1
                    if checkpoint_path is not None and checkpoint_fingerprint is not None:
                        _persist_timing_recovery_checkpoint(
                            checkpoint_path, checkpoint_fingerprint, current.segments,
                            item_index, shortened,
                        )
                    logger.info("Timing-recovery TTS succeeded for segment %d", item_index)


def _localize_single_video(
    source_video_file: str,
    transcription_options: TranscriptionOptions,
    model_path: str,
    translation_provider: TranslationProvider,
    tts_provider: TTSProvider,
    options: LocalizationWorkflowOptions,
    *,
    progress_callback: Callable[[LocalizationProgress], None] | None = None,
    cancel_event: threading.Event | None = None,
    transcribe_func: Callable[..., LocalizationTranscript] = transcribe_for_localization,
    asr_provider: str = "whisper",
    source_duration: float | None = None,
) -> LocalizationWorkflowResult:
    source = Path(source_video_file)
    if not source.is_file():
        raise ValueError(f"Input media file does not exist: {source_video_file}")

    output_dir = Path(options.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / f"{source.stem}_localization_work"
    workspace.mkdir(parents=True, exist_ok=True)
    if source_duration is None:
        source_duration = probe_media_duration(source)

    steps = 9 if options.use_background_separation else 8
    done = 0

    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise LocalizationCancelled("Localization was cancelled")

    def report(stage: LocalizationStage, message: str):
        nonlocal done
        check_cancelled()
        done += 1
        if progress_callback is not None:
            progress_callback(LocalizationProgress(stage, done, steps, message))

    def report_tts_segment(completed: int, total: int):
        check_cancelled()
        # The final stage report is retained for one-segment jobs.  Multi-
        # segment jobs get intermediate updates without changing that report.
        if progress_callback is not None and total > 1:
            progress_callback(LocalizationProgress(
                LocalizationStage.TTS,
                done + completed / total,
                steps,
                f"Tạo giọng {completed}/{total}",
            ))

    def waiting(stage: LocalizationStage, resource: str):
        if progress_callback is not None:
            progress_callback(LocalizationProgress(stage, done, steps, f"Đang chờ tài nguyên: {resource}"))

    @contextmanager
    def gated(resource: str, stage: LocalizationStage, label: str):
        try:
            with options.resource_scheduler.acquire(resource, cancel_event=cancel_event, waiting=lambda: waiting(stage, label)):
                yield
        except ResourceCancelled as exc:
            raise LocalizationCancelled(str(exc)) from exc

    check_cancelled()
    transcribe_kwargs = {"asr_provider": asr_provider} if asr_provider != "whisper" else {}
    with gated("asr", LocalizationStage.TRANSCRIBE, "Nhận dạng"):
        transcript = transcribe_func(str(source), transcription_options, model_path, **transcribe_kwargs)
    report(LocalizationStage.TRANSCRIBE, "Source transcription completed")

    with gated("gemini", LocalizationStage.TRANSLATE, "Gemini"):
        translated = translate_for_localization(transcript, translation_provider, checkpoint_path=workspace / "translation-checkpoint.json")
    report(LocalizationStage.TRANSLATE, "Vietnamese translation completed")

    translated = _coalesce_trailing_tiny_translation(translated, source_duration)
    recovery_checkpoint = workspace / "timing-recovery-checkpoint.json"
    recovery_fingerprint = _timing_recovery_fingerprint(translated, options, source_duration)
    translated = _load_timing_recovery_checkpoint(
        recovery_checkpoint, recovery_fingerprint, translated,
    )

    tts_dir = workspace / "tts"

    def tts_segment_context(_index: int, _total: int):
        return options.resource_scheduler.acquire(
            "tts",
            cancel_event=cancel_event,
            waiting=lambda: waiting(LocalizationStage.TTS, "TTS"),
        )

    try:
        synthesized = synthesize_for_localization(
            translated,
            tts_provider,
            tts_dir,
            voice=options.tts_voice,
            options=dict(options.tts_options),
            segment_context=tts_segment_context,
            progress_callback=report_tts_segment,
        )
    except ResourceCancelled as exc:
        raise LocalizationCancelled(str(exc)) from exc
    report(LocalizationStage.TTS, "Vietnamese TTS completed")

    try:
        timed = _synchronize_with_translation_recovery(
            synthesized, translation_provider, tts_provider, options,
            media_end=source_duration,
            cancel_event=cancel_event,
            waiting_callback=lambda resource: waiting(LocalizationStage.TIMING, resource),
            checkpoint_path=recovery_checkpoint,
            checkpoint_fingerprint=recovery_fingerprint,
        )
    except ResourceCancelled as exc:
        raise LocalizationCancelled(str(exc)) from exc
    report(LocalizationStage.TIMING, "Speech timing plan completed")

    with gated("render", LocalizationStage.RENDER, "Render"):
        extracted_audio = workspace / "source_audio.wav"
        extract_audio_track(
            source, extracted_audio,
            sample_rate=options.audio_options.sample_rate,
            channels=options.audio_options.channels,
        )
        report(LocalizationStage.EXTRACT_AUDIO, "Source audio extracted")

        background_audio = None
        if options.use_background_separation:
            background_audio = workspace / "background.wav"
            separate_background_with_demucs(extracted_audio, background_audio)
            report(LocalizationStage.SEPARATE_BACKGROUND, "Background audio separated")

        localized_audio = workspace / "localized_audio.wav"
        mix_localized_audio(timed, localized_audio, background_audio_file=background_audio,
                            options=options.audio_options, target_duration=source_duration)
        report(LocalizationStage.MIX_AUDIO, "Localized audio mixed")
        subtitle_file = output_dir / f"{source.stem}.vi.{options.subtitle_format}"
        subtitle_result = write_subtitles(timed, subtitle_file, format=options.subtitle_format,
                                          options=options.subtitle_options)
        report(LocalizationStage.SUBTITLES, "Vietnamese subtitles created")
        final_file = output_dir / f"{source.stem}.vi.mp4"
        final_result = render_localized_mp4(
            source, localized_audio, final_file,
            subtitle_file=subtitle_file if options.render_options.subtitle_mode != "none" else None,
            options=options.render_options,
        )
        report(LocalizationStage.RENDER, "Final MP4 rendered")

    if progress_callback is not None:
        progress_callback(
            LocalizationProgress(
                LocalizationStage.COMPLETED,
                steps,
                steps,
                "Localization completed",
            )
        )

    return LocalizationWorkflowResult(
        final_video=final_result,
        subtitle=subtitle_result,
        localized_audio_file=str(localized_audio),
        workspace=str(workspace),
    )


def localize_video(
    source_video_file: str,
    transcription_options: TranscriptionOptions,
    model_path: str,
    translation_provider: TranslationProvider,
    tts_provider: TTSProvider,
    options: LocalizationWorkflowOptions,
    *,
    progress_callback: Callable[[LocalizationProgress], None] | None = None,
    cancel_event: threading.Event | None = None,
    transcribe_func: Callable[..., LocalizationTranscript] = transcribe_for_localization,
    asr_provider: str = "whisper",
) -> LocalizationWorkflowResult:
    source = Path(source_video_file)
    if not source.is_file():
        raise ValueError(f"Input media file does not exist: {source_video_file}")
    if cancel_event is not None and cancel_event.is_set():
        raise LocalizationCancelled("Localization was cancelled")
    duration = probe_media_duration(source)
    if options.chunk_duration_seconds:
        if duration > options.chunk_duration_seconds:
            from buzz.localization.chunked import localize_chunked_video
            return localize_chunked_video(
                source, duration, transcription_options, model_path,
                translation_provider, tts_provider, options, progress_callback,
                cancel_event, transcribe_func, asr_provider,
            )
    return _localize_single_video(
        source_video_file, transcription_options, model_path,
        translation_provider, tts_provider, options,
        progress_callback=progress_callback, cancel_event=cancel_event,
        transcribe_func=transcribe_func, asr_provider=asr_provider,
        source_duration=duration,
    )


def cleanup_localization_workspace(result: LocalizationWorkflowResult) -> None:
    workspace = Path(result.workspace)
    if workspace.is_dir():
        shutil.rmtree(workspace)
