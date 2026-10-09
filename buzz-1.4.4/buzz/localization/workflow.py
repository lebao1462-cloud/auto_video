from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import hashlib
import json
import logging
import math
import os
import re
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
from buzz.localization.still_frame import (
    StillFrameExtensionError, StillFrameExtensionOptions,
    plan_still_frame_extensions, probe_video_fps, render_still_frame_video,
)
from buzz.localization.resources import DEFAULT_RESOURCE_SCHEDULER, ResourceCancelled
from buzz.localization.subtitles import SubtitleOptions, SubtitleResult, write_subtitles
from buzz.localization.timing import (
    TimingPolicy,
    TimingSynchronizationError,
    _MAX_PLAYBACK_RATE_RELATIVE_TOLERANCE,
    synchronize_for_localization,
)
from buzz.localization.transcript import (
    LocalizationTranscript, LocalizationTranscriptionCancelled,
    transcribe_for_localization,
)
from buzz.localization.translation import (
    TranslatedLocalizationSegment,
    TranslatedLocalizationTranscript,
    TranslationProvider,
    translate_for_localization,
)
from buzz.localization.providers import (
    EdgeTTSProvider, GeminiTranslationProvider, LocalizationProviderError,
)
from buzz.localization.tts import (
    SynthesizedLocalizationTranscript,
    TTSBatchFallbackRequired,
    TTSProvider,
    TTSRequest,
    TTSResult,
    TTSServiceTemporarilyUnavailable,
    _synthesize_batch_with_checkpoints,
    synthesize_segment,
    synthesize_for_localization,
)
from buzz.transcriber.transcriber import TranscriptionOptions

logger = logging.getLogger(__name__)

# Recovery calls include source and translated text; small batches avoid
# repeatedly stalling many measured overflows behind one slow request.
TIMING_RECOVERY_BATCH_SIZE = 12
TIMING_RECOVERY_TEMPORARY_TTS_RETRIES = 3
# After the provider tries alternate configured Gemini models, wait outside the
# global Gemini resource semaphore before attempting the pending batch again.
GEMINI_TIMING_TRANSIENT_RETRY_DELAYS = (15.0, 30.0, 60.0, 120.0, 180.0)


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
    # Experimental: only used after all concise-translation/TTS recovery is exhausted.
    still_frame_extension: StillFrameExtensionOptions = field(default_factory=StillFrameExtensionOptions)
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


def _retry_gemini_timing_transient_failure(
    operation: Callable[[], object],
    *,
    cancel_event: threading.Event | None = None,
    waiting_callback: Callable[[str], None] | None = None,
    max_retries: int | None = None,
):
    """Retry transient Gemini timing calls without repeating finished TTS work.

    The operation includes resource acquisition so the slot is released while
    waiting. Provider-level model fallback runs before each cooldown. Auth,
    malformed response and daily quota failures must not loop forever.
    """
    failures = 0
    if max_retries is None:
        # Keep the generic helper's historic retry budget even when tests or
        # deployments collapse the delay schedule to a single value.
        max_retries = 5
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise LocalizationCancelled("Localization was cancelled")
        try:
            return operation()
        except LocalizationProviderError as exc:
            message = str(exc).upper()
            if not GeminiTranslationProvider._is_gemini_transient_error(exc):
                raise
            if failures >= max_retries:
                raise
            delay = GEMINI_TIMING_TRANSIENT_RETRY_DELAYS[
                min(failures, len(GEMINI_TIMING_TRANSIENT_RETRY_DELAYS) - 1)
            ]
            failures += 1
            logger.warning(
                "Gemini timing request temporarily failed; retry %d in %.0fs: %s",
                failures, delay, exc,
            )
            if (waiting_callback is not None
                    and GeminiTranslationProvider._is_gemini_dns_error(exc)):
                waiting_callback(
                    f"Gemini DNS/network temporarily unavailable; retry {failures} in {int(delay)}s"
                )
            if (waiting_callback is not None
                    and not GeminiTranslationProvider._is_gemini_dns_error(exc)):
                waiting_callback(
                    f"Gemini 504/timeout, tự thử lại lần {failures} sau {int(delay)} giây"
                )
            # A user click on Cancel must interrupt the backoff immediately.
            sleeper = cancel_event if cancel_event is not None else threading.Event()
            if sleeper.wait(delay) and cancel_event is not None:
                raise LocalizationCancelled("Localization was cancelled")


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
    source_video: Path | None = None,
):
    current = synthesized
    attempts_by_segment: dict[int, int] = {}
    final_edge_rate_attempts: dict[int, set[int]] = {}

    def wait_for_temporary_tts(exc: TTSServiceTemporarilyUnavailable) -> None:
        if waiting_callback is not None:
            waiting_callback(
                f"Edge TTS tạm thời không khả dụng, tự thử lại sau {math.ceil(exc.retry_after_seconds)} giây"
            )
        if cancel_event is not None and cancel_event.wait(exc.retry_after_seconds):
            raise LocalizationCancelled("Localization was cancelled")
        if cancel_event is None:
            threading.Event().wait(exc.retry_after_seconds)

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
            # Credit only an immediately preceding physical source gap. This
            # mirrors the timing plan enough to avoid needless Gemini calls;
            # a prior utterance that extends beyond its subtitle owns that gap.
            if index:
                prior = current.segments[index - 1]
                prior_audio_end = float(prior.start) + min(
                    float(prior.audio_duration),
                    max(0.0, float(segment.start) - float(prior.start)),
                )
                physical_gap = max(0.0, float(segment.start) - float(prior.end))
                if prior_audio_end <= float(prior.end) + 1e-9:
                    available_duration += max(0.0, min(0.25, physical_gap - 0.005))
            required_rate = float(segment.audio_duration) / available_duration
            tolerated_max_rate = options.timing_policy.max_playback_rate * (
                1.0 + _MAX_PLAYBACK_RATE_RELATIVE_TOLERANCE
            )
            if required_rate > tolerated_max_rate:
                failures.append((index, required_rate))
        return failures

    def regenerate(
        segment, shortened: str, *, tts_options: Mapping[str, object] | None = None,
        output_suffix: str = "",
    ):
        for temporary_retry in range(TIMING_RECOVERY_TEMPORARY_TTS_RETRIES + 1):
            try:
                with options.resource_scheduler.acquire(
                    "tts", cancel_event=cancel_event,
                    waiting=(lambda: waiting_callback("TTS")) if waiting_callback else None,
                ):
                    result = synthesize_segment(TTSRequest(
                        text=shortened,
                        language=synthesized.target_language,
                        output_file_stem=(
                            str(Path(segment.audio_file).with_suffix("")) + output_suffix
                        ),
                        voice=options.tts_voice,
                        options=dict(options.tts_options if tts_options is None else tts_options),
                    ), tts_provider)
                break
            except TTSServiceTemporarilyUnavailable as exc:
                if temporary_retry >= TIMING_RECOVERY_TEMPORARY_TTS_RETRIES:
                    raise RuntimeError(
                        "Edge TTS remained unavailable during timing recovery "
                        f"after {TIMING_RECOVERY_TEMPORARY_TTS_RETRIES} temporary-outage retries"
                    ) from exc
                wait_for_temporary_tts(exc)
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

    def recovery_batch_size() -> int | None:
        """Return an explicit provider batch size without changing generic behavior."""
        if not callable(getattr(tts_provider, "synthesize_batch", None)):
            return None
        configured_size = getattr(tts_provider, "batch_size", None)
        if isinstance(configured_size, bool):
            return None
        try:
            configured_size = int(configured_size)
        except (TypeError, ValueError, OverflowError):
            return None
        if configured_size < 2:
            return None
        return min(5, max(2, configured_size))

    def replacement_from_result(segment, shortened: str, result: TTSResult):
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
        return replace(
            segment, translated_text=shortened, audio_file=result.audio_file,
            audio_duration=float(result.audio_duration), provider=result.provider,
            voice=result.voice, provider_metadata=dict(result.metadata),
        )

    def apply_replacement(
        item_index: int, shortened: str, replacement, *, semantic_shortening: bool = True,
    ):
        nonlocal current
        segments = list(current.segments)
        segments[item_index] = replacement
        current = replace(current, segments=tuple(segments))
        if semantic_shortening:
            attempts_by_segment[item_index] = attempts_by_segment.get(item_index, 0) + 1
        if (semantic_shortening and checkpoint_path is not None
                and checkpoint_fingerprint is not None):
            _persist_timing_recovery_checkpoint(
                checkpoint_path, checkpoint_fingerprint, current.segments,
                item_index, shortened,
            )
        logger.info("Timing-recovery TTS succeeded for segment %d", item_index)

    def edge_rate_options(
        required_rate: float, *, max_required_rate: float, rate_cap: int,
    ) -> dict[str, object] | None:
        """Return a bounded Edge-only rate increase, if the user left rate at zero."""
        if not isinstance(tts_provider, EdgeTTSProvider):
            return None
        configured_rate = options.tts_options.get("rate")
        if configured_rate is not None and str(configured_rate) not in {"+0%", "0%"}:
            return None
        maximum = options.timing_policy.max_playback_rate
        if required_rate > max_required_rate:
            return None
        # One point beyond the rounded-up required reduction absorbs small TTS
        # duration variance, without changing the synchronization ceiling.
        increase = min(
            rate_cap,
            max(1, math.ceil((required_rate / maximum - 1.0) * 100) + 1),
        )
        fast_options = dict(options.tts_options)
        fast_options["rate"] = f"+{increase}%"
        return fast_options

    def final_edge_rate_options(
        required_rate: float, attempted_rates: set[int],
    ) -> tuple[dict[str, object], int] | None:
        """Return the next bounded Edge rate retry after concise retries are exhausted."""
        initial = edge_rate_options(
            required_rate,
            max_required_rate=options.timing_policy.max_playback_rate * 1.10,
            rate_cap=10,
        )
        if initial is None:
            return None
        initial_rate = int(str(initial["rate"]).strip("+%"))
        increase = initial_rate if not attempted_rates else 10
        if increase in attempted_rates:
            return None
        fast_options = dict(options.tts_options)
        fast_options["rate"] = f"+{increase}%"
        return fast_options, increase

    while True:
        try:
            return synchronize_for_localization(
                current, options.timing_policy, media_end=media_end
            )
        except TimingSynchronizationError as exc:
            failures = overflows() if exc.segment_index is not None else []
            index = exc.segment_index
            # The preflight scan is deliberately approximate: it avoids
            # unnecessary recovery requests, but cannot fully reproduce the
            # synchronizer's prior-segment timing decisions.  In particular,
            # a preceding source gap may appear free here even though the
            # preceding utterance was slowed to occupy it.  The synchronizer's
            # exception is authoritative, so retain its exact failed segment
            # and rate, replacing any approximate duplicate and keeping the
            # recovery batches in timeline order.
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(current.segments)
                and isinstance(exc.required_rate, (int, float))
                and not isinstance(exc.required_rate, bool)
                and math.isfinite(exc.required_rate)
            ):
                failures_by_segment = {item_index: required_rate
                                       for item_index, required_rate in failures}
                failures_by_segment[index] = float(exc.required_rate)
                failures = sorted(failures_by_segment.items())
            shorten = getattr(translation_provider, "shorten_translation", None)
            batch_shorten = getattr(translation_provider, "shorten_translations", None)
            attempts = attempts_by_segment.get(index, 0) if index is not None else 0
            recovery_exhausted = (
                callable(batch_shorten) or callable(shorten)
            ) and any(
                attempts_by_segment.get(item_index, 0) >= options.timing_recovery_attempts
                for item_index, _ in failures
            )
            final_rate_retried = False
            if recovery_exhausted:
                for item_index, required_rate in failures:
                    attempted_rates = final_edge_rate_attempts.setdefault(item_index, set())
                    next_rate = final_edge_rate_options(required_rate, attempted_rates)
                    if next_rate is None:
                        continue
                    fast_options, increase = next_rate
                    attempted_rates.add(increase)
                    segment = current.segments[item_index]
                    logger.info(
                        "Timing-recovery final Edge rate retry starting for segment %d at +%d%%",
                        item_index, increase,
                    )
                    replacement = regenerate(
                        segment, segment.translated_text, tts_options=fast_options,
                        output_suffix=f"-timing-rate-{increase:02d}",
                    )
                    apply_replacement(
                        item_index, segment.translated_text, replacement,
                        semantic_shortening=False,
                    )
                    final_rate_retried = True
            if final_rate_retried:
                # At most two Edge-only retries are allowed per segment; remeasure normally.
                continue
            if (not failures or (not callable(batch_shorten) and not callable(shorten))
                    or recovery_exhausted):
                if recovery_exhausted and options.still_frame_extension.enabled:
                    if options.use_background_separation:
                        raise TimingSynchronizationError(
                            "Still-frame extension cannot be used with background separation; "
                            "background audio would need the identical retiming.",
                            segment_index=index, required_rate=exc.required_rate,
                            max_playback_rate=exc.max_playback_rate,
                        ) from exc
                    try:
                        extension_options = options.still_frame_extension
                        if source_video is not None:
                            extension_options = replace(
                                extension_options, fps=probe_video_fps(source_video),
                            )
                        plan = plan_still_frame_extensions(
                            current, options.timing_policy, media_end=media_end,
                            options=extension_options,
                        )
                        timed = synchronize_for_localization(
                            plan.transcript, options.timing_policy,
                            media_end=(media_end + plan.total_duration
                                       if media_end is not None else None),
                        )
                        return replace(timed, video_extensions=plan.events)
                    except StillFrameExtensionError as plan_error:
                        raise TimingSynchronizationError(
                            f"{exc}; still-frame extension was refused: {plan_error}",
                            segment_index=index, required_rate=exc.required_rate,
                            max_playback_rate=exc.max_playback_rate,
                        ) from plan_error
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

            def timing_requests(batch):
                return [
                    {"id": f"segment-{item_index:06d}",
                     "source_text": current.segments[item_index].source_text,
                     "translated_text": current.segments[item_index].translated_text,
                     "audio_duration": current.segments[item_index].audio_duration,
                     "available_duration": (
                         float(current.segments[item_index + 1].start)
                         if item_index + 1 < len(current.segments)
                         else media_end if media_end is not None
                         else float(current.segments[item_index].end)
                     ) - float(current.segments[item_index].start),
                     "required_rate": required_rate,
                     "max_playback_rate": options.timing_policy.max_playback_rate}
                    for item_index, required_rate in batch
                ]

            def unusable_structured_output(error: LocalizationProviderError) -> bool:
                return any(marker in str(error).lower() for marker in (
                    "unusable structured output", "malformed structured output",
                ))

            def transient_transport_error(error: LocalizationProviderError) -> bool:
                message = str(error).upper()
                return (bool(re.search(r"(?<!\d)(?:500|502|503|504)(?!\d)", message))
                        or any(marker in message for marker in (
                            "DEADLINE_EXCEEDED", "GATEWAY TIMEOUT", "TIMED OUT",
                            "CONNECTION RESET", "SERVER DISCONNECTED",
                            "REMOTE PROTOCOL ERROR", "SERVICE UNAVAILABLE",
                        )))

            def shorten_timing_batch(batch):
                """Shorten one batch, isolating persistent transient failures too."""
                requests = timing_requests(batch)
                def request_once():
                    with options.resource_scheduler.acquire(
                        "gemini_timing", cancel_event=cancel_event,
                        waiting=(lambda: waiting_callback("Gemini")) if waiting_callback else None,
                    ):
                        return batch_shorten(
                            segments=requests, source_language=current.source_language,
                            target_language=current.target_language,
                        )

                try:
                    shortened_items = _retry_gemini_timing_transient_failure(
                        request_once, cancel_event=cancel_event,
                        waiting_callback=waiting_callback,
                        max_retries=1,
                    )
                except LocalizationProviderError as provider_error:
                    if len(batch) > 1 and (unusable_structured_output(provider_error)
                                           or transient_transport_error(provider_error)):
                        if waiting_callback is not None:
                            waiting_callback(
                                f"Gemini timing recovery is splitting {len(batch)} slow segments"
                            )
                        midpoint = len(batch) // 2
                        return shorten_timing_batch(batch[:midpoint]) + shorten_timing_batch(batch[midpoint:])
                    raise
                if (not isinstance(shortened_items, (list, tuple))
                        or len(shortened_items) != len(batch)
                        or any(not isinstance(item, str) or not item.strip()
                               for item in shortened_items)):
                    if len(batch) > 1:
                        midpoint = len(batch) // 2
                        return shorten_timing_batch(batch[:midpoint]) + shorten_timing_batch(batch[midpoint:])
                    raise ValueError("Translation provider returned an invalid shortened batch")
                return list(shortened_items)

            daily_quota_exhausted = False
            for start in range(0, len(failures), TIMING_RECOVERY_BATCH_SIZE):
                batch = failures[start:start + TIMING_RECOVERY_BATCH_SIZE]
                if callable(batch_shorten):
                    try:
                        shortened_items = shorten_timing_batch(batch)
                    except LocalizationProviderError as provider_error:
                        if "daily quota is exhausted" not in str(provider_error).lower():
                            raise
                        daily_quota_exhausted = True
                        logger.warning(
                            "Gemini daily quota exhausted during timing recovery; "
                            "using bounded Edge rate fallback"
                        )
                        break
                    if not isinstance(shortened_items, (list, tuple)) or len(shortened_items) != len(batch):
                        raise ValueError("Translation provider returned an invalid shortened batch")
                else:
                    def request_singletons():
                        with options.resource_scheduler.acquire(
                            "gemini_timing", cancel_event=cancel_event,
                            waiting=(lambda: waiting_callback("Gemini")) if waiting_callback else None,
                        ):
                            return [shorten(
                                source_text=current.segments[item_index].source_text,
                                translated_text=current.segments[item_index].translated_text,
                                source_language=current.source_language,
                                target_language=current.target_language,
                                required_rate=required_rate,
                                max_playback_rate=options.timing_policy.max_playback_rate,
                            ) for item_index, required_rate in batch]

                    try:
                        shortened_items = _retry_gemini_timing_transient_failure(
                            request_singletons, cancel_event=cancel_event,
                            waiting_callback=waiting_callback,
                        )
                    except LocalizationProviderError as provider_error:
                        if "daily quota is exhausted" not in str(provider_error).lower():
                            raise
                        daily_quota_exhausted = True
                        logger.warning(
                            "Gemini daily quota exhausted during timing recovery; "
                            "using bounded Edge rate fallback"
                        )
                        break
                jobs = []
                for (item_index, _), shortened in zip(batch, shortened_items):
                    if not isinstance(shortened, str) or not shortened.strip():
                        raise ValueError("Translation provider returned an invalid shortened segment")
                    jobs.append((item_index, current.segments[item_index], shortened.strip()))

                if waiting_callback is not None:
                    waiting_callback(
                        f"Timing recovery: Gemini processed {min(start + len(batch), len(failures))}/{len(failures)} overflowing segments"
                    )

                batch_size = recovery_batch_size()
                if batch_size is None:
                    # Do not queue later Edge requests: failure must stop this batch now.
                    for item_index, segment, shortened in jobs:
                        logger.info("Timing-recovery TTS starting for segment %d", item_index)
                        try:
                            replacement = regenerate(segment, shortened)
                        except Exception:
                            logger.info("Timing-recovery TTS failed for segment %d", item_index, exc_info=True)
                            raise
                        apply_replacement(item_index, shortened, replacement)
                else:
                    for first in range(0, len(jobs), batch_size):
                        tts_jobs = jobs[first:first + batch_size]
                        for item_index, _, _ in tts_jobs:
                            logger.info("Timing-recovery TTS starting for segment %d", item_index)
                        try:
                            for temporary_retry in range(TIMING_RECOVERY_TEMPORARY_TTS_RETRIES + 1):
                                try:
                                    with options.resource_scheduler.acquire(
                                        "tts", cancel_event=cancel_event,
                                        waiting=(lambda: waiting_callback("TTS")) if waiting_callback else None,
                                    ):
                                        results = _synthesize_batch_with_checkpoints(
                                            [TTSRequest(
                                                text=shortened,
                                                language=synthesized.target_language,
                                                output_file_stem=str(Path(segment.audio_file).with_suffix("")),
                                                voice=options.tts_voice,
                                                options=dict(options.tts_options),
                                            ) for _, segment, shortened in tts_jobs], tts_provider,
                                        )
                                    break
                                except TTSBatchFallbackRequired:
                                    # Batch capacity is already released. Retry each
                                    # recovery segment through regenerate(), which
                                    # reacquires TTS capacity independently.
                                    for item_index, segment, shortened in tts_jobs:
                                        replacement = regenerate(segment, shortened)
                                        apply_replacement(item_index, shortened, replacement)
                                    results = None
                                    break
                                except TTSServiceTemporarilyUnavailable as unavailable:
                                    if temporary_retry >= TIMING_RECOVERY_TEMPORARY_TTS_RETRIES:
                                        raise RuntimeError(
                                            "Edge TTS remained unavailable during timing recovery "
                                            f"after {TIMING_RECOVERY_TEMPORARY_TTS_RETRIES} temporary-outage retries"
                                        ) from unavailable
                                    wait_for_temporary_tts(unavailable)
                            if results is None:
                                continue
                            if len(results) != len(tts_jobs):
                                raise ValueError("TTS batch provider returned an invalid response count")
                        except Exception:
                            logger.info(
                                "Timing-recovery TTS batch failed for segments %s",
                                [item_index for item_index, _, _ in tts_jobs], exc_info=True,
                            )
                            raise
                        for (item_index, segment, shortened), result in zip(tts_jobs, results):
                            apply_replacement(
                                item_index, shortened,
                                replacement_from_result(segment, shortened, result),
                            )

            if daily_quota_exhausted:
                for item_index, _ in failures:
                    attempts_by_segment[item_index] = options.timing_recovery_attempts
                continue


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
    if cancel_event is not None and asr_provider in {"paraformer", "paraformer-zh"}:
        transcribe_kwargs["cancel_event"] = cancel_event
    with gated("asr", LocalizationStage.TRANSCRIBE, "Nhận dạng"):
        try:
            transcript = transcribe_func(str(source), transcription_options, model_path, **transcribe_kwargs)
        except LocalizationTranscriptionCancelled as exc:
            raise LocalizationCancelled(str(exc)) from exc
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

    def retry_temporary_tts(exc: TTSServiceTemporarilyUnavailable):
        delay = exc.retry_after_seconds
        if progress_callback is not None:
            progress_callback(LocalizationProgress(
                LocalizationStage.TTS, done, steps,
                f"Edge TTS tạm thời không khả dụng, tự thử lại sau {math.ceil(delay)} giây",
            ))
        if cancel_event is not None and cancel_event.wait(delay):
            raise LocalizationCancelled("Localization was cancelled")
        if cancel_event is None:
            threading.Event().wait(delay)

    try:
        synthesized = synthesize_for_localization(
            translated,
            tts_provider,
            tts_dir,
            voice=options.tts_voice,
            options=dict(options.tts_options),
            segment_context=tts_segment_context,
            progress_callback=report_tts_segment,
            temporary_unavailable_callback=retry_temporary_tts,
        )
    except ResourceCancelled as exc:
        raise LocalizationCancelled(str(exc)) from exc
    report(LocalizationStage.TTS, "Vietnamese TTS completed")

    try:
        def timing_status(message: str):
            if progress_callback is not None:
                progress_callback(LocalizationProgress(
                    LocalizationStage.TIMING, done, steps, message,
                ))

        timed = _synchronize_with_translation_recovery(
            synthesized, translation_provider, tts_provider, options,
            media_end=source_duration,
            cancel_event=cancel_event,
            waiting_callback=timing_status,
            checkpoint_path=recovery_checkpoint,
            checkpoint_fingerprint=recovery_fingerprint,
            source_video=source,
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

        # The retimed silent video is the sole video input for final render.
        # It is created only when recovery explicitly planned frozen frames.
        render_source = source
        extended_duration = source_duration
        if timed.video_extensions:
            render_source = render_still_frame_video(
                source, workspace / "retimed_source_video.mp4", timed.video_extensions,
            )
            extended_duration += sum(event.duration for event in timed.video_extensions)
        localized_audio = workspace / "localized_audio.wav"
        mix_localized_audio(timed, localized_audio, background_audio_file=background_audio,
                            options=options.audio_options, target_duration=extended_duration)
        report(LocalizationStage.MIX_AUDIO, "Localized audio mixed")
        subtitle_file = output_dir / f"{source.stem}.vi.{options.subtitle_format}"
        subtitle_result = write_subtitles(timed, subtitle_file, format=options.subtitle_format,
                                          options=options.subtitle_options)
        report(LocalizationStage.SUBTITLES, "Vietnamese subtitles created")
        final_file = output_dir / f"{source.stem}.vi.mp4"
        final_result = render_localized_mp4(
            render_source, localized_audio, final_file,
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
