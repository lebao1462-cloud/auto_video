from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import shutil
import threading
from typing import Callable, Mapping

from buzz.localization.audio_mix import (
    AudioMixingOptions,
    extract_audio_track,
    mix_localized_audio,
    probe_media_duration,
    separate_background_with_demucs,
)
from buzz.localization.final_render import FinalRenderOptions, FinalRenderResult, render_localized_mp4
from buzz.localization.subtitles import SubtitleOptions, SubtitleResult, write_subtitles
from buzz.localization.timing import (
    TimingPolicy,
    TimingSynchronizationError,
    _MAX_PLAYBACK_RATE_RELATIVE_TOLERANCE,
    synchronize_for_localization,
)
from buzz.localization.transcript import LocalizationTranscript, transcribe_for_localization
from buzz.localization.translation import TranslationProvider, translate_for_localization
from buzz.localization.tts import (
    SynthesizedLocalizationTranscript,
    TTSProvider,
    TTSRequest,
    TTSResult,
    synthesize_segment,
    synthesize_for_localization,
)
from buzz.transcriber.transcriber import TranscriptionOptions


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
    completed_steps: int
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


def _synchronize_with_translation_recovery(
    synthesized: SynthesizedLocalizationTranscript,
    translation_provider: TranslationProvider,
    tts_provider: TTSProvider,
    options: LocalizationWorkflowOptions,
):
    current = synthesized
    attempts_by_segment: dict[int, int] = {}

    def overflows():
        """Find every rate overflow using the synchronizer's natural-gap rules."""
        failures = []
        for index, segment in enumerate(current.segments):
            next_start = (
                float(current.segments[index + 1].start)
                if index + 1 < len(current.segments) else float(segment.end)
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

    # Respect provider concurrency while keeping undeclared/serial providers safe.
    # EdgeTTSProvider declares max_concurrency=1 because parallel requests can
    # trigger repeated NoAudioReceived/WebSocket failures.
    declared_concurrency = getattr(tts_provider, "max_concurrency", None)
    if (
        isinstance(declared_concurrency, bool)
        or not isinstance(declared_concurrency, int)
        or declared_concurrency < 1
    ):
        recovery_tts_workers = 1
    else:
        recovery_tts_workers = min(declared_concurrency, 5)

    while True:
        try:
            return synchronize_for_localization(current, options.timing_policy)
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
                    shortened_items = batch_shorten(
                        segments=requests, source_language=current.source_language,
                        target_language=current.target_language,
                    )
                    if not isinstance(shortened_items, (list, tuple)) or len(shortened_items) != len(batch):
                        raise ValueError("Translation provider returned an invalid shortened batch")
                else:
                    shortened_items = [shorten(
                        source_text=current.segments[item_index].source_text,
                        translated_text=current.segments[item_index].translated_text,
                        source_language=current.source_language,
                        target_language=current.target_language,
                        required_rate=required_rate,
                        max_playback_rate=options.timing_policy.max_playback_rate,
                    ) for item_index, required_rate in batch]
                replacements = []
                jobs = []
                for (item_index, _), shortened in zip(batch, shortened_items):
                    if not isinstance(shortened, str) or not shortened.strip():
                        raise ValueError("Translation provider returned an invalid shortened segment")
                    jobs.append((item_index, current.segments[item_index], shortened.strip()))

                with ThreadPoolExecutor(max_workers=recovery_tts_workers) as executor:
                    futures = [
                        (item_index, executor.submit(regenerate, segment, shortened))
                        for item_index, segment, shortened in jobs
                    ]
                    replacements = [
                        (item_index, future.result())
                        for item_index, future in futures
                    ]

                segments = list(current.segments)
                for item_index, replacement in replacements:
                    segments[item_index] = replacement
                    attempts_by_segment[item_index] = attempts_by_segment.get(item_index, 0) + 1
                current = replace(current, segments=tuple(segments))


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
) -> LocalizationWorkflowResult:
    source = Path(source_video_file)
    if not source.is_file():
        raise ValueError(f"Input media file does not exist: {source_video_file}")

    output_dir = Path(options.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / f"{source.stem}_localization_work"
    workspace.mkdir(parents=True, exist_ok=True)

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

    check_cancelled()
    transcribe_kwargs = {"asr_provider": asr_provider} if asr_provider != "whisper" else {}
    transcript = transcribe_func(str(source), transcription_options, model_path, **transcribe_kwargs)
    report(LocalizationStage.TRANSCRIBE, "Source transcription completed")

    translated = translate_for_localization(
        transcript,
        translation_provider,
        checkpoint_path=workspace / "translation-checkpoint.json",
    )
    report(LocalizationStage.TRANSLATE, "Vietnamese translation completed")

    tts_dir = workspace / "tts"
    synthesized = synthesize_for_localization(
        translated,
        tts_provider,
        tts_dir,
        voice=options.tts_voice,
        options=dict(options.tts_options),
    )
    report(LocalizationStage.TTS, "Vietnamese TTS completed")

    timed = _synchronize_with_translation_recovery(
        synthesized, translation_provider, tts_provider, options
    )
    report(LocalizationStage.TIMING, "Speech timing plan completed")

    source_duration = probe_media_duration(source)

    extracted_audio = workspace / "source_audio.wav"
    extract_audio_track(
        source,
        extracted_audio,
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
    mix_localized_audio(
        timed,
        localized_audio,
        background_audio_file=background_audio,
        options=options.audio_options,
        target_duration=source_duration,
    )
    report(LocalizationStage.MIX_AUDIO, "Localized audio mixed")

    subtitle_file = output_dir / f"{source.stem}.vi.{options.subtitle_format}"
    subtitle_result = write_subtitles(
        timed,
        subtitle_file,
        format=options.subtitle_format,
        options=options.subtitle_options,
    )
    report(LocalizationStage.SUBTITLES, "Vietnamese subtitles created")

    final_file = output_dir / f"{source.stem}.vi.mp4"
    final_result = render_localized_mp4(
        source,
        localized_audio,
        final_file,
        subtitle_file=subtitle_file
        if options.render_options.subtitle_mode != "none"
        else None,
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
    if options.chunk_duration_seconds:
        duration = probe_media_duration(source)
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
    )


def cleanup_localization_workspace(result: LocalizationWorkflowResult) -> None:
    workspace = Path(result.workspace)
    if workspace.is_dir():
        shutil.rmtree(workspace)
