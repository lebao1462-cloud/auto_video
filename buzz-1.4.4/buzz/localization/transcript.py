from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from buzz.transcriber.file_transcriber import FileTranscriber
from buzz.transcriber.transcriber import (
    FileTranscriptionOptions,
    FileTranscriptionTask,
    Segment,
    Task,
    TranscriptionOptions,
)

SUPPORTED_SOURCE_LANGUAGES = {"en", "zh"}
SOURCE_LANGUAGE_ALIASES = {
    "chinese": "zh",
    "english": "en",
    "zh-cn": "zh",
    "zh-tw": "zh",
}


@dataclass(frozen=True)
class LocalizationSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class LocalizationTranscript:
    source_language: str
    source_file: str
    segments: tuple[LocalizationSegment, ...]

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "source_language": self.source_language,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def localization_transcript_from_segments(
    source_file: str,
    source_language: str,
    segments: Iterable[Segment],
) -> LocalizationTranscript:
    if not source_file:
        raise ValueError("Source file is required")
    if isinstance(source_language, str):
        normalized_language = source_language.lower()
        source_language = SOURCE_LANGUAGE_ALIASES.get(
            normalized_language, normalized_language
        )
    if source_language not in SUPPORTED_SOURCE_LANGUAGES:
        raise ValueError("Phase 1 source language must be 'en' or 'zh'")

    return LocalizationTranscript(
        source_file=source_file,
        source_language=source_language,
        segments=tuple(
            LocalizationSegment(
                start=segment.start / 1000,
                end=segment.end / 1000,
                text=segment.text,
            )
            for segment in segments
        ),
    )


def transcribe_for_localization(
    video_path: str,
    transcription_options: TranscriptionOptions,
    model_path: str,
    asr_provider: str = "whisper",
    transcriber_factory: (
        Callable[[FileTranscriptionTask], FileTranscriber] | None
    ) = None,
) -> LocalizationTranscript:
    """Transcribe media through Buzz and return a Phase 1 normalized transcript."""
    source_path = Path(video_path)
    if not source_path.is_file():
        raise ValueError(f"Input media file does not exist: {video_path}")
    if transcription_options.task != Task.TRANSCRIBE:
        raise ValueError("Phase 1 requires the transcribe task")
    if (
        transcription_options.language is not None
        and transcription_options.language not in SUPPORTED_SOURCE_LANGUAGES
    ):
        raise ValueError(
            "Phase 1 requires an English ('en') or Chinese ('zh') source language"
        )
    # ``paraformer`` was used by an unreleased UI build. Accept it at the API
    # boundary so saved drafts do not break, while persistently using the
    # explicit engine identifier below.
    if asr_provider == "paraformer":
        asr_provider = "paraformer-zh"
    if asr_provider not in {"whisper", "paraformer-zh"}:
        raise ValueError("Localization ASR provider must be 'whisper' or 'paraformer-zh'")
    if asr_provider == "paraformer-zh":
        if transcription_options.language == "en":
            raise ValueError("Paraformer-zh is Chinese optimized. Choose Whisper for explicit English transcription.")
        from buzz.localization.paraformer import transcribe_with_paraformer
        return transcribe_with_paraformer(str(source_path))

    task = FileTranscriptionTask(
        file_path=str(source_path),
        transcription_options=transcription_options,
        file_transcription_options=FileTranscriptionOptions(),
        model_path=model_path,
    )

    if transcriber_factory is None:
        from buzz.transcriber.file_transcriber_factory import create_file_transcriber

        transcriber_factory = create_file_transcriber

    transcriber = transcriber_factory(task)
    segments = transcriber.transcribe()
    source_language = (
        getattr(transcriber, "detected_language", None)
        or transcription_options.language
    )
    if not source_language:
        # Some backends omit detection on quiet clips. Use the recognized
        # writing system only when no language code was returned.
        text = " ".join(segment.text or "" for segment in segments)
        has_chinese = sum("\u4e00" <= char <= "\u9fff" for char in text) >= 2
        has_latin = sum(char.isascii() and char.isalpha() for char in text) >= 3
        if has_chinese:
            source_language = "zh"
        elif has_latin or not segments:
            source_language = "en"

    return localization_transcript_from_segments(
        source_file=str(source_path),
        source_language=source_language,
        segments=segments,
    )
