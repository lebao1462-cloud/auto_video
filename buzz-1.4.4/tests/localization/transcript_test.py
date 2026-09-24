from unittest.mock import Mock

import pytest

from buzz.localization.transcript import (
    LocalizationSegment,
    localization_transcript_from_segments,
    transcribe_for_localization,
)
from buzz.transcriber.transcriber import Segment, Task, TranscriptionOptions


def test_converts_buzz_segments_and_preserves_timestamp_precision():
    transcript = localization_transcript_from_segments(
        source_file="input.mp4",
        source_language="en",
        segments=[Segment(start=1234, end=5678, text="Hello everyone")],
    )

    assert transcript.segments == (
        LocalizationSegment(start=1.234, end=5.678, text="Hello everyone"),
    )
    assert transcript.to_dict() == {
        "source_file": "input.mp4",
        "source_language": "en",
        "segments": [
            {"start": 1.234, "end": 5.678, "text": "Hello everyone"}
        ],
    }


def test_preserves_chinese_source_language_and_text():
    transcript = localization_transcript_from_segments(
        source_file="input.mp4",
        source_language="zh",
        segments=[Segment(start=0, end=2400, text="大家好")],
    )

    assert transcript.source_language == "zh"
    assert transcript.segments[0].text == "大家好"


def test_handles_empty_transcription_results():
    transcript = localization_transcript_from_segments(
        source_file="input.mp4",
        source_language="en",
        segments=[],
    )

    assert transcript.segments == ()
    assert transcript.to_dict()["segments"] == []


@pytest.mark.parametrize("source_language", ["", "vi", None])
def test_rejects_unsupported_or_missing_source_language(source_language):
    with pytest.raises(ValueError, match="source language"):
        localization_transcript_from_segments(
            source_file="input.mp4",
            source_language=source_language,
            segments=[],
        )


def test_transcribes_with_existing_buzz_task_and_backend(tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"test media placeholder")
    transcriber = Mock(detected_language="en")
    transcriber.transcribe.return_value = [Segment(0, 4100, "Welcome")]
    factory = Mock(return_value=transcriber)
    options = TranscriptionOptions(language="en")

    transcript = transcribe_for_localization(
        video_path=str(video_path),
        transcription_options=options,
        model_path="model.bin",
        transcriber_factory=factory,
    )

    task = factory.call_args.args[0]
    assert task.file_path == str(video_path)
    assert task.transcription_options is options
    assert task.file_transcription_options.output_formats == set()
    transcriber.transcribe.assert_called_once_with()
    assert transcript.segments[0].end == 4.1


def test_preserves_backend_detected_language(tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"test media placeholder")
    transcriber = Mock(detected_language="zh")
    transcriber.transcribe.return_value = [Segment(0, 2400, "大家好")]

    transcript = transcribe_for_localization(
        video_path=str(video_path),
        transcription_options=TranscriptionOptions(language=None),
        model_path="model.bin",
        transcriber_factory=Mock(return_value=transcriber),
    )

    assert transcript.source_language == "zh"


def test_normalizes_backend_language_name():
    transcript = localization_transcript_from_segments(
        source_file="input.mp4",
        source_language="English",
        segments=[],
    )

    assert transcript.source_language == "en"


def test_rejects_missing_input_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        transcribe_for_localization(
            video_path=str(tmp_path / "missing.mp4"),
            transcription_options=TranscriptionOptions(language="en"),
            model_path="model.bin",
            transcriber_factory=Mock(),
        )


def test_rejects_translation_task(tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"test media placeholder")

    with pytest.raises(ValueError, match="transcribe task"):
        transcribe_for_localization(
            video_path=str(video_path),
            transcription_options=TranscriptionOptions(
                language="en", task=Task.TRANSLATE
            ),
            model_path="model.bin",
            transcriber_factory=Mock(),
        )


def test_propagates_transcriber_errors(tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"test media placeholder")
    transcriber = Mock()
    transcriber.transcribe.side_effect = RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="backend failed"):
        transcribe_for_localization(
            video_path=str(video_path),
            transcription_options=TranscriptionOptions(language="zh"),
            model_path="model.bin",
            transcriber_factory=Mock(return_value=transcriber),
        )
