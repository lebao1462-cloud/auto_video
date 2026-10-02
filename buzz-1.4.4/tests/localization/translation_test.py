import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from buzz.localization.transcript import LocalizationSegment, LocalizationTranscript
from buzz.localization.translation import (
    TranslatedLocalizationSegment,
    translate_for_localization,
)


class FakeTranslationProvider:
    def __init__(self, translations):
        self.translations = translations
        self.calls = []

    def translate(self, text, source_language, target_language):
        self.calls.append((text, source_language, target_language))
        return self.translations[text]


def make_transcript(source_language="en", segments=None):
    return LocalizationTranscript(
        source_file="input.mp4",
        source_language=source_language,
        segments=tuple(
            segments
            if segments is not None
            else [LocalizationSegment(0.0, 3.2, "Hello everyone")]
        ),
    )


def test_translates_english_to_vietnamese_and_preserves_source_data():
    transcript = make_transcript()
    vietnamese = "Xin ch\u00e0o m\u1ecdi ng\u01b0\u1eddi"
    provider = FakeTranslationProvider({"Hello everyone": vietnamese})

    result = translate_for_localization(transcript, provider)

    assert result.source_file == "input.mp4"
    assert result.source_language == "en"
    assert result.target_language == "vi"
    assert result.segments == (
        TranslatedLocalizationSegment(
            start=0.0,
            end=3.2,
            source_text="Hello everyone",
            translated_text=vietnamese,
        ),
    )
    assert provider.calls == [("Hello everyone", "en", "vi")]


def test_translates_chinese_to_vietnamese_without_losing_unicode():
    chinese = "\u5927\u5bb6\u597d"
    vietnamese = "Xin ch\u00e0o m\u1ecdi ng\u01b0\u1eddi"
    transcript = make_transcript(
        source_language="zh",
        segments=[LocalizationSegment(1.25, 2.75, chinese)],
    )
    provider = FakeTranslationProvider({chinese: vietnamese})

    result = translate_for_localization(transcript, provider)

    assert result.segments[0].source_text == chinese
    assert result.segments[0].translated_text == vietnamese
    assert provider.calls == [(chinese, "zh", "vi")]


def test_preserves_segment_timestamps_and_order():
    transcript = make_transcript(
        segments=[
            LocalizationSegment(5.678901, 8.123456, "Second in time"),
            LocalizationSegment(0.123456, 4.987654, "First in time"),
        ]
    )
    provider = FakeTranslationProvider(
        {"Second in time": "Th\u1ee9 hai", "First in time": "Th\u1ee9 nh\u1ea5t"}
    )

    result = translate_for_localization(transcript, provider)

    assert [(segment.start, segment.end) for segment in result.segments] == [
        (5.678901, 8.123456),
        (0.123456, 4.987654),
    ]
    assert [segment.source_text for segment in result.segments] == [
        "Second in time",
        "First in time",
    ]


def test_output_is_json_compatible():
    vietnamese = "Xin ch\u00e0o m\u1ecdi ng\u01b0\u1eddi"
    result = translate_for_localization(
        make_transcript(),
        FakeTranslationProvider({"Hello everyone": vietnamese}),
    )
    encoded = json.dumps(result.to_dict(), ensure_ascii=False)

    assert json.loads(encoded) == result.to_dict()
    assert vietnamese in encoded


def test_rejects_empty_transcript_without_calling_provider():
    provider = FakeTranslationProvider({})

    with pytest.raises(ValueError, match="empty transcript"):
        translate_for_localization(make_transcript(segments=[]), provider)

    assert provider.calls == []


@pytest.mark.parametrize("text", ["", "   ", None])
def test_rejects_empty_source_segment_before_translation(text):
    transcript = make_transcript(segments=[LocalizationSegment(0, 1, text)])
    provider = FakeTranslationProvider({})

    with pytest.raises(ValueError, match="empty source segment"):
        translate_for_localization(transcript, provider)

    assert provider.calls == []


@pytest.mark.parametrize("source_language", ["", "vi", "English", None])
def test_rejects_unsupported_source_language(source_language):
    with pytest.raises(ValueError, match="source language"):
        translate_for_localization(
            make_transcript(source_language=source_language),
            FakeTranslationProvider({"Hello everyone": "Xin ch\u00e0o"}),
        )


@pytest.mark.parametrize("target_language", ["en", "zh", "VI", ""])
def test_rejects_unsupported_target_language(target_language):
    with pytest.raises(ValueError, match="target language"):
        translate_for_localization(
            make_transcript(),
            FakeTranslationProvider({"Hello everyone": "Xin ch\u00e0o"}),
            target_language=target_language,
        )


def test_propagates_provider_exception():
    class FailingProvider:
        def translate(self, text, source_language, target_language):
            raise TimeoutError("provider timed out")

    with pytest.raises(TimeoutError, match="provider timed out"):
        translate_for_localization(make_transcript(), FailingProvider())


@pytest.mark.parametrize("response", ["", "   ", None, {"text": "Xin ch\u00e0o"}])
def test_rejects_malformed_provider_response(response):
    provider = FakeTranslationProvider({"Hello everyone": response})

    with pytest.raises(ValueError, match="invalid response"):
        translate_for_localization(make_transcript(), provider)


def test_does_not_mutate_input_transcript():
    transcript = make_transcript()
    original = transcript.to_dict()

    translate_for_localization(
        transcript,
        FakeTranslationProvider(
            {"Hello everyone": "Xin ch\u00e0o m\u1ecdi ng\u01b0\u1eddi"}
        ),
    )

    assert transcript.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        transcript.source_language = "zh"


def test_partial_provider_failure_raises_without_returning_partial_data():
    transcript = make_transcript(
        segments=[
            LocalizationSegment(0, 1, "First"),
            LocalizationSegment(1, 2, "Second"),
        ]
    )

    class PartiallyFailingProvider:
        def translate(self, text, source_language, target_language):
            if text == "Second":
                raise ConnectionError("provider unavailable")
            return "\u0110\u1ea7u ti\u00ean"

    with pytest.raises(ConnectionError, match="provider unavailable"):
        translate_for_localization(transcript, PartiallyFailingProvider())

    assert transcript.segments == (
        LocalizationSegment(0, 1, "First"),
        LocalizationSegment(1, 2, "Second"),
    )


def test_batch_translation_keeps_timestamps_and_corrected_chinese():
    class BatchProvider:
        def __init__(self):
            self.calls = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            self.calls.append((len(segments), len(context)))
            return [(f"修正{i}", f"Dịch {i}") for i, _ in enumerate(segments)]

    provider = BatchProvider()
    transcript = make_transcript(
        source_language="zh",
        segments=[LocalizationSegment(float(i), float(i + 1), f"文本{i}") for i in range(31)],
    )
    result = translate_for_localization(transcript, provider)
    assert provider.calls == [(30, 0), (1, 5)]
    assert result.segments[30].start == 30
    assert result.segments[30].end == 31
    assert result.segments[30].source_text == "修正0"
    assert result.segments[30].translated_text == "Dịch 0"


def test_batch_wrong_count_retries_then_recursively_splits():
    class SplittingProvider:
        model = "test-model"

        def __init__(self):
            self.calls = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            self.calls.append(tuple(item.segment_id for item in segments))
            if len(segments) > 1:
                return []
            return [(segments[0].text, f"Dịch {segments[0].text}")]

    provider = SplittingProvider()
    transcript = make_transcript(segments=[
        LocalizationSegment(0, 1, "one"),
        LocalizationSegment(1, 2, "two"),
    ])

    result = translate_for_localization(transcript, provider)

    assert len(provider.calls) == 4
    assert provider.calls[0] == provider.calls[1]
    assert provider.calls[2][0] == provider.calls[0][0]
    assert provider.calls[3][0] == provider.calls[0][1]
    assert [item.translated_text for item in result.segments] == [
        "Dịch one", "Dịch two"
    ]


def test_translation_checkpoint_resumes_by_source_provider_and_model():
    class CheckpointProvider:
        model = "model-a"

        def __init__(self):
            self.calls = 0

        def translate_segments(self, segments, source_language, target_language, *, context):
            self.calls += 1
            return [(item.text, f"Dịch {item.text}") for item in segments]

    checkpoint = Path("translation-checkpoint-test.json").resolve()
    checkpoint.unlink(missing_ok=True)
    transcript = make_transcript(segments=[LocalizationSegment(0, 1, "one")])
    try:
        first = CheckpointProvider()
        translate_for_localization(transcript, first, checkpoint_path=checkpoint)
        resumed = CheckpointProvider()

        result = translate_for_localization(
            transcript, resumed, checkpoint_path=checkpoint
        )

        assert resumed.calls == 0
        assert result.segments[0].translated_text == "Dịch one"

        changed_model = CheckpointProvider()
        changed_model.model = "model-b"
        translate_for_localization(
            transcript, changed_model, checkpoint_path=checkpoint
        )
        assert changed_model.calls == 1
    finally:
        checkpoint.unlink(missing_ok=True)
