import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from buzz.localization.transcript import LocalizationSegment, LocalizationTranscript
from buzz.localization.translation import (
    TranslatedLocalizationSegment,
    _checkpoint_fingerprint,
    _valid_reviewed_text,
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


@pytest.mark.parametrize(
    "response", ["", "   ", "...", None, {"text": "Xin ch\u00e0o"}]
)
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
    assert provider.calls == [(30, 0), (1, 12)]
    assert result.segments[30].start == 30
    assert result.segments[30].end == 31
    assert result.segments[30].source_text == "修正0"
    assert result.segments[30].translated_text == "Dịch 0"


def test_batch_translation_passes_completed_vietnamese_context_across_boundary():
    class BatchProvider:
        def __init__(self):
            self.contexts = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            self.contexts.append(tuple(context))
            return [(f"corrected {item.text}", f"Vietnamese {item.text}")
                    for item in segments]

    provider = BatchProvider()
    transcript = make_transcript(segments=[
        LocalizationSegment(float(index), float(index + 1), f"line {index}")
        for index in range(31)
    ])

    translate_for_localization(transcript, provider)

    context = provider.contexts[1]
    assert len(context) == 12
    assert [item.text for item in context] == [
        f"corrected line {index}" for index in range(18, 30)
    ]
    assert [item.translated_text for item in context] == [
        f"Vietnamese line {index}" for index in range(18, 30)
    ]


def test_translation_checkpoint_fingerprint_uses_policy_not_secret_or_fallback_order():
    class Provider:
        model = "model-a"
        api_key = "first-secret"
        rate_limit_fallback_models = ("model-b", "model-c")
        translation_policy_version = "policy-one"

    transcript = make_transcript()
    provider = Provider()
    baseline = _checkpoint_fingerprint(transcript, provider, "vi")
    provider.api_key = "second-secret"
    provider.rate_limit_fallback_models = ("model-c", "model-b")
    assert _checkpoint_fingerprint(transcript, provider, "vi") == baseline
    provider.translation_policy_version = "policy-two"
    assert _checkpoint_fingerprint(transcript, provider, "vi") != baseline


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


def test_checkpoint_retranslates_only_punctuation_entry_and_rewrites_it():
    class CheckpointProvider:
        def __init__(self):
            self.calls = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            self.calls.append([item.text for item in segments])
            return [(item.text, f"Dich {item.text}") for item in segments]

    transcript = make_transcript(segments=[
        LocalizationSegment(0, 1, "good"),
        LocalizationSegment(1, 2, "retry"),
    ])
    checkpoint = Path("translation-checkpoint-invalid-entry-test.json").resolve()
    checkpoint.unlink(missing_ok=True)
    try:
        translate_for_localization(transcript, CheckpointProvider(), checkpoint_path=checkpoint)
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        retry_id = next(key for key, value in saved["segments"].items()
                        if value["source_text"] == "retry")
        saved["segments"][retry_id]["translated_text"] = "..."
        checkpoint.write_text(json.dumps(saved), encoding="utf-8")

        resumed = CheckpointProvider()
        result = translate_for_localization(transcript, resumed, checkpoint_path=checkpoint)

        assert resumed.calls == [["retry"]]
        assert [item.translated_text for item in result.segments] == ["Dich good", "Dich retry"]
        rewritten = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert rewritten["segments"][retry_id]["translated_text"] == "Dich retry"
    finally:
        checkpoint.unlink(missing_ok=True)


def test_batch_rejects_punctuation_only_response():
    class BatchProvider:
        def translate_segments(self, segments, source_language, target_language, *, context):
            return [(item.text, "...") for item in segments]

    with pytest.raises(ValueError, match="invalid response"):
        translate_for_localization(make_transcript(), BatchProvider())


def test_singleton_batch_uses_plain_translation_after_invalid_structured_attempts():
    class BatchProvider:
        def __init__(self):
            self.batch_calls = 0
            self.translate_calls = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            self.batch_calls += 1
            return [(item.text, "...") for item in segments]

        def translate(self, text, source_language, target_language):
            self.translate_calls.append((text, source_language, target_language))
            return "Ban dich du phong"

    provider = BatchProvider()
    result = translate_for_localization(make_transcript(), provider)

    assert provider.batch_calls == 2
    assert provider.translate_calls == [("Hello everyone", "en", "vi")]
    assert result.segments[0].source_text == "Hello everyone"
    assert result.segments[0].translated_text == "Ban dich du phong"


def test_register_review_runs_after_raw_translation_and_uses_reviewed_batches(tmp_path):
    class Provider:
        model = "test"
        translation_policy_version = "vi-dialogue-register-v3"

        def __init__(self):
            self.contexts = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            return [(item.text, f"raw {item.text}") for item in segments]

        def review_translation_register(self, segments, source_language, target_language, *, context):
            self.contexts.append(tuple(context))
            return [(item.segment_id, f"reviewed {item.translated_text}") for item in segments]

    transcript = make_transcript(segments=[
        LocalizationSegment(index, index + 1, f"line {index}") for index in range(121)
    ])
    provider = Provider()
    result = translate_for_localization(
        transcript, provider, checkpoint_path=tmp_path / "translation-checkpoint.json"
    )
    assert result.segments[0].translated_text == "reviewed raw line 0"
    assert len(provider.contexts) == 5
    assert len(provider.contexts[-1]) == 18
    assert [item.translated_text for item in provider.contexts[-1]] == [
        f"reviewed raw line {index}" for index in range(102, 120)
    ]

    resumed = Provider()
    translate_for_localization(
        transcript, resumed, checkpoint_path=tmp_path / "translation-checkpoint.json"
    )
    assert resumed.contexts == []


def test_register_review_checkpoint_is_bound_to_raw_content(tmp_path):
    class Provider:
        model = "test"
        translation_policy_version = "vi-dialogue-register-v3"

        def __init__(self, raw):
            self.raw, self.review_calls = raw, 0

        def translate_segments(self, segments, source_language, target_language, *, context):
            return [(item.text, self.raw) for item in segments]

        def review_translation_register(self, segments, source_language, target_language, *, context):
            self.review_calls += 1
            return [(item.segment_id, f"reviewed {item.translated_text}") for item in segments]

    checkpoint = tmp_path / "translation-checkpoint.json"
    transcript = make_transcript()
    first = Provider("raw one")
    translate_for_localization(transcript, first, checkpoint_path=checkpoint)
    # Mimic nondeterministic refreshed raw translation under a valid raw checkpoint identity.
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    next(iter(data["segments"].values()))["translated_text"] = "raw two"
    checkpoint.write_text(json.dumps(data), encoding="utf-8")
    second = Provider("unused")
    result = translate_for_localization(transcript, second, checkpoint_path=checkpoint)
    assert second.review_calls == 1
    assert result.segments[0].translated_text == "reviewed raw two"


def test_malformed_register_review_retries_then_splits_without_losing_entries():
    class Provider:
        def __init__(self):
            self.calls = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            return [(item.text, f"raw {item.text}") for item in segments]

        def review_translation_register(self, segments, source_language, target_language, *, context):
            self.calls.append(tuple(item.segment_id for item in segments))
            if len(segments) > 1:
                return []
            item = segments[0]
            return [(item.segment_id, f"reviewed {item.translated_text}")]

    provider = Provider()
    result = translate_for_localization(make_transcript(segments=[
        LocalizationSegment(0, 1, "one"), LocalizationSegment(1, 2, "two"),
    ]), provider)
    assert [item.translated_text for item in result.segments] == [
        "reviewed raw one", "reviewed raw two",
    ]
    assert len(provider.calls) == 4


def test_register_review_salvages_valid_items_without_resending_them(tmp_path):
    class Provider:
        def __init__(self):
            self.calls = []

        def translate_segments(self, segments, source_language, target_language, *, context):
            return [(item.text, f"raw {item.text}") for item in segments]

        def review_translation_register(self, segments, source_language, target_language, *, context):
            self.calls.append(tuple(item.segment_id for item in segments))
            if len(segments) == 1:
                return [(segments[0].segment_id, "...")]
            return [
                (segments[0].segment_id, f"reviewed {segments[0].translated_text}"),
                (segments[1].segment_id, "..."),
            ]

    checkpoint = tmp_path / "translation-checkpoint.json"
    provider = Provider()
    result = translate_for_localization(make_transcript(segments=[
        LocalizationSegment(0, 1, "one"), LocalizationSegment(1, 2, "two"),
        LocalizationSegment(2, 3, "three"),
    ]), provider, checkpoint_path=checkpoint)

    assert [item.translated_text for item in result.segments] == [
        "reviewed raw one", "reviewed raw two", "raw three",
    ]
    assert provider.calls == [
        (provider.calls[0][0], provider.calls[0][1], provider.calls[0][2]),
        (provider.calls[0][1], provider.calls[0][2]),
        # Both retries were already used by the batch; avoid extra singleton requests.
    ]
    saved = json.loads(
        (tmp_path / "translation-register-checkpoint.json").read_text(encoding="utf-8")
    )
    assert saved["segments"][provider.calls[0][2]]["review_status"] == "raw_fallback"


def test_invalid_singleton_review_uses_checkpointed_raw_fallback_on_resume(tmp_path):
    class Provider:
        def __init__(self):
            self.review_calls = 0

        def translate_segments(self, segments, source_language, target_language, *, context):
            return [(item.text, "tao sẽ giết người") for item in segments]

        def review_translation_register(self, segments, source_language, target_language, *, context):
            self.review_calls += 1
            return [(segments[0].segment_id, "tao sẽ giết người")]

    checkpoint = tmp_path / "translation-checkpoint.json"
    first = Provider()
    with pytest.raises(ValueError, match="refusing to put an unsafe raw fallback"):
        translate_for_localization(make_transcript(), first, checkpoint_path=checkpoint)
    assert first.review_calls == 2
    # Never cache a disallowed fallback and never silently return it on resume.
    reviewed = tmp_path / "translation-register-checkpoint.json"
    if reviewed.exists():
        assert "raw_fallback" not in reviewed.read_text(encoding="utf-8")


def test_register_review_provider_outage_raises_instead_of_raw_fallback():
    class Provider:
        def __init__(self):
            self.review_calls = 0

        def translate_segments(self, segments, source_language, target_language, *, context):
            return [(item.text, "raw translation") for item in segments]

        def review_translation_register(self, segments, source_language, target_language, *, context):
            self.review_calls += 1
            raise ConnectionError("review provider unavailable")

    provider = Provider()
    with pytest.raises(ConnectionError, match="review provider unavailable"):
        translate_for_localization(make_transcript(), provider)
    assert provider.review_calls == 2


@pytest.mark.parametrize("text, valid", [
    ("tao sẽ giết người", False), ("HỐ CHA thật", False),
    ("ta sẽ giết người", True), ("tao sẽ giết mày", False), ("Bình thường", True), ("Lông mày đẹp", True), ("MÀY đừng đi", False),
])
def test_review_validator(text, valid):
    assert _valid_reviewed_text(text) is valid
