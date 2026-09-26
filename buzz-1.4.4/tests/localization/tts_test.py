import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from buzz.localization.translation import (
    TranslatedLocalizationSegment,
    TranslatedLocalizationTranscript,
)
from buzz.localization.tts import (
    SynthesizedLocalizationSegment,
    TTSRequest,
    TTSResult,
    synthesize_for_localization,
)


class FakeTTSProvider:
    def __init__(self, durations=None, provider="fake", voice="vi-test"):
        self.durations = durations or {}
        self.provider = provider
        self.voice = voice
        self.calls = []

    def synthesize(self, request: TTSRequest):
        self.calls.append(request)
        audio_file = request.output_file_stem + ".wav"
        Path(audio_file).write_bytes(b"fake audio")
        return TTSResult(
            audio_file=audio_file,
            audio_duration=self.durations.get(request.text, 1.25),
            provider=self.provider,
            voice=request.voice or self.voice,
            metadata={"format": "wav"},
        )


def make_transcript(segments=None, target_language="vi"):
    return TranslatedLocalizationTranscript(
        source_file="input.mp4",
        source_language="en",
        target_language=target_language,
        segments=tuple(
            segments
            if segments is not None
            else [
                TranslatedLocalizationSegment(
                    start=0.0,
                    end=3.2,
                    source_text="Hello everyone",
                    translated_text="Xin chào mọi người",
                )
            ]
        ),
    )


def test_synthesizes_vietnamese_text_unchanged(tmp_path):
    transcript = make_transcript()
    provider = FakeTTSProvider()

    result = synthesize_for_localization(
        transcript, provider, tmp_path, voice="vi-female"
    )

    request = provider.calls[0]
    assert request.text == "Xin chào mọi người"
    assert request.language == "vi"
    assert request.output_file_stem == str(tmp_path / "segment-000000")
    assert request.voice == "vi-female"
    assert result.segments == (
        SynthesizedLocalizationSegment(
            start=0.0,
            end=3.2,
            source_text="Hello everyone",
            translated_text="Xin chào mọi người",
            audio_file=str(tmp_path / "segment-000000.wav"),
            audio_duration=1.25,
            provider="fake",
            voice="vi-female",
            provider_metadata={"format": "wav"},
        ),
    )


def test_preserves_segment_order_and_audio_mapping(tmp_path):
    transcript = make_transcript(
        segments=[
            TranslatedLocalizationSegment(5.0, 7.0, "Second", "Thứ hai"),
            TranslatedLocalizationSegment(0.0, 2.0, "First", "Thứ nhất"),
        ]
    )
    provider = FakeTTSProvider({"Thứ hai": 0.8, "Thứ nhất": 1.1})

    result = synthesize_for_localization(transcript, provider, tmp_path)

    assert [segment.translated_text for segment in result.segments] == [
        "Thứ hai",
        "Thứ nhất",
    ]
    assert [segment.audio_file for segment in result.segments] == [
        str(tmp_path / "segment-000000.wav"),
        str(tmp_path / "segment-000001.wav"),
    ]
    assert [segment.audio_duration for segment in result.segments] == [0.8, 1.1]


def test_preserves_unicode_and_provider_metadata(tmp_path):
    text = "Tôi đang học tiếng Việt ở Hà Nội."
    transcript = make_transcript(
        segments=[TranslatedLocalizationSegment(0.2, 2.9, "Source", text)]
    )
    provider = FakeTTSProvider(provider="example", voice="vi-VN-1")

    result = synthesize_for_localization(transcript, provider, tmp_path)

    assert result.segments[0].translated_text == text
    assert result.segments[0].provider == "example"
    assert result.segments[0].voice == "vi-VN-1"
    assert result.segments[0].provider_metadata == {"format": "wav"}


def test_passes_provider_options_without_mutating_them(tmp_path):
    options = {"rate": "+5%", "pitch": "default"}
    provider = FakeTTSProvider()

    synthesize_for_localization(
        make_transcript(), provider, tmp_path, options=options
    )

    assert dict(provider.calls[0].options) == options
    assert options == {"rate": "+5%", "pitch": "default"}


def test_output_is_json_compatible(tmp_path):
    result = synthesize_for_localization(
        make_transcript(), FakeTTSProvider(), tmp_path
    )

    encoded = json.dumps(result.to_dict(), ensure_ascii=False)

    assert json.loads(encoded) == result.to_dict()
    assert "Xin chào mọi người" in encoded


def test_rejects_empty_transcript_without_calling_provider(tmp_path):
    provider = FakeTTSProvider()

    with pytest.raises(ValueError, match="empty transcript"):
        synthesize_for_localization(
            make_transcript(segments=[]), provider, tmp_path
        )

    assert provider.calls == []


@pytest.mark.parametrize("text", ["", "   ", None])
def test_rejects_empty_translated_segment(text, tmp_path):
    transcript = make_transcript(
        segments=[TranslatedLocalizationSegment(0, 1, "Source", text)]
    )
    provider = FakeTTSProvider()

    with pytest.raises(ValueError, match="empty translated segment"):
        synthesize_for_localization(transcript, provider, tmp_path)

    assert provider.calls == []


def test_rejects_non_vietnamese_target_language(tmp_path):
    with pytest.raises(ValueError, match="target language"):
        synthesize_for_localization(
            make_transcript(target_language="en"),
            FakeTTSProvider(),
            tmp_path,
        )


@pytest.mark.parametrize("output_directory", ["", None])
def test_rejects_missing_output_directory(output_directory):
    with pytest.raises(ValueError, match="Output directory"):
        synthesize_for_localization(
            make_transcript(), FakeTTSProvider(), output_directory
        )


def test_propagates_provider_exception(tmp_path):
    class FailingProvider:
        def synthesize(self, request):
            raise TimeoutError("tts timed out")

    with pytest.raises(TimeoutError, match="tts timed out"):
        synthesize_for_localization(
            make_transcript(), FailingProvider(), tmp_path
        )


def test_rejects_invalid_provider_response(tmp_path):
    class InvalidProvider:
        def synthesize(self, request):
            return {"audio_file": request.output_file_stem + ".wav"}

    with pytest.raises(ValueError, match="invalid response"):
        synthesize_for_localization(
            make_transcript(), InvalidProvider(), tmp_path
        )


@pytest.mark.parametrize("duration", [0, -1, None, "1.0", True])
def test_rejects_invalid_audio_duration(duration, tmp_path):
    class InvalidDurationProvider:
        def synthesize(self, request):
            audio_file = request.output_file_stem + ".wav"
            Path(audio_file).write_bytes(b"fake")
            return TTSResult(
                audio_file=audio_file,
                audio_duration=duration,
                provider="fake",
            )

    with pytest.raises(ValueError, match="invalid audio duration"):
        synthesize_for_localization(
            make_transcript(), InvalidDurationProvider(), tmp_path
        )


def test_rejects_missing_audio_file(tmp_path):
    class MissingFileProvider:
        def synthesize(self, request):
            return TTSResult(
                audio_file=request.output_file_stem + ".wav",
                audio_duration=1.0,
                provider="fake",
            )

    with pytest.raises(ValueError, match="does not exist"):
        synthesize_for_localization(
            make_transcript(), MissingFileProvider(), tmp_path
        )


@pytest.mark.parametrize("provider_value", ["", "   ", None])
def test_rejects_invalid_provider_metadata(provider_value, tmp_path):
    class InvalidMetadataProvider:
        def synthesize(self, request):
            audio_file = request.output_file_stem + ".wav"
            Path(audio_file).write_bytes(b"fake")
            return TTSResult(
                audio_file=audio_file,
                audio_duration=1.0,
                provider=provider_value,
            )

    with pytest.raises(ValueError, match="provider metadata"):
        synthesize_for_localization(
            make_transcript(), InvalidMetadataProvider(), tmp_path
        )


def test_does_not_mutate_input_transcript(tmp_path):
    transcript = make_transcript()
    original = transcript.to_dict()

    synthesize_for_localization(
        transcript, FakeTTSProvider(), tmp_path
    )

    assert transcript.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        transcript.target_language = "en"
