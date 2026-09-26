from pathlib import Path
import wave

import pytest

from buzz.localization.providers import (
    EdgeTTSProvider,
    LocalizationProviderError,
    OpenAICompatibleTranslationProvider,
    _probe_audio_duration,
    _trim_edge_tts_silence,
)
from buzz.localization.tts import TTSRequest


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, content):
        self.completions = FakeCompletions(content)


class FakeClient:
    def __init__(self, content):
        self.chat = FakeChat(content)


def test_openai_compatible_translation_provider_returns_vietnamese_text():
    provider = OpenAICompatibleTranslationProvider(
        api_key="test-key",
        model="test-model",
    )
    provider._client = FakeClient(" Xin chào thế giới ")

    result = provider.translate("Hello world", "en", "vi")

    assert result == "Xin chào thế giới"
    call = provider._client.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"][1]["content"] == "Hello world"


def test_translation_provider_supports_chinese_source():
    provider = OpenAICompatibleTranslationProvider(
        api_key="test-key",
        model="test-model",
    )
    provider._client = FakeClient("Chào buổi sáng")

    assert provider.translate("早上好", "zh", "vi") == "Chào buổi sáng"


@pytest.mark.parametrize("source", ["fr", "", None])
def test_translation_provider_rejects_unsupported_source(source):
    provider = OpenAICompatibleTranslationProvider(
        api_key="test-key",
        model="test-model",
    )
    provider._client = FakeClient("unused")

    with pytest.raises(ValueError, match="Source language"):
        provider.translate("text", source, "vi")


def test_translation_provider_requires_credentials_and_model():
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleTranslationProvider(api_key="", model="model")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleTranslationProvider(api_key="key", model="")


def test_probe_audio_duration_uses_ffprobe(tmp_path):
    wav_path = tmp_path / "audio.wav"
    sample_rate = 8000
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate)

    assert _probe_audio_duration(wav_path) == pytest.approx(1.0, abs=0.02)


def test_edge_tts_rejects_non_vietnamese_request(tmp_path):
    provider = EdgeTTSProvider()

    with pytest.raises(ValueError, match="Vietnamese"):
        provider.synthesize(
            TTSRequest(
                text="Hello",
                language="en",
                output_file_stem=str(tmp_path / "speech"),
            )
        )


def test_edge_tts_rejects_empty_text(tmp_path):
    provider = EdgeTTSProvider()

    with pytest.raises(ValueError, match="cannot be empty"):
        provider.synthesize(
            TTSRequest(
                text=" ",
                language="vi",
                output_file_stem=str(tmp_path / "speech"),
            )
        )

def test_trim_edge_tts_silence_removes_boundary_silence(tmp_path):
    wav_path = tmp_path / "edge.wav"
    sample_rate = 8000
    leading_silence = b"\x00\x00" * int(sample_rate * 0.2)
    tone = b"\x10\x27\xf0\xd8" * int(sample_rate * 0.25)
    trailing_silence = b"\x00\x00" * int(sample_rate * 1.0)

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(leading_silence + tone + trailing_silence)

    before = _probe_audio_duration(wav_path)
    _trim_edge_tts_silence(wav_path)
    after = _probe_audio_duration(wav_path)

    assert before == pytest.approx(1.7, abs=0.05)
    assert after < before - 0.7
    assert 0.4 <= after <= 0.8
