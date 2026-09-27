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


def test_argos_provider_uses_worker(monkeypatch):
    from buzz.localization.providers import ArgosTranslationProvider

    calls = []

    def fake_worker(command, source_language, target_language="vi", text=None):
        calls.append((command, source_language, target_language, text))
        return {"ok": True, "route": ["en", "vi"], "text": "Xin chào"}

    monkeypatch.setattr(
        "buzz.localization.providers._run_argos_worker",
        fake_worker,
    )

    provider = ArgosTranslationProvider()
    assert provider.translate("Hello", "en", "vi") == "Xin chào"
    assert calls == [("translate", "en", "vi", "Hello")]


def test_argos_provider_supports_chinese_worker_route(monkeypatch):
    from buzz.localization.providers import ArgosTranslationProvider

    monkeypatch.setattr(
        "buzz.localization.providers._run_argos_worker",
        lambda *args, **kwargs: {
            "ok": True,
            "route": ["zh", "en", "vi"],
            "text": "Chào buổi sáng",
        },
    )

    assert (
        ArgosTranslationProvider().translate("早上好", "zh", "vi")
        == "Chào buổi sáng"
    )


def test_argos_provider_rejects_empty_text():
    from buzz.localization.providers import ArgosTranslationProvider

    with pytest.raises(ValueError, match="cannot be empty"):
        ArgosTranslationProvider().translate(" ", "en", "vi")


def test_argos_provider_rejects_unsupported_source():
    from buzz.localization.providers import ArgosTranslationProvider

    with pytest.raises(ValueError, match="Source language"):
        ArgosTranslationProvider().translate("Bonjour", "fr", "vi")


def test_argos_route_available_reports_worker_route(monkeypatch):
    from buzz.localization.providers import argos_route_available

    monkeypatch.setattr(
        "buzz.localization.providers._run_argos_worker",
        lambda *args, **kwargs: {"ok": True, "route": ["zh", "en", "vi"]},
    )

    assert argos_route_available("zh") == (
        True,
        "Argos route available: zh -> en -> vi",
    )


def test_argos_route_available_reports_worker_error(monkeypatch):
    from buzz.localization.providers import argos_route_available

    def fail(*args, **kwargs):
        raise LocalizationProviderError("missing zh->en route")

    monkeypatch.setattr(
        "buzz.localization.providers._run_argos_worker",
        fail,
    )

    assert argos_route_available("zh") == (False, "missing zh->en route")


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGeminiModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeGeminiResponse(self.text)


class FakeGeminiClient:
    def __init__(self, text):
        self.models = FakeGeminiModels(text)


def test_gemini_provider_returns_vietnamese_text():
    from buzz.localization.providers import GeminiTranslationProvider

    provider = GeminiTranslationProvider(api_key="test-key", model="gemini-test")
    provider._client = FakeGeminiClient(" Xin chào thế giới ")

    assert provider.translate("Hello world", "en", "vi") == "Xin chào thế giới"
    call = provider._client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert "Hello world" in call["contents"]


def test_gemini_provider_supports_chinese_source():
    from buzz.localization.providers import GeminiTranslationProvider

    provider = GeminiTranslationProvider(api_key="test-key", model="gemini-test")
    provider._client = FakeGeminiClient("Chào buổi sáng")
    assert provider.translate("早上好", "zh", "vi") == "Chào buổi sáng"


def test_gemini_provider_requires_key_and_model():
    from buzz.localization.providers import GeminiTranslationProvider

    with pytest.raises(ValueError, match="API key"):
        GeminiTranslationProvider(api_key="", model="gemini-test")
    with pytest.raises(ValueError, match="model"):
        GeminiTranslationProvider(api_key="key", model="")


def test_gemini_provider_rejects_empty_response():
    from buzz.localization.providers import GeminiTranslationProvider

    provider = GeminiTranslationProvider(api_key="key", model="gemini-test")
    provider._client = FakeGeminiClient("   ")
    with pytest.raises(LocalizationProviderError, match="empty"):
        provider.translate("Hello", "en", "vi")


def test_gemini_provider_wraps_network_error():
    from buzz.localization.providers import GeminiTranslationProvider

    class FailingModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("network down")

    class FailingClient:
        models = FailingModels()

    provider = GeminiTranslationProvider(api_key="SECRET", model="gemini-test")
    provider._client = FailingClient()
    with pytest.raises(LocalizationProviderError, match="Gemini translation request failed"):
        provider.translate("Hello", "en", "vi")
