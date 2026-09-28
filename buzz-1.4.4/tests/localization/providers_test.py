from pathlib import Path
from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace
import wave

import pytest

from buzz.localization.providers import (
    EdgeTTSProvider,
    LocalizationProviderError,
    OpenAICompatibleTranslationProvider,
    _probe_audio_duration,
    _trim_edge_tts_silence,
)
from buzz.localization.transcript import LocalizationSegment
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


def test_nllb_model_path_uses_configured_model_root(monkeypatch, tmp_path):
    from buzz.localization.providers import nllb_model_path

    monkeypatch.setenv("BUZZ_MODEL_ROOT", str(tmp_path / "models"))

    assert nllb_model_path() == tmp_path / "models" / "nllb" / "nllb-200-distilled-600M"


def _write_complete_nllb_model(path):
    path.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"):
        (path / filename).write_text("{}", encoding="utf-8")


def test_nllb_model_availability_rejects_incomplete_directory(monkeypatch, tmp_path):
    from buzz.localization.providers import nllb_model_is_available

    monkeypatch.setattr("buzz.localization.providers.nllb_model_path", lambda: tmp_path)
    _write_complete_nllb_model(tmp_path)
    assert nllb_model_is_available() is True

    (tmp_path / "model.safetensors").unlink()
    assert nllb_model_is_available() is False

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"model.layers.0": "model-00001-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    assert nllb_model_is_available() is False
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    assert nllb_model_is_available() is True


def test_nllb_load_downloads_once_then_loads_only_from_local_path(monkeypatch, tmp_path):
    from buzz.localization.providers import NLLBTranslationProvider, NLLB_MODEL_ID

    local_path = tmp_path / "nllb" / "nllb-200-distilled-600M"
    snapshot_calls = []
    tokenizer_calls = []
    model_calls = []

    def snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        _write_complete_nllb_model(Path(kwargs["local_dir"]))
        return str(kwargs["local_dir"])

    class FakeModel:
        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentencepiece", ModuleType("sentencepiece"))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer_calls.append((args, kwargs)) or object()),
            AutoModelForSeq2SeqLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model_calls.append((args, kwargs)) or FakeModel()),
        ),
    )
    monkeypatch.setattr("buzz.localization.providers.nllb_model_path", lambda: local_path)

    NLLBTranslationProvider()._load()

    assert snapshot_calls[0]["repo_id"] == NLLB_MODEL_ID
    assert snapshot_calls[0]["local_dir"] == local_path
    assert tokenizer_calls == [((local_path,), {"local_files_only": True})]
    assert model_calls == [((local_path,), {"local_files_only": True, "torch_dtype": fake_torch.float32})]

    NLLBTranslationProvider()._load()
    assert len(snapshot_calls) == 1


def test_nllb_translates_segments_independently_and_preserves_source(monkeypatch):
    from buzz.localization.providers import NLLBTranslationProvider

    provider = NLLBTranslationProvider(batch_size=2)
    calls = []
    monkeypatch.setattr(
        provider,
        "_translate_texts",
        lambda texts, language: calls.append((texts, language)) or ["Một", "Hai"],
    )
    segments = [
        LocalizationSegment(0, 1, " First  source "),
        LocalizationSegment(1, 2, "Second source"),
    ]

    assert provider.translate_segments(segments, "zh", "vi") == [
        (" First  source ", "Một"), ("Second source", "Hai")
    ]
    assert calls == [([" First  source ", "Second source"], "zh")]


def test_nllb_generation_uses_deterministic_beam_search_without_loading_model(monkeypatch):
    from buzz.localization.providers import NLLBTranslationProvider

    generation_calls = []

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["Hello"]
            assert kwargs == {
                "return_tensors": "pt", "padding": True, "truncation": True,
                "max_length": 512,
            }
            return {"input_ids": "input"}

        def convert_tokens_to_ids(self, token):
            assert token == "vie_Latn"
            return 42

        def batch_decode(self, generated, **kwargs):
            assert generated == ["generated"]
            assert kwargs == {"skip_special_tokens": True}
            return ["Xin chao"]

    class FakeModel:
        def generate(self, **kwargs):
            generation_calls.append(kwargs)
            return ["generated"]

    provider = NLLBTranslationProvider()
    provider._tokenizer = FakeTokenizer()
    provider._model = FakeModel()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=lambda: nullcontext()))

    assert provider.translate("Hello", "en", "vi") == "Xin chao"
    assert generation_calls == [{
        "input_ids": "input",
        "forced_bos_token_id": 42,
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
        "max_new_tokens": 256,
    }]


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


def _install_fake_edge_tts(monkeypatch, save_behavior=None):
    """Install a no-network Edge TTS module and return its save-call list."""
    calls = []

    class NoAudioReceived(Exception):
        pass

    class WebSocketError(Exception):
        pass

    class Communicate:
        def __init__(self, text, voice, **options):
            self.text = text
            self.voice = voice
            self.options = options

        def save_sync(self, output_file):
            calls.append((self.text, self.voice, self.options, output_file))
            if save_behavior:
                return save_behavior(output_file)
            Path(output_file).write_bytes(b"fake mp3")

    edge_module = ModuleType("edge_tts")
    edge_module.Communicate = Communicate
    edge_module.__version__ = "test-edge-tts"
    exceptions_module = ModuleType("edge_tts.exceptions")
    exceptions_module.NoAudioReceived = NoAudioReceived
    exceptions_module.WebSocketError = WebSocketError
    monkeypatch.setitem(sys.modules, "edge_tts", edge_module)
    monkeypatch.setitem(sys.modules, "edge_tts.exceptions", exceptions_module)
    return calls, NoAudioReceived, WebSocketError


def test_edge_tts_reuses_verified_matching_checkpoint(monkeypatch, tmp_path):
    calls, _, _ = _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.5)
    provider = EdgeTTSProvider()
    request = TTSRequest("Xin chao", "vi", str(tmp_path / "segment-000000"))

    first = provider.synthesize(request)
    second = provider.synthesize(request)

    assert len(calls) == 1
    assert second == first
    assert (tmp_path / "segment-000000.tts-cache.json").is_file()


@pytest.mark.parametrize("changed_request", [
    {"text": "Xin chao lai"},
    {"voice": "vi-VN-NamMinhNeural"},
    {"options": {"rate": "+10%"}},
])
def test_edge_tts_cache_invalidates_when_request_changes(
    monkeypatch, tmp_path, changed_request
):
    calls, _, _ = _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.5)
    provider = EdgeTTSProvider()
    base = TTSRequest("Xin chao", "vi", str(tmp_path / "segment-000000"), voice="vi-VN-HoaiMyNeural")
    provider.synthesize(base)
    provider.synthesize(TTSRequest(
        changed_request.get("text", base.text), base.language, base.output_file_stem,
        voice=changed_request.get("voice", base.voice),
        options=changed_request.get("options", base.options),
    ))

    assert len(calls) == 2


def test_edge_tts_regenerates_corrupt_checkpointed_audio(monkeypatch, tmp_path):
    calls, _, _ = _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    probe_calls = []
    def probe(path):
        probe_calls.append(Path(path))
        if len(probe_calls) == 2:
            raise LocalizationProviderError("corrupt audio")
        return 1.5
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", probe)
    provider = EdgeTTSProvider()
    request = TTSRequest("Xin chao", "vi", str(tmp_path / "segment-000000"))
    provider.synthesize(request)
    provider.synthesize(request)

    assert len(calls) == 2


def test_edge_tts_writes_checkpoint_after_success(monkeypatch, tmp_path):
    _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 2.0)

    EdgeTTSProvider().synthesize(TTSRequest(
        "Xin chao", "vi", str(tmp_path / "segment-000000"),
        voice="vi-VN-HoaiMyNeural", options={"rate": "+5%"},
    ))

    checkpoint = json.loads((tmp_path / "segment-000000.tts-cache.json").read_text())
    assert len(checkpoint["fingerprint"]) == 64
    assert checkpoint["voice"] == "vi-VN-HoaiMyNeural"
    assert checkpoint["metadata"]["rate"] == "+5%"


def test_edge_tts_transient_retries_are_extended_and_finite(monkeypatch, tmp_path):
    def no_audio(output_file):
        raise no_audio_exception("No audio")
    calls, no_audio_exception, _ = _install_fake_edge_tts(monkeypatch, no_audio)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)

    with pytest.raises(LocalizationProviderError, match="after 8 attempts"):
        EdgeTTSProvider().synthesize(TTSRequest("Xin chao", "vi", str(tmp_path / "segment")))

    assert len(calls) == 8
    assert sleeps == [5, 10, 20, 40, 60, 90, 120]
    assert not (tmp_path / "segment.mp3").exists()


def test_edge_tts_non_transient_error_fails_without_retry(monkeypatch, tmp_path):
    def invalid_parameter(output_file):
        Path(output_file).write_bytes(b"partial")
        raise ValueError("invalid parameter")
    calls, _, _ = _install_fake_edge_tts(monkeypatch, invalid_parameter)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)

    with pytest.raises(LocalizationProviderError, match="invalid parameter"):
        EdgeTTSProvider().synthesize(TTSRequest("Xin chao", "vi", str(tmp_path / "segment")))

    assert len(calls) == 1
    assert sleeps == []
    assert not (tmp_path / "segment.mp3").exists()

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

def test_gemini_batch_corrects_chinese_and_validates_ids():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider
    from buzz.localization.transcript import LocalizationSegment

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            payload = json.loads(kwargs["contents"].split("Input JSON:\n", 1)[1])
            assert len(payload["segments"]) == 2
            assert payload["segments"][0]["start"] == 0
            assert payload["context_only"][0]["text"] == "机器人"
            return SimpleNamespace(text=json.dumps({
                "segments": [
                    {"id": 1, "corrected_text": "自动回充", "translated_text": "Tự về sạc"},
                    {"id": 0, "corrected_text": "吸力很强", "translated_text": "Lực hút mạnh"},
                ]
            }, ensure_ascii=False))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._client = SimpleNamespace(models=FakeModels())
    result = provider.translate_segments(
        [LocalizationSegment(0, 1, "吸力很强"),
         LocalizationSegment(1, 2, "自动回冲")],
        "zh", "vi", context=[LocalizationSegment(-1, 0, "机器人")],
    )
    assert result == [("吸力很强", "Lực hút mạnh"), ("自动回充", "Tự về sạc")]
    assert provider._client.models.calls[0]["config"]["response_mime_type"] == "application/json"


def test_gemini_batch_rejects_missing_id():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider
    from buzz.localization.transcript import LocalizationSegment

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs:
            SimpleNamespace(text='{"segments":[{"id":9,"corrected_text":"a","translated_text":"b"}]}'))
    )
    with pytest.raises(LocalizationProviderError, match="segment ID"):
        provider.translate_segments([LocalizationSegment(0, 1, "a")], "zh", "vi")


def test_gemini_display_name_is_normalized_to_api_id():
    from buzz.localization.providers import GeminiTranslationProvider

    provider = GeminiTranslationProvider(
        api_key="test-key",
        model="Gemini 3.8 Flash",
    )
    assert provider.model == "gemini-3.8-flash"


def test_gemini_retries_and_falls_back_on_503(monkeypatch):
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise RuntimeError("503 UNAVAILABLE")
            return SimpleNamespace(text="Xin chào")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model = "gemini-3.8-flash"
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr(
        "buzz.localization.providers.time.sleep",
        lambda seconds: None,
    )

    assert provider.translate("Hello", "en", "vi") == "Xin chào"
    assert calls == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ]
    assert provider.last_model_used == "gemini-3.5-flash-lite"
