from pathlib import Path
import asyncio
from contextlib import nullcontext
import json
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import wave

import socket

import pytest
import buzz.localization.providers as providers

from buzz.localization.providers import (
    EdgeTTSProvider,
    LocalizationProviderError,
    OpenAICompatibleTranslationProvider,
    _probe_audio_duration,
    _trim_edge_tts_silence,
)
from buzz.localization.transcript import LocalizationSegment
from buzz.localization.translation import (
    TranslatedLocalizationSegment, TranslatedLocalizationTranscript,
)
from buzz.localization.tts import (
    TTSBatchFallbackRequired, TTSRequest, TTSServiceTemporarilyUnavailable,
    synthesize_for_localization,
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


@pytest.fixture(autouse=True)
def _disable_edge_tts_pacing_waits(monkeypatch):
    """Keep provider tests fast; pacing behavior has focused deterministic tests."""
    monkeypatch.setattr(
        EdgeTTSProvider, "_pacing_sleep", staticmethod(lambda seconds: None)
    )
    monkeypatch.setattr(EdgeTTSProvider, "_last_network_request_monotonic", None)
    monkeypatch.setattr(EdgeTTSProvider, "_service_circuit_until", 0.0)


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


def test_tts_audio_probe_uses_explicit_robust_utf8_decoding(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return __import__("subprocess").CompletedProcess(
            command, 0, "2.5\n", "合成营业音频"
        )

    monkeypatch.setattr("buzz.localization.providers.subprocess.run", fake_run)

    assert _probe_audio_duration(Path("中文语音.mp3")) == 2.5
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


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


def test_openai_compatible_provider_falls_back_after_rate_limit():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary":
                raise RuntimeError("429 rate limit exceeded")
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Bản dịch dự phòng")
            )])

    provider = OpenAICompatibleTranslationProvider(
        api_key="test-key", model="primary, backup",
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    assert provider.translate("Hello", "en", "vi") == "Bản dịch dự phòng"
    assert calls == ["primary", "backup"]
    assert provider.last_model_used == "backup"


def test_openai_compatible_provider_reports_all_models_rate_limited():
    provider = OpenAICompatibleTranslationProvider(
        api_key="test-key", model="primary, backup",
    )
    provider._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("429 rate limit exceeded")
            )
        )
    ))

    with pytest.raises(LocalizationProviderError, match="every configured model"):
        provider.translate("Hello", "en", "vi")


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


@pytest.mark.parametrize("text", ["...", "??????", "???"])
def test_edge_tts_rejects_non_speakable_text_before_calling_edge(monkeypatch, tmp_path, text):
    calls, _, _ = _install_fake_edge_tts(monkeypatch)

    with pytest.raises(ValueError, match="at least one letter or number"):
        EdgeTTSProvider().synthesize(TTSRequest(text, "vi", str(tmp_path / "speech")))

    assert calls == []


def test_edge_tts_accepts_speakable_text_with_punctuation(monkeypatch, tmp_path):
    calls, _, _ = _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)

    EdgeTTSProvider().synthesize(TTSRequest("18 gi???", "vi", str(tmp_path / "speech")))

    assert len(calls) == 1


def _install_fake_edge_tts(monkeypatch, save_behavior=None, stream_behavior=None):
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

        async def save(self, output_file):
            calls.append((self.text, self.voice, self.options, output_file))
            if save_behavior:
                result = save_behavior(output_file)
                if hasattr(result, "__await__"):
                    return await result
                return result
            Path(output_file).write_bytes(b"fake mp3")

        async def stream(self):
            calls.append((self.text, self.voice, self.options, "stream"))
            if stream_behavior:
                result = stream_behavior(self)
                if hasattr(result, "__aiter__"):
                    async for event in result:
                        yield event
                    return
                if hasattr(result, "__await__"):
                    await result
                    return
            yield {"type": "audio", "data": b"fake mp3"}

    edge_module = ModuleType("edge_tts")
    edge_module.Communicate = Communicate
    edge_module.__version__ = "test-edge-tts"
    exceptions_module = ModuleType("edge_tts.exceptions")
    exceptions_module.NoAudioReceived = NoAudioReceived
    exceptions_module.WebSocketError = WebSocketError
    monkeypatch.setitem(sys.modules, "edge_tts", edge_module)
    monkeypatch.setitem(sys.modules, "edge_tts.exceptions", exceptions_module)

    # Unit tests keep the fake in-process; production always uses the spawned
    # worker, which cannot inherit pytest's sys.modules monkeypatches on Windows.
    def fake_attempt(text, voice, options, output_file, *, batch, timeout_seconds=45):
        communicate = Communicate(text, voice, **options)
        if not batch:
            try:
                return asyncio.run(asyncio.wait_for(
                    communicate.save(output_file), timeout=timeout_seconds
                ))
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Edge TTS synthesis exceeded {timeout_seconds} seconds"
                ) from exc

        async def stream():
            boundaries = []
            with Path(output_file).open("wb") as handle:
                async for event in communicate.stream():
                    if event.get("type") == "audio":
                        handle.write(event["data"])
                    elif event.get("type") == "WordBoundary":
                        boundaries.append(event)
            return boundaries

        try:
            return asyncio.run(asyncio.wait_for(stream(), timeout=timeout_seconds))
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Edge TTS synthesis exceeded {timeout_seconds} seconds"
            ) from exc

    monkeypatch.setattr("buzz.localization.providers._run_edge_tts_attempt", fake_attempt)
    return calls, NoAudioReceived, WebSocketError


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_edge_tts_hard_attempt_timeout_terminates_hung_child_promptly(
    monkeypatch, tmp_path, batch
):
    """The parent must kill a hung child instead of waiting for cancellation."""
    processes = []

    class HungProcess:
        def __init__(self, *, target, args):
            self.args = args
            self.alive = True
            self.terminated = False

        def start(self):
            Path(self.args[3]).write_bytes(b"partial")

        def join(self, _timeout=None):
            return None

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.alive = False

    class Context:
        @staticmethod
        def Process(*, target, args):
            process = HungProcess(target=target, args=args)
            processes.append(process)
            return process

    monkeypatch.setattr(providers.multiprocessing, "get_context", lambda _: Context())
    output = tmp_path / ("batch.mp3" if batch else "single.mp3")
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="0.001 seconds"):
        providers._run_edge_tts_attempt(
            "Xin chao", "vi-VN-HoaiMyNeural", {}, str(output), batch=batch,
            timeout_seconds=0.001,
        )

    assert time.monotonic() - started < 0.5
    assert processes[0].terminated is True
    assert not output.exists()


def test_edge_tts_batch_marker_boundaries_make_marker_free_intervals():
    intervals = EdgeTTSProvider._batch_intervals_from_boundaries(
        ["MARK1", "MARK2"],
        [
            {"text": "MARK1", "offset": 10_000_000, "duration": 2_000_000},
            {"text": "MARK2", "offset": 30_000_000, "duration": 1_000_000},
        ],
        5.0,
    )
    assert intervals == [(0.0, 1.0), (1.2, 3.0), (3.1, 5.0)]


def test_edge_tts_batch_marker_validation_rejects_duplicates():
    with pytest.raises(ValueError, match="missing or duplicated"):
        EdgeTTSProvider._batch_intervals_from_boundaries(
            ["MARK1"],
            [{"text": "MARK1", "offset": 1, "duration": 1},
             {"text": "MARK1", "offset": 2, "duration": 1}],
            1.0,
        )


def test_edge_tts_batch_marker_validation_rejects_out_of_order_boundaries():
    with pytest.raises(ValueError, match="out of order"):
        EdgeTTSProvider._batch_intervals_from_boundaries(
            ["MARK1", "MARK2"],
            [
                {"text": "MARK1", "offset": 30_000_000, "duration": 1},
                {"text": "MARK2", "offset": 10_000_000, "duration": 1},
            ],
            5.0,
        )


def test_edge_tts_batch_stream_splits_four_outputs_and_writes_provider_caches(
    monkeypatch, tmp_path
):
    async def stream_events(communicate):
        markers = [word for word in communicate.text.split() if word.startswith("ZXQ")]
        yield {"type": "audio", "data": b"combined"}
        for index, marker in enumerate(markers):
            yield {"type": "WordBoundary", "text": marker,
                   "offset": (1 + index * 2) * 10_000_000,
                   "duration": 2_000_000}

    calls, _, _ = _install_fake_edge_tts(monkeypatch, stream_behavior=stream_events)
    intervals = []
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 8.0)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr(
        EdgeTTSProvider, "_split_batch_audio",
        staticmethod(lambda combined, output, start, end: (intervals.append((start, end)), output.write_bytes(b"split"))),
    )
    requests = [TTSRequest(f"text {index}", "vi", str(tmp_path / f"segment-{index}"))
                for index in range(4)]

    results = EdgeTTSProvider(retry_delays=(0,)).synthesize_batch(requests)

    assert [call[3] for call in calls] == ["stream"]
    assert len(results) == 4
    assert intervals == [(0.0, 1.0), (1.2, 3.0), (3.2, 5.0), (5.2, 8.0)]
    assert all(Path(result.audio_file).is_file() for result in results)
    assert all((tmp_path / f"segment-{index}.tts-cache.json").is_file() for index in range(4))


def test_edge_tts_batch_nonce_avoids_marker_text(monkeypatch):
    values = iter([SimpleNamespace(hex="AB000000"), SimpleNamespace(hex="CD000000")])
    monkeypatch.setattr("buzz.localization.providers.uuid.uuid4", lambda: next(values))

    assert EdgeTTSProvider._batch_markers(2, "text ZXQAB0 already here") == ["ZXQCD0"]


def test_edge_tts_partial_provider_cache_batches_only_misses_in_localization(
    monkeypatch, tmp_path
):
    async def stream_events(communicate):
        markers = [word for word in communicate.text.split() if word.startswith("ZXQ")]
        yield {"type": "audio", "data": b"combined"}
        for index, marker in enumerate(markers):
            yield {"type": "WordBoundary", "text": marker,
                   "offset": (1 + index * 2) * 10_000_000,
                   "duration": 2_000_000}

    calls, _, _ = _install_fake_edge_tts(monkeypatch, stream_behavior=stream_events)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 8.0)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr(EdgeTTSProvider, "_split_batch_audio",
                        staticmethod(lambda _c, output, _s, _e: output.write_bytes(b"split")))
    provider = EdgeTTSProvider(retry_delays=(0,))
    segments = [TranslatedLocalizationSegment(index, index + 1, str(index), f"text {index}")
                for index in range(4)]
    transcript = TranslatedLocalizationTranscript("en", "vi", "input.mp4", tuple(segments))
    provider.synthesize(TTSRequest("text 0", "vi", str(tmp_path / "segment-000000")))
    calls.clear()

    result = synthesize_for_localization(transcript, provider, tmp_path)

    assert len(calls) == 1
    assert calls[0][3] == "stream"
    assert "text 0" not in calls[0][0]
    assert [segment.translated_text for segment in result.segments] == [f"text {i}" for i in range(4)]
    assert all((tmp_path / f"segment-{index:06d}.localization-tts.json").is_file()
               for index in range(4))


def test_edge_tts_batch_no_audio_exhaustion_falls_back_to_ordered_singles(monkeypatch, tmp_path):
    async def no_audio(_communicate):
        raise NoAudioReceived("empty")
        yield  # pragma: no cover

    calls, NoAudioReceived, _ = _install_fake_edge_tts(monkeypatch, stream_behavior=no_audio)
    provider = EdgeTTSProvider(retry_delays=(0,))
    requests = [TTSRequest(f"text {index}", "vi", str(tmp_path / f"segment-{index}"))
                for index in range(2)]

    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)

    with pytest.raises(TTSBatchFallbackRequired):
        provider.synthesize_batch(requests)

    assert [call[3] for call in calls] == ["stream"]
    assert [request.text for request in requests] == ["text 0", "text 1"]
    assert not list(tmp_path.glob(".edge-batch-*"))
    assert not any((tmp_path / f"segment-{index}.tts-cache.json").exists()
                   for index in range(2))


def test_edge_tts_disables_batch_after_two_failures(monkeypatch, tmp_path):
    async def no_audio(_communicate):
        raise NoAudioReceived("empty")
        yield  # pragma: no cover

    calls, NoAudioReceived, _ = _install_fake_edge_tts(
        monkeypatch, stream_behavior=no_audio
    )
    provider = EdgeTTSProvider(retry_delays=(0,))
    requests = [
        TTSRequest(f"text {index}", "vi", str(tmp_path / f"segment-{index}"))
        for index in range(2)
    ]

    for _ in range(2):
        with pytest.raises(TTSBatchFallbackRequired):
            provider.synthesize_batch(requests)

    calls_before_disabled_attempt = len(calls)
    with pytest.raises(TTSBatchFallbackRequired):
        provider.synthesize_batch(requests)

    assert provider._batch_disabled is True
    assert provider._batch_failure_count == 2
    assert len(calls) == calls_before_disabled_attempt


def test_edge_tts_batch_can_be_disabled_without_network_call(tmp_path):
    provider = EdgeTTSProvider(batch_enabled=False)
    requests = [
        TTSRequest(f"text {index}", "vi", str(tmp_path / f"segment-{index}"))
        for index in range(2)
    ]

    with pytest.raises(TTSBatchFallbackRequired):
        provider.synthesize_batch(requests)

    assert provider._batch_failure_count == 0
    assert provider._batch_disabled is False


def test_edge_tts_batch_no_audio_falls_back_after_two_attempts_with_default_delays(
    monkeypatch, tmp_path,
):
    async def no_audio(_communicate):
        raise NoAudioReceived("empty")
        yield  # pragma: no cover

    calls, NoAudioReceived, _ = _install_fake_edge_tts(monkeypatch, stream_behavior=no_audio)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    requests = [TTSRequest(f"text {index}", "vi", str(tmp_path / f"segment-{index}"))
                for index in range(2)]

    with pytest.raises(TTSBatchFallbackRequired):
        EdgeTTSProvider().synthesize_batch(requests)

    assert [call[3] for call in calls] == ["stream"]
    assert sleeps == []


def test_edge_tts_safe_singleton_payload_preserves_request_and_reuses_cache(
    monkeypatch, tmp_path
):
    _, NoAudioReceived, _ = _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    attempted_payloads = []

    def save(text, _voice, _options, output_file):
        attempted_payloads.append(text)
        if text.endswith("?"):
            raise NoAudioReceived("empty")
        Path(output_file).write_bytes(b"audio")

    provider = EdgeTTSProvider(retry_delays=(0,))
    monkeypatch.setattr(provider, "_save_with_attempt_timeout", save)
    request = TTSRequest("Ả phụ nữ đó là ai?", "vi", str(tmp_path / "bad"))

    result = provider.synthesize(request)

    assert request.text == "Ả phụ nữ đó là ai?"
    assert attempted_payloads == [
        "Ả phụ nữ đó là ai?", "Ả phụ nữ đó là ai?", "Ả phụ nữ đó là ai",
    ]
    checkpoint = json.loads((tmp_path / "bad.tts-cache.json").read_text())
    assert checkpoint["fingerprint"] != provider._cache_fingerprint(
        request, provider.default_voice, "+0%", "+0%", "+0Hz", "test-edge-tts",
    )
    attempted_payloads.clear()

    assert provider.synthesize(request) == result
    assert attempted_payloads == []


def test_edge_tts_global_no_audio_opens_circuit_after_short_singleton_probe(
    monkeypatch, tmp_path,
):
    _, NoAudioReceived, _ = _install_fake_edge_tts(monkeypatch)
    calls = []

    def no_audio(text, _voice, _options, _output):
        calls.append(text)
        raise NoAudioReceived("empty")

    provider = EdgeTTSProvider(retry_delays=(0,))
    monkeypatch.setattr(provider, "_save_with_attempt_timeout", no_audio)
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda _delay: None)
    request = TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))

    with pytest.raises(TTSServiceTemporarilyUnavailable):
        provider.synthesize(request)
    assert calls == ["Xin chao", "Xin chao", "Xin chào, đây là câu kiểm tra bình thường."]
    assert EdgeTTSProvider._circuit_remaining_seconds() > 0

    second = EdgeTTSProvider(retry_delays=(0,))
    monkeypatch.setattr(second, "_save_with_attempt_timeout", no_audio)
    with pytest.raises(TTSServiceTemporarilyUnavailable):
        second.synthesize(TTSRequest("Khac", "vi", str(tmp_path / "other")))
    assert len(calls) == 3


def test_edge_tts_payload_no_audio_with_successful_health_probe_defers_retry_to_workflow(
    monkeypatch, tmp_path,
):
    _, NoAudioReceived, _ = _install_fake_edge_tts(monkeypatch)
    calls = []
    health_text = "Xin chào, đây là câu kiểm tra bình thường."

    def no_audio_except_health_probe(text, _voice, _options, output_file):
        calls.append(text)
        if text == health_text:
            Path(output_file).write_bytes(b"health audio")
            return
        raise NoAudioReceived("empty")

    provider = EdgeTTSProvider(retry_delays=(0,))
    monkeypatch.setattr(provider, "_save_with_attempt_timeout", no_audio_except_health_probe)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)

    with pytest.raises(TTSServiceTemporarilyUnavailable) as caught:
        provider.synthesize(TTSRequest("Xin chao", "vi", str(tmp_path / "segment")))

    assert calls == ["Xin chao", "Xin chao", health_text]
    assert sleeps == [1.0]
    assert caught.value.retry_after_seconds == 12
    assert EdgeTTSProvider._circuit_remaining_seconds() == 0


def test_edge_tts_payload_healthy_probe_does_not_retry_inside_provider(
    monkeypatch, tmp_path,
):
    _, NoAudioReceived, _ = _install_fake_edge_tts(monkeypatch)
    calls = []
    health_text = "Xin chào, đây là câu kiểm tra bình thường."

    def succeeds_on_degraded_retry(text, _voice, _options, output_file):
        calls.append(text)
        if text == health_text or calls.count(text) >= 3:
            Path(output_file).write_bytes(b"audio")
            return
        raise NoAudioReceived("empty")

    provider = EdgeTTSProvider(retry_delays=(0,))
    monkeypatch.setattr(provider, "_save_with_attempt_timeout", succeeds_on_degraded_retry)
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda _delay: None)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)

    with pytest.raises(TTSServiceTemporarilyUnavailable):
        provider.synthesize(TTSRequest("Xin chao", "vi", str(tmp_path / "segment")))

    assert calls == ["Xin chao", "Xin chao", health_text]
    assert EdgeTTSProvider._circuit_remaining_seconds() == 0


def test_edge_tts_safe_payload_replaces_only_known_initial_wording():
    assert providers._edge_tts_safe_payloads("Ả phụ nữ đó là ai") == (
        "Người phụ nữ đó là ai",
    )


def test_edge_tts_batch_structural_marker_failure_falls_back_to_singles(
    monkeypatch, tmp_path
):
    calls, _, _ = _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr(
        "buzz.localization.providers._trim_edge_tts_silence", lambda path: None
    )
    monkeypatch.setattr(
        "buzz.localization.providers._probe_audio_duration", lambda path: 1.0
    )
    provider = EdgeTTSProvider(retry_delays=(0,))

    def missing_markers(_text, _voice, _options, combined):
        Path(combined).write_bytes(b"combined audio")
        return []

    monkeypatch.setattr(provider, "_stream_batch_with_attempt_timeout", missing_markers)
    requests = [
        TTSRequest("Xin chao", "vi", str(tmp_path / "one")),
        TTSRequest("Tam biet", "vi", str(tmp_path / "two")),
    ]
    from buzz.localization.tts import TTSBatchFallbackRequired

    with pytest.raises(TTSBatchFallbackRequired):
        provider.synthesize_batch(requests)
    # The orchestration layer then retries each request under its own TTS gate.
    results = [provider.synthesize(request) for request in requests]

    assert len(results) == 2
    assert len(calls) == 2
    assert all(Path(result.audio_file).is_file() for result in results)
    assert not list(tmp_path.glob(".edge-batch-*"))
    assert all((tmp_path / f"{name}.tts-cache.json").is_file()
               for name in ("one", "two"))


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


def test_edge_tts_cache_hit_does_not_pace(monkeypatch, tmp_path):
    _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.5)
    provider = EdgeTTSProvider()
    request = TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))
    pacing_calls = []
    monkeypatch.setattr(provider, "_pace_before_network_synthesis", lambda: pacing_calls.append(True))

    provider.synthesize(request)
    provider.synthesize(request)

    assert pacing_calls == [True]


def test_edge_tts_verified_cache_hit_bypasses_open_service_circuit(monkeypatch, tmp_path):
    calls, _, _ = _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    provider = EdgeTTSProvider(retry_delays=(0,))
    request = TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))

    provider.synthesize(request)
    calls.clear()
    monkeypatch.setattr(EdgeTTSProvider, "_service_circuit_until", time.monotonic() + 60)

    assert provider.synthesize(request).audio_file.endswith("segment.mp3")
    assert calls == []


def test_edge_tts_success_after_expired_circuit_clears_service_circuit(monkeypatch, tmp_path):
    _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    monkeypatch.setattr(EdgeTTSProvider, "_service_circuit_until", time.monotonic() - 1)

    EdgeTTSProvider(retry_delays=(0,)).synthesize(
        TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))
    )

    assert EdgeTTSProvider._service_circuit_until == 0.0


def test_edge_tts_successful_health_probe_clears_expired_service_circuit(monkeypatch):
    _install_fake_edge_tts(monkeypatch)
    provider = EdgeTTSProvider(retry_delays=(0,))
    monkeypatch.setattr(EdgeTTSProvider, "_service_circuit_until", time.monotonic() - 1)

    provider._health_probe(provider.default_voice, {})

    assert EdgeTTSProvider._service_circuit_until == 0.0


def test_edge_tts_batch_propagates_open_or_singleton_opened_service_circuit(monkeypatch, tmp_path):
    calls, NoAudioReceived, _ = _install_fake_edge_tts(monkeypatch)
    provider = EdgeTTSProvider(retry_delays=(0,))
    requests = [TTSRequest(f"text {index}", "vi", str(tmp_path / f"segment-{index}"))
                for index in range(2)]
    monkeypatch.setattr(EdgeTTSProvider, "_service_circuit_until", time.monotonic() + 60)

    with pytest.raises(TTSServiceTemporarilyUnavailable):
        provider.synthesize_batch(requests)
    assert calls == []

    monkeypatch.setattr(EdgeTTSProvider, "_service_circuit_until", 0.0)
    monkeypatch.setattr(provider, "_stream_batch_with_attempt_timeout",
                        lambda *_args: (_ for _ in ()).throw(NoAudioReceived("empty")))
    monkeypatch.setattr(provider, "_save_with_attempt_timeout",
                        lambda *_args: (_ for _ in ()).throw(NoAudioReceived("empty")))
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda _delay: None)

    with pytest.raises(TTSBatchFallbackRequired):
        provider.synthesize_batch(requests)


def test_edge_tts_legacy_retry_schedule_remains_available_for_no_audio_handling():
    assert EdgeTTSProvider.transient_retry_delays == (5, 10, 20, 40, 60, 90, 120)


def test_edge_tts_default_fresh_request_pacing_is_conservative():
    provider = EdgeTTSProvider()

    assert provider.pacing_min_seconds == 1.5
    assert provider.pacing_max_seconds == 2.5


def test_edge_tts_paces_successive_fresh_syntheses_deterministically(
    monkeypatch, tmp_path
):
    _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    monotonic_values = iter((100.0, 100.0, 100.0, 101.0))
    provider = EdgeTTSProvider()
    sleeps = []
    monkeypatch.setattr(provider, "_pacing_monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(provider, "_pacing_jitter_seconds", lambda: 1.0)
    monkeypatch.setattr(provider, "_pacing_sleep", sleeps.append)

    provider.synthesize(TTSRequest("Mot", "vi", str(tmp_path / "first")))
    provider.synthesize(TTSRequest("Hai", "vi", str(tmp_path / "second")))

    assert sleeps == [1.0]


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


def test_edge_tts_generic_transient_is_bounded_and_releases_for_workflow_retry(monkeypatch, tmp_path):
    def websocket_failure(_output_file):
        raise websocket_exception("Connection failed")
    calls, _, websocket_exception = _install_fake_edge_tts(monkeypatch, websocket_failure)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)

    with pytest.raises(TTSServiceTemporarilyUnavailable) as caught:
        EdgeTTSProvider().synthesize(TTSRequest("Xin chao", "vi", str(tmp_path / "segment")))

    assert caught.value.retry_after_seconds == 12
    assert len(calls) == 2
    assert sleeps == [1.0]
    assert not (tmp_path / "segment.mp3").exists()


def test_edge_tts_batch_generic_transient_is_bounded_and_releases_for_workflow_retry(
    monkeypatch, tmp_path
):
    def websocket_failure(_communicate):
        raise websocket_exception("Connection failed")

    calls, _, websocket_exception = _install_fake_edge_tts(
        monkeypatch, stream_behavior=websocket_failure
    )
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)

    with pytest.raises(TTSServiceTemporarilyUnavailable) as caught:
        EdgeTTSProvider(retry_delays=(120,)).synthesize_batch([
            TTSRequest("Mot", "vi", str(tmp_path / "one")),
            TTSRequest("Hai", "vi", str(tmp_path / "two")),
        ])

    assert caught.value.retry_after_seconds == 12
    assert len(calls) == 1
    assert sleeps == []


def test_edge_tts_attempt_timeout_retries_then_succeeds(monkeypatch, tmp_path):
    async def hang_then_succeed(output_file):
        if len(calls) == 1:
            Path(output_file).write_bytes(b"")
            await asyncio.Event().wait()
        Path(output_file).write_bytes(b"fake mp3")

    calls, _, _ = _install_fake_edge_tts(monkeypatch, hang_then_succeed)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.EDGE_TTS_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)

    result = EdgeTTSProvider(retry_delays=(0,)).synthesize(
        TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))
    )

    assert len(calls) == 2
    assert sleeps == [1.0]
    assert calls[0][2]["connect_timeout"] == 10
    assert calls[0][2]["receive_timeout"] == 60
    assert Path(result.audio_file).is_file()
    assert Path(result.audio_file).stat().st_size > 0


def test_edge_tts_attempt_timeouts_fail_finitely_and_remove_partial_output(
    monkeypatch, tmp_path, caplog
):
    async def hang(output_file):
        Path(output_file).write_bytes(b"partial")
        await asyncio.Event().wait()

    calls, _, _ = _install_fake_edge_tts(monkeypatch, hang)
    monkeypatch.setattr("buzz.localization.providers.EDGE_TTS_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda delay: None)

    with pytest.raises(TTSServiceTemporarilyUnavailable) as caught:
        EdgeTTSProvider(retry_delays=(0,)).synthesize(
            TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))
        )

    assert caught.value.retry_after_seconds == 12
    assert len(calls) == 2
    assert "TimeoutError: Edge TTS synthesis exceeded 0.01 seconds" in caplog.text
    assert not (tmp_path / "segment.mp3").exists()


def test_edge_tts_no_audio_with_proxifier_fails_early_after_two_attempts(
    monkeypatch, tmp_path
):
    def no_audio(output_file):
        raise no_audio_exception("No audio")

    calls, no_audio_exception, _ = _install_fake_edge_tts(monkeypatch, no_audio)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)
    monkeypatch.setattr("buzz.localization.providers._is_proxifier_running", lambda: True)

    with pytest.raises(LocalizationProviderError, match="Proxifier.*disable"):
        EdgeTTSProvider().synthesize(TTSRequest("Xin chao", "vi", str(tmp_path / "segment")))

    assert len(calls) == 2
    assert sleeps == [1.0]


def test_edge_tts_no_audio_without_proxifier_opens_service_circuit(monkeypatch, tmp_path):
    def no_audio(output_file):
        raise no_audio_exception("No audio")

    calls, no_audio_exception, _ = _install_fake_edge_tts(monkeypatch, no_audio)
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)
    monkeypatch.setattr("buzz.localization.providers._is_proxifier_running", lambda: False)

    with pytest.raises(TTSServiceTemporarilyUnavailable):
        EdgeTTSProvider().synthesize(TTSRequest("Xin chao", "vi", str(tmp_path / "segment")))

    assert len(calls) == 3  # two payload attempts, then one health probe
    assert sleeps == [1.0]


def test_edge_tts_no_audio_uses_stronger_retry_delay_and_writes_cache(
    monkeypatch, tmp_path
):
    attempts = 0

    def no_audio_then_succeed(output_file):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise no_audio_exception("No audio")
        Path(output_file).write_bytes(b"fake mp3")

    _, no_audio_exception, _ = _install_fake_edge_tts(
        monkeypatch, no_audio_then_succeed
    )
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)

    result = EdgeTTSProvider(retry_delays=(0,)).synthesize(
        TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))
    )

    assert sleeps == [1.0]
    assert Path(result.audio_file).is_file()
    assert (tmp_path / "segment.tts-cache.json").is_file()


def test_edge_tts_generic_transient_uses_normal_retry_delay(monkeypatch, tmp_path):
    attempts = 0

    def websocket_then_succeed(output_file):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise websocket_exception("Connection failed")
        Path(output_file).write_bytes(b"fake mp3")

    _, _, websocket_exception = _install_fake_edge_tts(
        monkeypatch, websocket_then_succeed
    )
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)

    EdgeTTSProvider(retry_delays=(3,)).synthesize(
        TTSRequest("Xin chao", "vi", str(tmp_path / "segment"))
    )

    assert sleeps == [1.0]


def test_edge_tts_websocket_error_does_not_check_proxifier(monkeypatch, tmp_path):
    def websocket_failure(output_file):
        raise websocket_exception("Connection failed")

    calls, _, websocket_exception = _install_fake_edge_tts(
        monkeypatch, websocket_failure
    )
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda delay: None)
    proxifier_checks = []
    monkeypatch.setattr(
        "buzz.localization.providers._is_proxifier_running",
        lambda: proxifier_checks.append(True) or True,
    )

    with pytest.raises(TTSServiceTemporarilyUnavailable):
        EdgeTTSProvider(retry_delays=(0,)).synthesize(TTSRequest(
            "Xin chao", "vi", str(tmp_path / "segment")
        ))

    assert len(calls) == 2
    assert proxifier_checks == []


def test_edge_tts_logs_sanitized_diagnostics_and_final_failure(
    monkeypatch, tmp_path, caplog
):
    def websocket_failure(output_file):
        error = websocket_exception(
            "HTTP 429 timeout token=super-secret "
            "wss://example.test/path?api_key=also-secret"
        )
        error.status_code = 429
        raise error

    _, _, websocket_exception = _install_fake_edge_tts(
        monkeypatch, websocket_failure
    )
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda delay: None)
    provider = EdgeTTSProvider(retry_delays=(0,))

    with caplog.at_level("WARNING", logger="buzz.localization.providers"):
        with pytest.raises(TTSServiceTemporarilyUnavailable) as caught:
            provider.synthesize(TTSRequest(
                "Xin chao", "vi", str(tmp_path / "diagnostic")
            ))

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "WebSocketError" in messages
    assert "status=429" in messages
    assert "HTTP 429 timeout" in messages
    assert "transient failure (1/2)" in messages
    assert "exhausted short retry budget" in messages
    assert "super-secret" not in messages
    assert "also-secret" not in messages
    assert "super-secret" not in str(caught.value)


def test_edge_tts_honors_retry_after(monkeypatch, tmp_path):
    attempts = 0

    def rate_limited_once(output_file):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = websocket_exception("HTTP 429 Too Many Requests")
            error.response = type("Response", (), {
                "status_code": 429, "headers": {"Retry-After": "12"}
            })()
            raise error
        Path(output_file).write_bytes(b"fake mp3")

    _, _, websocket_exception = _install_fake_edge_tts(
        monkeypatch, rate_limited_once
    )
    sleeps = []
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)

    EdgeTTSProvider(retry_delays=(1,)).synthesize(TTSRequest(
        "Xin chao", "vi", str(tmp_path / "rate-limited")
    ))

    assert sleeps == [1.0]


def test_edge_tts_conservative_default_serializes_concurrent_calls(
    monkeypatch, tmp_path
):
    active = 0
    peak_active = 0
    state_lock = threading.Lock()

    def slow_save(output_file):
        nonlocal active, peak_active
        with state_lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        Path(output_file).write_bytes(b"fake mp3")
        with state_lock:
            active -= 1

    _install_fake_edge_tts(monkeypatch, slow_save)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    provider = EdgeTTSProvider()
    errors = []

    def synthesize(index):
        try:
            provider.synthesize(TTSRequest(
                "Xin chao", "vi", str(tmp_path / f"concurrent-{index}")
            ))
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=synthesize, args=(index,)) for index in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert errors == []
    assert peak_active == 1


def test_edge_tts_child_certificate_error_retries_then_succeeds(monkeypatch, tmp_path):
    _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    attempts = []

    def run_attempt(text, voice, options, output_file, **kwargs):
        attempts.append(output_file)
        if len(attempts) == 1:
            raise providers._EdgeTTSAttemptError(
                "ClientConnectorCertificateError", "temporary self-signed certificate"
            )
        Path(output_file).write_bytes(b"audio")
        return []

    monkeypatch.setattr(providers, "_run_edge_tts_attempt", run_attempt)
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda delay: None)
    result = EdgeTTSProvider(retry_delays=(0,)).synthesize(
        TTSRequest("Xin chao", "vi", str(tmp_path / "cert-retry"))
    )
    assert len(attempts) == 2
    assert Path(result.audio_file).is_file()


def test_edge_tts_batch_certificate_error_retries_then_succeeds(monkeypatch, tmp_path):
    _install_fake_edge_tts(monkeypatch)
    monkeypatch.setattr("buzz.localization.providers._trim_edge_tts_silence", lambda path: None)
    monkeypatch.setattr("buzz.localization.providers._probe_audio_duration", lambda path: 1.0)
    provider = EdgeTTSProvider(retry_delays=(0,))
    attempts = []

    def stream_batch(text, voice, options, combined):
        attempts.append(str(combined))
        if len(attempts) == 1:
            raise providers._EdgeTTSAttemptError(
                "ClientConnectorCertificateError", "temporary self-signed certificate"
            )
        Path(combined).write_bytes(b"combined")
        return []

    monkeypatch.setattr(provider, "_stream_batch_with_attempt_timeout", stream_batch)
    monkeypatch.setattr(provider, "_batch_intervals_from_boundaries",
                        lambda markers, boundaries, duration: [(0.0, 0.5), (0.5, 1.0)])
    monkeypatch.setattr(provider, "_split_batch_audio",
                        lambda combined, output, start, end: Path(output).write_bytes(b"audio"))
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda delay: None)
    requests = [
        TTSRequest("Mot", "vi", str(tmp_path / "one")),
        TTSRequest("Hai", "vi", str(tmp_path / "two")),
    ]
    # A transient batch failure must release its worker slot immediately.
    # The higher-level workflow can retry the batch later.
    with pytest.raises(TTSServiceTemporarilyUnavailable):
        provider.synthesize_batch(requests)
    assert len(attempts) == 1
    results = provider.synthesize_batch(requests)
    assert len(attempts) == 2
    assert len(results) == 2


def test_edge_tts_child_non_transient_error_fails_without_retry(monkeypatch, tmp_path):
    _install_fake_edge_tts(monkeypatch)
    sleeps = []
    attempts = []

    def fail_attempt(*args, **kwargs):
        attempts.append((args, kwargs))
        raise providers._EdgeTTSAttemptError("ValueError", "invalid parameter")

    monkeypatch.setattr(providers, "_run_edge_tts_attempt", fail_attempt)
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleeps.append)

    with pytest.raises(LocalizationProviderError, match="invalid parameter"):
        EdgeTTSProvider().synthesize(
            TTSRequest("Xin chao", "vi", str(tmp_path / "child-non-transient"))
        )

    assert len(attempts) == 1
    assert sleeps == []


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


def test_gemini_shortens_one_translation_for_timing_recovery():
    from buzz.localization.providers import GeminiTranslationProvider

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    captured = {}

    class Response:
        text = "Cau ngan, du nghia"

    def generate(**kwargs):
        captured.update(kwargs)
        return Response()

    provider._generate_content_with_retry = generate
    result = provider.shorten_translation(
        source_text="Keep every important detail",
        translated_text="Ban dich hien tai qua dai",
        source_language="en", target_language="vi",
        required_rate=2.083, max_playback_rate=2.0,
    )

    assert result == "Cau ngan, du nghia"
    assert "Compress wording, never facts" in captured["contents"]
    assert "96%" in captured["contents"]


def test_gemini_batch_shortens_in_one_json_request_and_validates_ids():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["contents"].split("Input JSON:\n", 1)[1])
        return SimpleNamespace(text=json.dumps({"segments": [
            {"id": payload["segments"][1]["id"], "shortened_text": "ngan hai"},
            {"id": payload["segments"][0]["id"], "shortened_text": "ngan mot"},
        ]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate
    segments = [
        {"id": "a", "source_text": "one", "translated_text": "mot dai",
         "required_rate": 1.5, "max_playback_rate": 1.1},
        {"id": "b", "source_text": "two", "translated_text": "hai dai",
         "required_rate": 1.4, "max_playback_rate": 1.1},
    ]
    assert provider.shorten_translations(
        segments=segments, source_language="en", target_language="vi"
    ) == ["ngan mot", "ngan hai"]
    assert len(calls) == 1
    assert calls[0]["config"]["response_mime_type"] == "application/json"
    schema = calls[0]["config"]["response_json_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["segments"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["segments"]["minItems"] == len(segments)
    assert schema["properties"]["segments"]["maxItems"] == len(segments)
    item_schema = schema["properties"]["segments"]["items"]
    assert item_schema["type"] == "object"
    assert item_schema["required"] == ["id", "shortened_text"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["id"] == {"type": "string"}
    assert item_schema["properties"]["shortened_text"] == {"type": "string"}

    provider._generate_content_with_retry = lambda **kwargs: SimpleNamespace(
        text='{"segments":[{"id":"a","shortened_text":"x y"}]}'
    )
    # The bounded batch recovery keeps verified originals on malformed IDs
    # rather than looping or inventing a potentially incomplete shortened line.
    assert provider.shorten_translations(
        segments=segments, source_language="en", target_language="vi"
    ) == ["x y", "hai dai"]


def test_gemini_batch_shortening_retries_when_word_count_exceeds_hard_max():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(text='{"segments":[{"id":"a","shortened_text":"mot hai ba"}]}')
        return SimpleNamespace(text='{"segments":[{"id":"a","shortened_text":"mot hai"}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate
    result = provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    )

    assert result == ["mot hai"]
    assert len(calls) == 2
    assert "between its minimum_word_count and target_word_count" in calls[1]["contents"]
    payload = json.loads(calls[0]["contents"].split("Input JSON:\n", 1)[1])
    assert payload["segments"][0]["target_word_count"] == 2


def test_gemini_batch_shortening_retries_when_p01_like_candidate_is_too_short():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    original = (
        "xin vui long doc truoc noi quy ky tuc xa quy tac mot can ho nay "
        "duoc bo tri phong sau nguoi"
    )
    too_short = "Hay doc noi quy Dieu mot Phong sau nguoi"
    fitting = "Hay doc noi quy Dieu mot Day la phong sau nguoi"
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        candidate = too_short if len(calls) == 1 else fitting
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "p01-5", "shortened_text": candidate,
        }]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "p01-5", "source_text": "Please read the dorm rules first.",
                   "translated_text": original, "required_rate": 1.4,
                   "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == [fitting]
    assert len(calls) == 2
    payload = json.loads(calls[0]["contents"].split("Input JSON:\n", 1)[1])
    assert payload["segments"][0]["target_word_count"] == 17
    assert payload["segments"][0]["minimum_word_count"] == 11
    assert "between its minimum_word_count and target_word_count" in calls[1]["contents"]


def test_gemini_batch_shortening_uses_original_when_candidate_is_below_bounds():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "a", "shortened_text": "mot",
        }]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2


def test_gemini_batch_shortening_uses_timing_proportional_budget_for_mild_overflow():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    original = " ".join(f"tu{index}" for index in range(35))
    candidate = " ".join(f"ngan{index}" for index in range(32))
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "segment-000000", "shortened_text": candidate,
        }]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{
            "id": "segment-000000", "source_text": "source",
            "translated_text": original, "required_rate": 1.193,
            "max_playback_rate": 1.10,
        }], source_language="en", target_language="vi",
    ) == [candidate]

    payload = json.loads(calls[0]["contents"].split("Input JSON:\n", 1)[1])
    assert payload["segments"][0]["target_word_count"] == 32
    assert len(calls) == 1



def test_gemini_batch_shortening_small_overflow_forces_one_word_reduction():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "segment-000000", "shortened_text": "mot hai ba bon",
        }]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{
            "id": "segment-000000", "source_text": "source",
            "translated_text": "mot hai ba bon nam", "required_rate": 1.12,
            "max_playback_rate": 1.10,
        }], source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]

    payload = json.loads(calls[0]["contents"].split("Input JSON:\n", 1)[1])
    assert payload["segments"][0]["target_word_count"] == 4
    assert len(calls) == 1

def test_gemini_batch_shortening_retries_non_speakable_result():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        text = "..." if len(calls) == 1 else "cau ngan"
        return SimpleNamespace(text=json.dumps({"segments": [
            {"id": "a", "shortened_text": text},
        ]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["cau ngan"]
    assert len(calls) == 2
    assert "between its minimum_word_count and target_word_count" in calls[1]["contents"]


def test_gemini_batch_shortening_uses_original_after_non_speakable_structured_output():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"segments":[{"id":"a","shortened_text":"..."}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2


def test_gemini_batch_shortening_uses_original_after_invalid_structured_bounds():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"segments":[{"id":"a","shortened_text":"mot hai ba"}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2


def test_gemini_batch_shortening_does_not_fan_out_overlong_candidates_to_singletons():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"segments":[{"id":"a","shortened_text":"mot hai ba"}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2


def test_gemini_batch_shortening_uses_original_not_unverified_overlong_candidate():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"segments":[{"id":"a","shortened_text":"mot hai ba"}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2


def test_gemini_batch_shortening_never_uses_too_short_candidate():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            return SimpleNamespace(text='{ "segments": [{"id":"a","shortened_text":"mot"}]}')
        return SimpleNamespace(text="mot")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2


def test_gemini_batch_shortening_preserves_id_alignment_after_id_errors():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"segments":[{"id":"wrong","shortened_text":"mot hai"}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2


def test_gemini_batch_shortening_uses_original_after_two_malformed_json_responses():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="not valid JSON")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate

    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "one", "translated_text": "mot hai ba bon",
                   "required_rate": 2.0, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["mot hai ba bon"]
    assert len(calls) == 2
    assert all("word count between" not in call["contents"] for call in calls[:2])


def test_gemini_batch_shortening_rate_limit_uses_originals_without_cooldown_loop():
    from buzz.localization.providers import GeminiTranslationProvider, LocalizationProviderError

    calls = []

    def rate_limited(**kwargs):
        calls.append(kwargs)
        raise LocalizationProviderError("Gemini rate-limit cooldown budget exhausted")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = rate_limited
    segments = [
        {"id": "first", "source_text": "one", "translated_text": "mot hai ba bon",
         "required_rate": 2.0, "max_playback_rate": 1.1},
        {"id": "second", "source_text": "two", "translated_text": "nam sau bay tam",
         "required_rate": 2.0, "max_playback_rate": 1.1},
    ]

    assert provider.shorten_translations(
        segments=segments, source_language="en", target_language="vi",
    ) == ["mot hai ba bon", "nam sau bay tam"]
    assert len(calls) == 1
    assert calls[0]["max_rate_limit_cooldown_rounds"] == 0


def test_gemini_batch_shortening_retries_only_missing_segment_and_preserves_order():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["contents"].split("Input JSON:\n", 1)[1])
        identifiers = [item["id"] for item in payload["segments"]]
        if len(calls) == 1:
            assert identifiers == ["a", "b", "c"]
            return SimpleNamespace(text=json.dumps({"segments": [
                {"id": "a", "shortened_text": "mot a"},
                {"id": "c", "shortened_text": "mot c"},
            ]}))
        assert identifiers == ["b"]
        return SimpleNamespace(text='{"segments":[{"id":"b","shortened_text":"mot b"}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate
    segments = [
        {"id": identifier, "source_text": identifier,
         "translated_text": "mot hai ba bon", "required_rate": 2.0,
         "max_playback_rate": 1.1}
        for identifier in ("a", "b", "c")
    ]

    assert provider.shorten_translations(
        segments=segments, source_language="en", target_language="vi",
    ) == ["mot a", "mot b", "mot c"]
    assert len(calls) == 2
    assert calls[0]["config"]["response_json_schema"]["required"] == ["segments"]
    assert calls[1]["config"]["response_mime_type"] == "application/json"


def test_gemini_batch_shortening_retries_only_invalid_ids_and_preserves_valid_results():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["contents"].split("Input JSON:\n", 1)[1])
        identifiers = [item["id"] for item in payload["segments"]]
        if len(calls) == 1:
            return SimpleNamespace(text=json.dumps({"segments": [
                {"id": "a", "shortened_text": "mot a"},
                {"id": "b", "shortened_text": "mot b"},
                {"id": "c", "shortened_text": "mot c dai"},
                {"id": "d", "shortened_text": "mot d"},
            ]}))
        assert identifiers == ["c"]
        return SimpleNamespace(text='{"segments":[{"id":"c","shortened_text":"mot c"}]}')

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate
    segments = [
        {"id": identifier, "source_text": identifier, "translated_text": "mot hai ba bon",
         "required_rate": 2.0, "max_playback_rate": 1.1}
        for identifier in ("a", "b", "c", "d")
    ]

    assert provider.shorten_translations(
        segments=segments, source_language="en", target_language="vi",
    ) == ["mot a", "mot b", "mot c", "mot d"]
    assert len(calls) == 2
    assert calls[1]["config"]["response_json_schema"]["properties"]["segments"]["minItems"] == 1


def test_gemini_batch_shortening_salvages_valid_ids_amid_missing_duplicate_and_unexpected_ids():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(text=json.dumps({"segments": [
                {"id": "a", "shortened_text": "mot a"},
                {"id": "b", "shortened_text": "mot b"},
                {"id": "b", "shortened_text": "hai b"},
                {"id": "d", "shortened_text": "mot d"},
                {"id": "unexpected", "shortened_text": "mot la"},
            ]}))
        payload = json.loads(kwargs["contents"].split("Input JSON:\n", 1)[1])
        assert [item["id"] for item in payload["segments"]] == ["b", "c"]
        return SimpleNamespace(text=json.dumps({"segments": [
            {"id": "b", "shortened_text": "mot b"},
            {"id": "c", "shortened_text": "mot c"},
        ]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate
    segments = [
        {"id": identifier, "source_text": identifier, "translated_text": "mot hai ba bon",
         "required_rate": 2.0, "max_playback_rate": 1.1}
        for identifier in ("a", "b", "c", "d")
    ]

    assert provider.shorten_translations(
        segments=segments, source_language="en", target_language="vi",
    ) == ["mot a", "mot b", "mot c", "mot d"]
    assert len(calls) == 2


def test_gemini_batch_shortening_uses_originals_for_unresolved_ids_without_singletons():
    import json
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["contents"].split("Input JSON:\n", 1)[1])
        identifiers = [item["id"] for item in payload["segments"]]
        if len(calls) == 1:
            return SimpleNamespace(text=json.dumps({"segments": [
                {"id": "a", "shortened_text": "mot a"},
                {"id": "b", "shortened_text": "mot b dai"},
                {"id": "c", "shortened_text": "mot c"},
                {"id": "d", "shortened_text": "mot d dai"},
            ]}))
        assert identifiers == ["b", "d"]
        return SimpleNamespace(text=json.dumps({"segments": [
            {"id": "b", "shortened_text": "mot b dai"},
            {"id": "d", "shortened_text": "mot d dai"},
        ]}))

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = generate
    segments = [
        {"id": identifier, "source_text": identifier, "translated_text": "mot hai ba bon",
         "required_rate": 2.0, "max_playback_rate": 1.1}
        for identifier in ("a", "b", "c", "d")
    ]

    assert provider.shorten_translations(
        segments=segments, source_language="en", target_language="vi",
    ) == ["mot a", "mot hai ba bon", "mot c", "mot hai ba bon"]
    assert len(calls) == 2
    assert ["a", "b", "c", "d"] == [item["id"] for item in json.loads(
        calls[0]["contents"].split("Input JSON:\n", 1)[1]
    )["segments"]]


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
                    {"id": "1", "corrected_text": "自动回充", "translated_text": "Tự về sạc"},
                    {"id": "0", "corrected_text": "吸力很强", "translated_text": "Lực hút mạnh"},
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


def test_gemini_batch_context_includes_translation_and_register_policy():
    from buzz.localization.providers import GeminiTranslationProvider

    captured = {}
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)

    def generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "0", "corrected_text": "source", "translated_text": "ban dich",
        }]}))

    provider._generate_content_with_retry = generate
    result = provider.translate_segments(
        [SimpleNamespace(start=0, end=1, text="source")], "en", "vi",
        context=[SimpleNamespace(
            start=-1, end=0, text="corrected context", translated_text="da dich truoc",
        )],
    )

    payload = json.loads(captured["contents"].split("Input JSON:\n", 1)[1])
    assert payload["context_only"][0]["text"] == "corrected context"
    assert payload["context_only"][0]["translated_text"] == "da dich truoc"
    assert result == [("source", "ban dich")]
    assert "authoritative style and register memory" in captured["contents"]
    assert "NEVER alternate" in captured["contents"]
    assert "never by awkward literal calques" in captured["contents"]


def test_gemini_register_review_payload_and_hard_rules():
    from buzz.localization.providers import GeminiTranslationProvider

    captured = {}
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)

    def generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "segment-1", "revised_translated_text": "ta sẽ giết người",
        }]}))

    provider._generate_content_with_retry = generate
    result = provider.review_translation_register(
        [SimpleNamespace(segment_id="segment-1", text="源", translated_text="tao sẽ giết người")],
        "zh", "vi", context=[SimpleNamespace(
            segment_id="previous", text="trước", translated_text="ta không nhường người",
        )],
    )
    payload = json.loads(captured["contents"].split("Input JSON:\n", 1)[1])
    assert payload["segments"][0] == {
        "id": "segment-1", "source_text": "源", "current_vietnamese": "tao sẽ giết người",
    }
    assert payload["context_only"][0]["reviewed_vietnamese"] == "ta không nhường người"
    assert result == [("segment-1", "ta sẽ giết người")]
    assert "ta-người" in captured["contents"]
    assert "坑爹" in captured["contents"]
    assert "hố cha" in captured["contents"]


def test_gemini_register_review_preserves_partial_and_invalid_items_for_recovery():
    from buzz.localization.providers import GeminiTranslationProvider

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = lambda **kwargs: SimpleNamespace(
        text=json.dumps({"segments": [
            {"id": "segment-1", "revised_translated_text": "ta sẽ giúp người"},
            {"id": "segment-2", "revised_translated_text": ""},
            {"id": "unexpected", "revised_translated_text": "not requested"},
            {"revised_translated_text": "missing ID"},
        ]})
    )
    items = [
        SimpleNamespace(segment_id="segment-1", text="源一", translated_text="ta giúp người"),
        SimpleNamespace(segment_id="segment-2", text="源二", translated_text="tôi giúp cậu"),
    ]
    assert provider.review_translation_register(items, "zh", "vi") == [
        ("segment-1", "ta sẽ giúp người"),
        ("segment-2", ""),
        ("unexpected", "not requested"),
        (None, "missing ID"),
    ]


def test_gemini_register_review_bad_json_is_retryable_but_api_error_is_not():
    from buzz.localization.providers import GeminiTranslationProvider

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    item = SimpleNamespace(segment_id="segment-1", text="源", translated_text="bản dịch")
    provider._generate_content_with_retry = lambda **kwargs: SimpleNamespace(text="{bad-json")
    assert provider.review_translation_register([item], "zh", "vi") == []

    def failed_request(**kwargs):
        raise ConnectionError("service unavailable")

    provider._generate_content_with_retry = failed_request
    with pytest.raises(LocalizationProviderError, match="service unavailable"):
        provider.review_translation_register([item], "zh", "vi")


def test_gemini_shortening_prompts_preserve_register():
    from buzz.localization.providers import GeminiTranslationProvider

    captured = []
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = lambda **kwargs: (
        captured.append(kwargs["contents"]) or SimpleNamespace(text="ta khong biet")
    )
    provider.shorten_translation(
        source_text="I do not know", translated_text="ta khong biet", source_language="en",
        target_language="vi", required_rate=2, max_playback_rate=1.1,
    )
    assert "Preserve the current Vietnamese pronouns" in captured[0]
    assert "never switch pronoun pair or register" in captured[0]


def test_gemini_batch_shortening_prompt_preserves_register():
    from buzz.localization.providers import GeminiTranslationProvider

    captured = []
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)

    def generate(**kwargs):
        captured.append(kwargs["contents"])
        if len(captured) <= 2:
            return SimpleNamespace(text='{"segments":[{"id":"a","shortened_text":"x"}]}')
        return SimpleNamespace(text="ta khong biet")

    provider._generate_content_with_retry = generate
    assert provider.shorten_translations(
        segments=[{"id": "a", "source_text": "I do not know",
                   "translated_text": "ta that su khong biet gi ca",
                   "required_rate": 2, "max_playback_rate": 1.1}],
        source_language="en", target_language="vi",
    ) == ["ta that su khong biet gi ca"]
    assert "Preserve the current Vietnamese pronouns" in captured[0]
    assert "Preserve the current Vietnamese pronouns" in captured[1]


def test_gemini_batch_rejects_missing_id():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider
    from buzz.localization.transcript import LocalizationSegment

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs:
            SimpleNamespace(text='{"segments":[{"id":"9","corrected_text":"a","translated_text":"b"}]}'))
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


def test_gemini_retries_and_uses_configured_fallback_on_429(monkeypatch):
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(text="Xin chào")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model = "gemini-3.8-flash"
    provider.rate_limit_fallback_models = ("gemini-3.5-flash-lite",)
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr(
        "buzz.localization.providers.time.sleep",
        lambda seconds: None,
    )

    assert provider.translate("Hello", "en", "vi") == "Xin chào"
    assert calls == [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ]
    assert provider.last_model_used == "gemini-3.5-flash-lite"


def test_gemini_skips_unavailable_model_and_remembers_it_for_later_requests():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "obsolete":
                raise RuntimeError(
                    "404 NOT_FOUND model models/obsolete is no longer available"
                )
            return SimpleNamespace(text="Xin chao")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "obsolete", ("fallback",)
    provider._client = SimpleNamespace(models=Models())

    assert provider.translate("Hello", "en", "vi") == "Xin chao"
    assert provider.translate("Again", "en", "vi") == "Xin chao"
    assert calls == ["obsolete", "fallback", "fallback"]
    assert provider._unavailable_models == {"obsolete"}


def test_gemini_uses_configured_fallback_after_retryable_503(monkeypatch):
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise RuntimeError("503 UNAVAILABLE")
            return SimpleNamespace(text="Xin chao")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model = "gemini-3.8-flash"
    provider.rate_limit_fallback_models = ("gemini-3.5-flash-lite",)
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda seconds: None)

    assert provider.translate("Hello", "en", "vi") == "Xin chao"
    assert calls == ["gemini-3.8-flash"] * 3 + ["gemini-3.5-flash-lite"]


def test_gemini_uses_configured_fallback_after_retryable_504_deadline_exceeded(monkeypatch):
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise RuntimeError("504 DEADLINE_EXCEEDED")
            return SimpleNamespace(text="Xin chao")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model = "gemini-3.8-flash"
    provider.rate_limit_fallback_models = ("gemini-3.5-flash-lite",)
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda seconds: None)

    assert provider.translate("Hello", "en", "vi") == "Xin chao"
    assert calls == ["gemini-3.8-flash"] * 3 + ["gemini-3.5-flash-lite"]


@pytest.mark.parametrize("message", (
    "Server disconnected without sending a response.",
    "httpx.RemoteProtocolError: peer closed connection without sending complete message body",
    "Connection reset by peer",
    "Broken pipe",
    "unexpected EOF while reading",
    "network error while reading response",
))
def test_gemini_transport_failures_are_transient(message):
    from buzz.localization.providers import GeminiTranslationProvider

    assert GeminiTranslationProvider._is_gemini_transient_error(RuntimeError(message))


@pytest.mark.parametrize("error", (
    socket.gaierror(11001, "getaddrinfo failed"),
    RuntimeError("[Errno 11001] getaddrinfo failed"),
    RuntimeError("Name or service not known"),
    RuntimeError("Temporary failure in name resolution"),
))
def test_gemini_dns_failures_are_transient(error):
    from buzz.localization.providers import GeminiTranslationProvider

    assert GeminiTranslationProvider._is_gemini_transient_error(error)


def test_gemini_dns_failure_retries_selected_model_without_fallback(monkeypatch):
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) < 3:
                raise OSError(11001, "getaddrinfo failed")
            return SimpleNamespace(text="Xin chao")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "primary", ("fallback",)
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda seconds: None)

    assert provider._generate_content_with_retry(contents="x").text == "Xin chao"
    assert calls == ["primary"] * 3


@pytest.mark.parametrize("message", (
    "400 INVALID_ARGUMENT",
    "API_KEY_INVALID",
    "ValueError: translated text cannot be empty",
))
def test_gemini_validation_failures_are_not_transient(message):
    from buzz.localization.providers import GeminiTranslationProvider

    assert not GeminiTranslationProvider._is_gemini_transient_error(ValueError(message))


def test_gemini_server_disconnect_retries_then_uses_fallback(monkeypatch):
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary":
                raise RuntimeError("Server disconnected without sending a response.")
            return SimpleNamespace(text="Xin chao")

    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "primary", ("fallback",)
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr("buzz.localization.providers.time.sleep", lambda seconds: None)

    assert provider._generate_content_with_retry(contents="x").text == "Xin chao"
    assert calls == ["primary"] * 3 + ["fallback"]
    assert provider.last_model_used == "fallback"


def test_gemini_default_fallback_order_dedupes_primary(monkeypatch):
    from buzz.localization.providers import GeminiTranslationProvider

    monkeypatch.delenv("BUZZ_GEMINI_429_FALLBACK_MODELS", raising=False)
    provider = GeminiTranslationProvider(api_key="test-key", model="gemini-3.5-flash-lite")
    assert provider._model_candidates() == (
        "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash",
        "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
        "gemini-3-flash-preview",
    )


def test_gemini_rpd_model_is_skipped_on_later_requests():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary":
                raise RuntimeError("429 RESOURCE_EXHAUSTED requests_per_day")
            return SimpleNamespace(text="ok")
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "primary", ("fallback",)
    provider._client = SimpleNamespace(models=Models())
    assert provider.translate("one", "en", "vi") == "ok"
    assert provider.translate("two", "en", "vi") == "ok"
    assert calls == ["primary", "fallback", "fallback"]


def test_gemini_rpm_falls_back_then_reenables_after_cooldown(monkeypatch):
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    now, calls = [0.0], []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary" and calls.count("primary") == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED requests_per_minute")
            return SimpleNamespace(text="ok")
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "primary", ("fallback",)
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr("buzz.localization.providers.time.monotonic", lambda: now[0])
    assert provider.translate("one", "en", "vi") == "ok"
    now[0] = 66.0
    assert provider.translate("two", "en", "vi") == "ok"
    assert calls == ["primary", "fallback", "primary"]


def test_gemini_ambiguous_429_uses_cooldown():
    from buzz.localization.providers import GeminiTranslationProvider

    assert GeminiTranslationProvider._gemini_quota_kind(RuntimeError("429 quota exceeded")) == "rpm"


def test_gemini_all_rpm_models_wait_and_retry(monkeypatch):
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    now, calls, sleeps = [0.0], [], []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) <= len(provider._model_candidates()):
                raise RuntimeError("429 RESOURCE_EXHAUSTED RPM")
            return SimpleNamespace(text="ok")
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "primary", ("fallback",)
    provider._client = SimpleNamespace(models=Models())
    monkeypatch.setattr("buzz.localization.providers.time.monotonic", lambda: now[0])
    monkeypatch.setattr("buzz.localization.providers.time.sleep", sleep)
    assert provider.translate("one", "en", "vi") == "ok"
    assert calls == [*provider._model_candidates(), "primary"]
    assert sleeps == [65.0]


def test_gemini_all_rpd_models_raise_clear_error():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider, LocalizationProviderError

    def fail(**kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED per day")
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "primary", ("fallback",)
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=fail))
    with pytest.raises(LocalizationProviderError, match="daily quota is exhausted"):
        provider.translate("one", "en", "vi")


def test_gemini_non_retryable_error_fails_immediately():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider

    calls = []
    def fail(**kwargs):
        calls.append(kwargs["model"])
        raise RuntimeError("400 INVALID_ARGUMENT API_KEY_INVALID")
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider.model, provider.rate_limit_fallback_models = "primary", ("fallback",)
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=fail))
    with pytest.raises(Exception, match="INVALID_ARGUMENT"):
        provider._generate_content_with_retry(contents="x")
    assert calls == ["primary"]


def test_gemini_v4_prompt_identifies_speakers_and_forbids_vulgar_pronouns():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider
    captured = {}
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)

    def generate(**kwargs):
        captured["contents"] = kwargs["contents"]
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "scene-1",
            "corrected_text": "你答应给我十块钱",
            "translated_text": "Anh đã hứa đưa em mười đồng.",
        }]}))

    provider._generate_content_with_retry = generate
    item = SimpleNamespace(segment_id="scene-1", start=0, end=2,
                           text="你答应给我十块钱")
    result = provider.translate_segments([item], "zh", "vi")
    assert result[0][1] == "Anh đã hứa đưa em mười đồng."
    prompt = captured["contents"]
    assert "who addresses whom" in prompt
    assert "romantic couple" in prompt
    assert "Never use the Vietnamese pronouns 'tao' or 'mày'" in prompt
    assert "NO missing" in prompt
    assert "source utterance" in prompt
    assert "Input JSON:\n" in prompt
    assert provider.translation_policy_version.endswith("no-tao-may-complete-concise")
    assert "3.5 to 4.2" in prompt
    assert "two or more speakers" in prompt
    assert "years, months, dates" in prompt


def test_gemini_v4_batch_rejects_prohibited_dialogue_without_losing_ids():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider, LocalizationProviderError
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = lambda **kwargs: SimpleNamespace(
        text=json.dumps({"segments": [{
            "id": "line-id", "corrected_text": "你", "translated_text": "Mày đi đi",
        }]}))
    with pytest.raises(LocalizationProviderError, match="prohibited Vietnamese pronouns"):
        provider.translate_segments([
            SimpleNamespace(segment_id="line-id", start=0, end=1, text="你")
        ], "zh", "vi")


def test_gemini_v4_review_repairs_incomplete_dialogue_without_hiding_context():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    captured = {}

    def generate(**kwargs):
        captured["prompt"] = kwargs["contents"]
        return SimpleNamespace(text=json.dumps({"segments": [{
            "id": "one", "revised_translated_text": "Anh đã hứa đưa em mười đồng.",
        }]}))

    provider._generate_content_with_retry = generate
    result = provider.review_translation_register([
        SimpleNamespace(segment_id="one", text="你答应给我十块钱",
                        translated_text="Anh hứa rồi.")
    ], "zh", "vi", context=[])
    assert result == [("one", "Anh đã hứa đưa em mười đồng.")]
    assert "source_text as ground truth" in captured["prompt"]
    assert "missing or mangled meaningful words" in captured["prompt"]
    assert "Input JSON:\n" in captured["prompt"]


def test_gemini_v4_single_translation_rejects_tao_may():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider, LocalizationProviderError
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = lambda **kwargs: SimpleNamespace(
        text="Tao đã bảo mày rồi.")
    with pytest.raises(LocalizationProviderError, match="prohibited Vietnamese pronouns"):
        provider.translate("我都说过了", "zh", "vi")


def test_gemini_v4_shortening_rejects_tao_may():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider, LocalizationProviderError
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry = lambda **kwargs: SimpleNamespace(
        text="Tao nói rồi")
    with pytest.raises(LocalizationProviderError, match="prohibited Vietnamese pronouns"):
        provider.shorten_translation(
            source_text="我都说过了", translated_text="Tôi đã nói rồi.",
            source_language="zh", target_language="vi", required_rate=2.2,
            max_playback_rate=2.0,
        )


def test_gemini_dialogue_qa_prompt_handles_neighbors_and_timing_brevity():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    seen = {}

    def generate(**kwargs):
        seen["prompt"] = kwargs["contents"]
        seen["mime"] = kwargs["config"]["response_mime_type"]
        return SimpleNamespace(text=json.dumps({"segments": [
            {"id": "line-000000", "revised_vietnamese": "Anh về ngay đi."}
        ]}))

    provider._generate_content_with_retry = generate
    results = provider.review_dialogue_alignment(
        segments=({"id": "line-000000", "source_text": "你马上回来",
                   "start":0,"end":1.5,"available_seconds":1.5,
                   "current_vietnamese":"Anh phải về nhà ngay lập tức!",
                   "needs_fidelity_or_timing_review": True},),
        context=({"id":"neighbor","source_text":"我会等你",
                  "current_vietnamese":"Em sẽ đợi anh."},),
        source_language="zh",target_language="vi",
    )
    assert results == [("line-000000", "Anh về ngay đi.")]
    assert "NEVER move an adjacent sentence" in seen["prompt"]
    assert "NEVER omit a source" in seen["prompt"]
    assert "context_only" in seen["prompt"]
    assert "5.2" in seen["prompt"]
    assert seen["mime"]=="application/json"


def test_gemini_dialogue_qa_returns_empty_on_bad_json():
    from types import SimpleNamespace
    from buzz.localization.providers import GeminiTranslationProvider
    provider = GeminiTranslationProvider.__new__(GeminiTranslationProvider)
    provider._generate_content_with_retry=lambda **kw:SimpleNamespace(text="{}")
    assert provider.review_dialogue_alignment(
        segments=({"id":"a","source_text":"你好",
            "current_vietnamese":"Xin chào"},),
        source_language="zh",target_language="vi",
    )==[]
