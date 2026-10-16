import json
from pathlib import Path
import threading

import pytest
from buzz.localization.providers import VieNeuTTSProvider
from buzz.localization.tts import TTSRequest, _request_fingerprint
from buzz.localization.voices import (
    TTS_VOICE_CHOICES,
    VIENEU_VOICE_CHOICES,
    parse_tts_voice,
)


EXPECTED_VIENEU_VOICES = (
    "Adam bựa",
    "Trúc Ly",
    "Thiện Minh",
    "Mai Anh",
    "Hải Đăng",
    "Thùy Dung",
    "Thiền Tâm Đức",
    "Ngọc Huyền",
    "Quang Sơn",
    "Ngọc Trân",
    "Minh Đức",
    "Phạm Tuyên",
    "Thái Sơn",
    "Xuân Vĩnh",
    "Thanh Bình",
    "Ngọc Linh",
    "Đoan Trang",
    "Thục Đoan",
    "Minh Triết",
    "Mỹ Duyên",
    "Quỳnh Anh",
    "Đức Trí",
    "Kim Thanh",
    "Adam",
    "Quốc Tuấn",
)


@pytest.fixture(autouse=True)
def clear_vieneu_shared_sessions():
    """Keep process-wide provider state from leaking between focused tests."""
    from buzz.localization import providers

    providers._cleanup_vieneu_sessions()
    yield
    providers._cleanup_vieneu_sessions()


def test_voice_choices_keep_edge_default_and_exact_vieneu_presets():
    assert TTS_VOICE_CHOICES[0][1] == "vi-VN-HoaiMyNeural"
    assert len(TTS_VOICE_CHOICES) == 27
    assert tuple(value.removeprefix("vieneu:") for _, value in VIENEU_VOICE_CHOICES) == (
        EXPECTED_VIENEU_VOICES
    )
    assert parse_tts_voice("vi-VN-HoaiMyNeural") == (
        "edge",
        "vi-VN-HoaiMyNeural",
    )
    assert parse_tts_voice("edge:vi-VN-NamMinhNeural") == (
        "edge",
        "vi-VN-NamMinhNeural",
    )
    assert parse_tts_voice("vieneu:Trúc Ly") == ("vieneu", "Trúc Ly")


def test_vieneu_worker_protocol_uses_utf8_and_wav(monkeypatch, tmp_path):
    runtime = tmp_path / "python.exe"
    model = tmp_path / "model"
    cache = tmp_path / "hf"
    output = tmp_path / "speech"
    runtime.write_text("")
    model.mkdir()
    cache.mkdir()

    class Input:
        def write(self, line):
            payload = json.loads(line)
            (tmp_path / "speech.wav").write_bytes(b"wav")
            self.response = json.dumps(
                {
                    "id": payload["id"],
                    "ok": True,
                    "duration": 1.25,
                    "metadata": {"format": "wav"},
                }
            ) + "\n"

        def flush(self):
            pass

    class Process:
        def __init__(self):
            self.stdin = Input()
            self.stdout = self

        def readline(self):
            return self.stdin.response

        def poll(self):
            return None

    monkeypatch.setattr(
        "buzz.localization.providers.subprocess.Popen",
        lambda *args, **kwargs: Process(),
    )
    provider = VieNeuTTSProvider(
        python_executable=str(runtime),
        model_dir=str(model),
        hf_home=str(cache),
    )
    result = provider.synthesize(
        TTSRequest(
            "Xin chào Việt Nam",
            "vi",
            str(output),
            voice="Trúc Ly",
        )
    )
    assert result.audio_file.endswith(".wav")
    assert result.audio_duration == 1.25
    assert result.provider == "vieneu-v3-turbo"
    assert result.voice == "Trúc Ly"


def test_vieneu_identity_differs_by_voice(tmp_path):
    runtime = tmp_path / "python.exe"
    model = tmp_path / "model"
    cache = tmp_path / "hf"
    runtime.write_text("")
    model.mkdir()
    cache.mkdir()

    request = TTSRequest(
        "Xin chào",
        "vi",
        str(tmp_path / "audio"),
        voice="Adam",
    )
    first = VieNeuTTSProvider(
        "Adam",
        str(runtime),
        str(model),
        str(cache),
    )
    second = VieNeuTTSProvider(
        "Trúc Ly",
        str(runtime),
        str(model),
        str(cache),
    )
    assert _request_fingerprint(request, first) != _request_fingerprint(
        request,
        second,
    )


def test_vieneu_providers_share_worker_until_final_close(monkeypatch, tmp_path):
    runtime = tmp_path / "python.exe"
    model = tmp_path / "model"
    cache = tmp_path / "hf"
    runtime.write_text("")
    model.mkdir()
    cache.mkdir()
    processes = []
    process_created = threading.Event()

    class Input:
        def __init__(self, process):
            self.process = process

        def write(self, line):
            payload = json.loads(line)
            self.process.payloads.append(payload)
            Path(payload["output_file"]).write_bytes(b"wav")
            self.process.responses.append(json.dumps({
                "id": payload["id"], "ok": True, "duration": 1.0,
            }) + "\n")
            if self.process.block_first and len(self.process.payloads) == 1:
                self.process.first_written.set()
                assert self.process.release_first.wait(2)

        def flush(self):
            pass

    class Process:
        def __init__(self, block_first=False):
            self.payloads = []
            self.responses = []
            self.terminated = False
            self.block_first = block_first
            self.first_written = threading.Event()
            self.release_first = threading.Event()
            self.stdin = Input(self)
            self.stdout = self

        def readline(self):
            return self.responses.pop(0)

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    def popen(*args, **kwargs):
        process = Process(block_first=(len(processes) == 0))
        processes.append(process)
        process_created.set()
        return process

    monkeypatch.setattr("buzz.localization.providers.subprocess.Popen", popen)
    config = {
        "python_executable": str(runtime),
        "model_dir": str(model),
        "hf_home": str(cache),
    }
    first = VieNeuTTSProvider(**config)
    second = VieNeuTTSProvider(**config)
    errors = []

    def synthesize(provider, stem):
        try:
            provider.synthesize(TTSRequest(stem, "vi", str(tmp_path / stem)))
        except BaseException as exc:
            errors.append(exc)

    first_thread = threading.Thread(target=synthesize, args=(first, "first"))
    first_thread.start()
    assert process_created.wait(2)
    assert processes[0].first_written.wait(2)
    second_thread = threading.Thread(target=synthesize, args=(second, "second"))
    second_thread.start()
    assert len(processes[0].payloads) == 1
    processes[0].release_first.set()
    first_thread.join(2)
    second_thread.join(2)

    assert not errors
    assert len(processes) == 1
    assert [payload["id"] for payload in processes[0].payloads] == [1, 2]
    first.close()
    assert not processes[0].terminated
    second.close()
    assert processes[0].terminated

    third = VieNeuTTSProvider(**config)
    third.synthesize(TTSRequest("third", "vi", str(tmp_path / "third")))
    assert len(processes) == 2
    third.close()
    assert processes[1].terminated


def test_vieneu_recovers_from_one_stalled_worker(monkeypatch, tmp_path):
    """A timeout must kill the stuck worker, then synthesize from a fresh process."""
    from buzz.localization.providers import LocalizationProviderError

    runtime = tmp_path / "python.exe"
    runtime.touch()
    model, cache = tmp_path / "model", tmp_path / "hf"
    model.mkdir()
    cache.mkdir()
    processes = []

    class Process:
        def __init__(self, stuck):
            self.stuck = stuck
            self.closed = threading.Event()
            self.stdin = self
            self.stdout = self

        def write(self, line):
            self.payload = json.loads(line)
            if not self.stuck:
                Path(self.payload["output_file"]).write_bytes(b"wav")

        def flush(self):
            pass

        def readline(self):
            if self.stuck:
                self.closed.wait(2)
                return ""
            return json.dumps({
                "id": self.payload["id"], "ok": True, "duration": 1.5,
            }) + "\n"

        def poll(self):
            return 0 if self.closed.is_set() else None

        def terminate(self):
            self.closed.set()

        def wait(self, timeout):
            return 0

    def launch(*args, **kwargs):
        proc = Process(stuck=len(processes) == 0)
        processes.append(proc)
        return proc

    monkeypatch.setattr("buzz.localization.providers.subprocess.Popen", launch)
    provider = VieNeuTTSProvider(
        python_executable=str(runtime), model_dir=str(model), hf_home=str(cache),
        worker_timeout_seconds=0.02, max_timeout_retries=1,
    )
    try:
        result = provider.synthesize(TTSRequest(
            "Xin chào", "vi", str(tmp_path / "result"), voice="Ngọc Huyền",
        ))
        assert result.audio_duration == 1.5
        assert len(processes) == 2
        assert processes[0].closed.is_set()
    finally:
        provider.close()


def test_vieneu_two_timeouts_raise_without_infinite_retry(monkeypatch, tmp_path):
    from buzz.localization.providers import LocalizationProviderError

    runtime = tmp_path / "python.exe"
    runtime.touch()
    model, cache = tmp_path / "model", tmp_path / "hf"
    model.mkdir()
    cache.mkdir()
    processes = []

    class StuckProcess:
        def __init__(self):
            self.closed = threading.Event()
            self.stdin = self
            self.stdout = self
        def write(self, line):
            pass
        def flush(self):
            pass
        def readline(self):
            self.closed.wait(2)
            return ""
        def poll(self):
            return 0 if self.closed.is_set() else None
        def terminate(self):
            self.closed.set()
        def wait(self, timeout):
            return 0

    def launch(*args, **kwargs):
        process = StuckProcess()
        processes.append(process)
        return process

    monkeypatch.setattr("buzz.localization.providers.subprocess.Popen", launch)
    provider = VieNeuTTSProvider(
        python_executable=str(runtime), model_dir=str(model), hf_home=str(cache),
        worker_timeout_seconds=0.02, max_timeout_retries=1,
    )
    with pytest.raises(LocalizationProviderError, match=r"after 2 attempts"):
        provider.synthesize(TTSRequest("Xin chào", "vi", str(tmp_path / "result")))
    assert len(processes) == 2
    assert all(process.closed.is_set() for process in processes)
    provider.close()


def test_vieneu_terminates_launcher_and_child_process_tree_on_windows(monkeypatch):
    import os
    from types import SimpleNamespace
    from buzz.localization import providers

    if os.name != "nt":
        pytest.skip("Windows virtualenv launcher behavior")

    class Process:
        pid = 45678
        def __init__(self):
            self.waited = False
        def poll(self):
            return None
        def wait(self, timeout):
            self.waited = True

    command_list = []
    def fake_run(command, **kwargs):
        command_list.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(providers.subprocess, "run", fake_run)
    proc = Process()
    session = SimpleNamespace(process=proc)
    providers.VieNeuTTSProvider._stop_worker(session)
    assert session.process is None
    assert command_list == [["taskkill", "/PID", "45678", "/T", "/F"]]
    assert proc.waited
