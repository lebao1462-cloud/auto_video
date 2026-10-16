"""The selector previews VieNeu voices without starting a localization job."""

from types import SimpleNamespace

from PyQt6.QtWidgets import QComboBox

from buzz.widgets import voice_preview


def test_preview_only_enabled_for_vieneu(qtbot):
    combo = QComboBox()
    combo.addItem("Edge - Hoài My", "vi-VN-HoaiMyNeural")
    combo.addItem("VieNeu - Ngọc Huyền", "vieneu:Ngọc Huyền")
    control = voice_preview.VoicePreviewControl(combo)
    qtbot.addWidget(control)
    assert not control.button.isEnabled()
    combo.setCurrentIndex(1)
    assert control.button.isEnabled()
    assert "Nghe thử" in control.button.text()


def test_cached_preview_plays_without_starting_synthesis(qtbot, tmp_path, monkeypatch):
    combo = QComboBox()
    combo.addItem("VieNeu - Trúc Ly", "vieneu:Trúc Ly")
    control = voice_preview.VoicePreviewControl(combo)
    qtbot.addWidget(control)
    fake_file = tmp_path / "voice.wav"
    fake_file.write_bytes(b"cache" * 400)
    monkeypatch.setattr(voice_preview, "_preview_file", lambda voice: fake_file)
    played = []
    monkeypatch.setattr(control, "_play", lambda path: played.append(path))
    control.preview_selected()
    assert played == [fake_file]
    assert control._thread is None


def test_voice_worker_generates_offline_and_emits_cache_path(tmp_path, monkeypatch):
    file = tmp_path / "preview.wav"
    monkeypatch.setattr(voice_preview, "_preview_file", lambda voice: file)

    class FakeProvider:
        def synthesize(self, request):
            assert request.text == voice_preview.PREVIEW_TEXT
            assert request.voice == "Ngọc Huyền"
            assert request.language == "vi"
            file.write_bytes(b"audio" * 300)
            return SimpleNamespace(audio_file=str(file))

    monkeypatch.setattr(voice_preview, "_PREVIEW_PROVIDER", FakeProvider())
    worker = voice_preview.VoicePreviewThread("Ngọc Huyền")
    succeeded = []
    failures = []
    worker.generated.connect(succeeded.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert succeeded == [str(file)]
    assert not failures
    worker.run()  # cached, no repeated synthesis
    assert succeeded == [str(file), str(file)]
