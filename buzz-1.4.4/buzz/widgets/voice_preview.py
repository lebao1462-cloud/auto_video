"""Play a cached short offline sample of the selected VieNeu voice in Auto Video."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton, QHBoxLayout, QWidget

from buzz.localization.voices import parse_tts_voice


PREVIEW_TEXT = (
    "Xin chào, đây là giọng đọc thử của kênh Ben review. "
    "Chúng ta cùng bắt đầu câu chuyện nhé!"
)
# Hold active QThreads until they finish even if their originating widget is closed.
_ACTIVE_PREVIEW_THREADS: set[QThread] = set()
_PREVIEW_PROVIDER = None


def _preview_file(voice: str) -> Path:
    """Stable path per voice and phrase; no keys, user data or video involved."""
    root = Path("D:/AutoVideoOutput/_voice_previews")
    if not root.parent.is_dir():
        root = Path(tempfile.gettempdir()) / "auto_video_voice_previews"
    identifier = hashlib.sha256((voice + "\n" + PREVIEW_TEXT).encode("utf-8")).hexdigest()[:20]
    return root / f"voice_{identifier}.wav"


class VoicePreviewThread(QThread):
    generated = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, voice: str, parent=None):
        super().__init__(parent)
        self.voice = voice

    def run(self):
        global _PREVIEW_PROVIDER
        from buzz.localization.providers import VieNeuTTSProvider
        from buzz.localization.tts import TTSRequest

        try:
            output = _preview_file(self.voice)
            if output.is_file() and output.stat().st_size > 1024:
                self.generated.emit(str(output))
                return
            output.parent.mkdir(parents=True, exist_ok=True)
            if _PREVIEW_PROVIDER is None:
                _PREVIEW_PROVIDER = VieNeuTTSProvider()
            result = _PREVIEW_PROVIDER.synthesize(
                TTSRequest(
                    text=PREVIEW_TEXT,
                    language="vi",
                    output_file_stem=str(output.with_suffix("")),
                    voice=self.voice,
                )
            )
            if not Path(result.audio_file).is_file():
                raise RuntimeError("VieNeu không tạo được tệp âm thanh.")
            self.generated.emit(result.audio_file)
        except Exception as exc:
            self.failed.emit(str(exc))


class VoicePreviewControl(QWidget):
    """Nonblocking sample player attached to an existing TTS voice dropdown."""

    def __init__(self, voice_combo: QComboBox, parent=None):
        super().__init__(parent)
        self.voice_combo = voice_combo
        self.button = QPushButton("▶ Nghe thử")
        self.button.setToolTip("Nghe thử giọng VieNeu đã chọn, không cần Gemini API.")
        self.status = QLabel("")
        self.status.setWordWrap(False)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.90)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self._pending_voice: str | None = None
        self._thread: VoicePreviewThread | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.button)
        row.addWidget(self.status, 1)
        self.button.clicked.connect(self.preview_selected)
        self.voice_combo.currentIndexChanged.connect(self._selection_changed)
        self.player.errorOccurred.connect(self._player_failed)
        self._selection_changed()

    def _selection_changed(self, _=None):
        provider, _voice = parse_tts_voice(self.voice_combo.currentData())
        self.button.setEnabled(provider == "vieneu" and self._thread is None)
        if provider == "vieneu":
            self.status.setText("Chọn giọng rồi bấm nghe thử.")
        else:
            self.status.setText("Nghe thử dành cho VieNeu.")
        self.player.stop()

    def preview_selected(self):
        provider, voice = parse_tts_voice(self.voice_combo.currentData())
        if provider != "vieneu" or self._thread is not None:
            return
        target = _preview_file(voice)
        self._pending_voice = voice
        if target.is_file() and target.stat().st_size > 1024:
            self._play(target)
            return

        self.button.setEnabled(False)
        self.status.setText("Đang tạo mẫu giọng, vui lòng chờ…")
        thread = VoicePreviewThread(voice)
        self._thread = thread
        _ACTIVE_PREVIEW_THREADS.add(thread)
        thread.generated.connect(self._generated)
        thread.failed.connect(self._failed)
        thread.finished.connect(lambda thread=thread: self._finished(thread))
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _generated(self, path: str):
        _provider, current_voice = parse_tts_voice(self.voice_combo.currentData())
        if current_voice == self._pending_voice:
            self._play(Path(path))
        else:
            self.status.setText("Đã lưu mẫu; chọn giọng và nghe lại.")

    def _play(self, path: Path):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()
        self.status.setText("Đang phát mẫu giọng.")

    def _failed(self, message: str):
        self.status.setText("Không tạo được mẫu: " + message[:130])

    def _player_failed(self, _error, error_string):
        self.status.setText("Không phát được âm thanh: " + error_string[:130])

    def _finished(self, thread: VoicePreviewThread):
        _ACTIVE_PREVIEW_THREADS.discard(thread)
        if self._thread is thread:
            self._thread = None
            self._pending_voice = None
            provider, _ = parse_tts_voice(self.voice_combo.currentData())
            self.button.setEnabled(provider == "vieneu")
