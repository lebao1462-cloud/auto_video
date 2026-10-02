import os
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from buzz.localization.final_render import FinalRenderOptions
from buzz.localization.timing import TimingPolicy
from buzz.localization.preflight import (
    run_localization_preflight,
    write_localization_diagnostics,
)
from buzz.localization.providers import (
    ArgosTranslationProvider,
    EdgeTTSProvider,
    GeminiTranslationProvider,
    NLLBTranslationProvider,
    OpenAICompatibleTranslationProvider,
    normalize_gemini_model_name,
)
from buzz.localization.workflow import (
    LocalizationCancelled,
    LocalizationProgress,
    LocalizationWorkflowOptions,
    localize_video,
)
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key, get_password, set_password
from buzz.transcriber.transcriber import Task, TranscriptionOptions


class LocalizationWorker(QObject):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        source_video: str,
        output_directory: str,
        source_language: str | None,
        translation_provider_name: str,
        translation_base_url: str | None,
        translation_api_key: str,
        translation_model: str,
        tts_voice: str,
        subtitle_mode: str,
        cover_original_subtitles: bool,
        use_background_separation: bool,
    ):
        super().__init__()
        self.source_video = source_video
        self.output_directory = output_directory
        self.source_language = source_language
        self.translation_provider_name = translation_provider_name
        self.translation_base_url = translation_base_url
        self.translation_api_key = translation_api_key
        self.translation_model = translation_model
        self.tts_voice = tts_voice
        self.subtitle_mode = subtitle_mode
        self.cover_original_subtitles = cover_original_subtitles
        self.use_background_separation = use_background_separation
        self.cancel_event = threading.Event()

    def request_cancel(self):
        self.cancel_event.set()

    def _on_progress(self, progress: LocalizationProgress):
        self.progress.emit(round(progress.fraction * 100), progress.message)

    def run(self):
        try:
            if self.translation_provider_name == "argos":
                translation_provider = ArgosTranslationProvider()
            elif self.translation_provider_name == "nllb":
                translation_provider = NLLBTranslationProvider()
            elif self.translation_provider_name == "gemini":
                translation_provider = GeminiTranslationProvider(
                    api_key=self.translation_api_key,
                    model=self.translation_model,
                )
            else:
                translation_provider = OpenAICompatibleTranslationProvider(
                    api_key=self.translation_api_key,
                    model=self.translation_model,
                    base_url=self.translation_base_url,
                )
            tts_provider = EdgeTTSProvider(default_voice=self.tts_voice)
            result = localize_video(
                self.source_video,
                TranscriptionOptions(
                    language=self.source_language,
                    task=Task.TRANSCRIBE,
                ),
                "",
                translation_provider,
                tts_provider,
                LocalizationWorkflowOptions(
                    output_directory=self.output_directory,
                    use_background_separation=self.use_background_separation,
                    tts_voice=self.tts_voice,
                    timing_policy=TimingPolicy(),
                    render_options=FinalRenderOptions(
                        subtitle_mode=self.subtitle_mode,
                        cover_original_subtitles=self.cover_original_subtitles,
                    ),
                ),
                progress_callback=self._on_progress,
                cancel_event=self.cancel_event,
                asr_provider="paraformer-zh",
            )
            self.completed.emit(result)
        except LocalizationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class LocalizationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Localize Video to Vietnamese")
        self.resize(720, 520)
        self.worker_thread: QThread | None = None
        self.worker: LocalizationWorker | None = None

        self.settings = Settings()
        self.source_edit = QLineEdit()
        self.output_edit = QLineEdit(self._default_output_directory())
        self.asr_provider_label = QLabel(
            "Paraformer-zh - Chinese / Offline / Recommended"
        )
        self.asr_provider_status = QLabel(
            "Chinese only; offline after the first model download. Models are stored "
            "in D:\\Dev\\modelscope-cache when D: is available; no API key."
        )

        self.translation_provider_combo = QComboBox()
        self.translation_provider_combo.addItem(
            "NLLB-200 - Offline / Free / No API key", "nllb"
        )
        self.translation_provider_combo.addItem(
            "Argos Translate - Offline / Free / No API key", "argos"
        )
        self.translation_provider_combo.addItem(
            "Gemini API - Online / Optional", "gemini"
        )
        self.translation_provider_combo.addItem(
            "OpenAI-compatible - Advanced", "openai-compatible"
        )
        saved_provider = self.settings.value(
            Settings.Key.LOCALIZATION_TRANSLATION_PROVIDER,
            "nllb",
        )
        saved_index = self.translation_provider_combo.findData(saved_provider)
        self.translation_provider_combo.setCurrentIndex(
            saved_index if saved_index >= 0 else 0
        )
        self.translation_provider_status = QLabel()

        self.base_url_edit = QLineEdit(
            os.getenv("BUZZ_TRANSLATION_API_BASE_URL", "")
        )
        self._translation_api_keys = {
            "gemini": (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("BUZZ_TRANSLATION_API_KEY")
                or get_password(Key.GEMINI_API_KEY)
                or ""
            ),
            "openai-compatible": (
                os.getenv("BUZZ_TRANSLATION_API_KEY")
                or get_password(Key.OPENAI_API_KEY)
                or ""
            ),
        }
        self._active_translation_provider = (
            self.translation_provider_combo.currentData()
        )
        self.api_key_edit = QLineEdit(
            self._translation_api_keys.get(
                self._active_translation_provider, ""
            )
        )
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.editingFinished.connect(
            self._save_current_api_key
        )
        self.translation_model_edit = QLineEdit(
            os.getenv("BUZZ_TRANSLATION_MODEL", "gemini-3.5-flash-lite")
        )
        self.translation_model_edit.editingFinished.connect(
            self._normalize_translation_model
        )

        self.language_combo = QComboBox()
        self.language_combo.addItem("Auto detect (Chinese)", None)
        self.language_combo.addItem("Chinese", "zh")

        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Hoai My - Female", "vi-VN-HoaiMyNeural")
        self.voice_combo.addItem("Nam Minh - Male", "vi-VN-NamMinhNeural")

        self.subtitle_mode_combo = QComboBox()
        self.subtitle_mode_combo.addItem("Soft subtitle", "soft")
        self.subtitle_mode_combo.addItem("Burn subtitle into video", "burn")
        self.subtitle_mode_combo.addItem("No subtitle in MP4", "none")

        self.cover_original_subtitles_checkbox = QCheckBox(
            "Blur original subtitles area (bottom 18%)"
        )
        self.cover_original_subtitles_checkbox.setChecked(
            self.settings.value(
                Settings.Key.LOCALIZATION_COVER_ORIGINAL_SUBTITLES,
                True,
            )
        )
        self.cover_original_subtitles_checkbox.setToolTip(
            "Blurs the lower 18% of the video, adds a light dark overlay, then burns Vietnamese subtitles on top."
        )

        self.background_checkbox = QCheckBox(
            "Separate original vocals and preserve background/music (Demucs)"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Ready")

        self.start_button = QPushButton("Start Localization")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)

        source_row = self._path_row(self.source_edit, self._browse_source)
        output_row = self._path_row(self.output_edit, self._browse_output)

        form = QFormLayout()
        form.addRow("Input video:", source_row)
        form.addRow("Output folder:", output_row)
        form.addRow("ASR engine:", self.asr_provider_label)
        form.addRow("", self.asr_provider_status)
        form.addRow("Source language:", self.language_combo)
        form.addRow("Translation provider:", self.translation_provider_combo)
        form.addRow("", self.translation_provider_status)
        self.base_url_label = QLabel("Translation API base URL:")
        self.api_key_label = QLabel("Translation API key:")
        self.translation_model_label = QLabel("Translation model:")
        form.addRow(self.base_url_label, self.base_url_edit)
        form.addRow(self.api_key_label, self.api_key_edit)
        form.addRow(self.translation_model_label, self.translation_model_edit)
        form.addRow("Vietnamese voice:", self.voice_combo)
        form.addRow("Subtitle mode:", self.subtitle_mode_combo)
        form.addRow("", self.cover_original_subtitles_checkbox)
        form.addRow("", self.background_checkbox)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)

        self.start_button.clicked.connect(self.start_localization)
        self.cancel_button.clicked.connect(self.cancel_localization)
        self.translation_provider_combo.currentIndexChanged.connect(
            self._on_translation_provider_changed
        )
        self.subtitle_mode_combo.currentIndexChanged.connect(
            self._on_subtitle_mode_changed
        )
        self._on_translation_provider_changed()
        self._on_subtitle_mode_changed()

    @staticmethod
    def _path_row(edit: QLineEdit, callback) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse...")
        button.clicked.connect(callback)
        layout.addWidget(edit)
        layout.addWidget(button)
        return widget

    @staticmethod
    def _default_output_directory() -> str:
        d_drive = Path("D:/AutoVideoOutput")
        if Path("D:/").exists():
            return str(d_drive)
        return str(Path.home() / "AutoVideoOutput")

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select source video",
            "",
            "Media Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)",
        )
        if path:
            self.source_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_edit.text(),
        )
        if path:
            self.output_edit.setText(path)

    def _on_subtitle_mode_changed(self):
        is_burn = self.subtitle_mode_combo.currentData() == "burn"
        self.cover_original_subtitles_checkbox.setEnabled(is_burn)

    def _normalize_translation_model(self):
        if self.translation_provider_combo.currentData() == "gemini":
            normalized = normalize_gemini_model_name(
                self.translation_model_edit.text()
            )
            if normalized:
                self.translation_model_edit.setText(normalized)

    def _save_current_api_key(self):
        provider = self.translation_provider_combo.currentData()
        api_key = self.api_key_edit.text().strip()
        if provider not in {"gemini", "openai-compatible"} or not api_key:
            return
        self._translation_api_keys[provider] = api_key
        key = (
            Key.GEMINI_API_KEY
            if provider == "gemini"
            else Key.OPENAI_API_KEY
        )
        try:
            set_password(key, api_key)
        except Exception:
            # A keyring failure must not prevent localization from running.
            return

    def _on_translation_provider_changed(self):
        provider = self.translation_provider_combo.currentData()
        previous = getattr(self, "_active_translation_provider", None)
        if hasattr(self, "api_key_edit") and previous != provider:
            if previous in {"gemini", "openai-compatible"}:
                self._translation_api_keys[previous] = (
                    self.api_key_edit.text().strip()
                )
            self.api_key_edit.setText(
                self._translation_api_keys.get(provider, "")
            )
            self._active_translation_provider = provider

        is_offline = provider in {"argos", "nllb"}
        is_argos = provider == "argos"
        is_nllb = provider == "nllb"
        is_gemini = provider == "gemini"
        is_openai = provider == "openai-compatible"

        self.base_url_label.setVisible(is_openai)
        self.base_url_edit.setVisible(is_openai)
        self.api_key_label.setVisible(not is_offline)
        self.api_key_edit.setVisible(not is_offline)
        self.translation_model_label.setVisible(not is_offline)
        self.translation_model_edit.setVisible(not is_offline)

        if is_argos:
            self.translation_provider_status.setText(
                "Offline / free. Uses installed Argos language packages; no API key."
            )
        elif is_nllb:
            self.translation_provider_status.setText(
                "Direct English/Chinese -> Vietnamese. First use downloads NLLB-200 once; then it works offline with no API key."
            )
        elif is_gemini:
            self.translation_provider_status.setText(
                "Gemini corrects Chinese transcript and translates 30 segments per request. API quota/network access may apply."
            )
            if self.translation_model_edit.text().strip() in {"", "gpt-4o-mini"}:
                self.translation_model_edit.setText(
                    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
                )
        else:
            self.translation_provider_status.setText(
                "Advanced OpenAI-compatible endpoint."
            )

        self.settings.set_value(
            Settings.Key.LOCALIZATION_TRANSLATION_PROVIDER,
            provider,
        )

    def _validate_inputs(self) -> str | None:
        if not Path(self.source_edit.text()).is_file():
            return "Please select a valid input video."
        if not self.output_edit.text().strip():
            return "Please select an output folder."
        provider = self.translation_provider_combo.currentData()
        if provider == "gemini":
            if not self.api_key_edit.text().strip():
                return "Gemini API key is required."
            if not self.translation_model_edit.text().strip():
                return "Gemini model is required."
        elif provider == "openai-compatible":
            if not self.api_key_edit.text().strip():
                return "Translation API key is required."
            if not self.translation_model_edit.text().strip():
                return "Translation model is required."
        return None

    def start_localization(self):
        self._normalize_translation_model()
        error = self._validate_inputs()
        if error:
            QMessageBox.warning(self, "Localization", error)
            return

        self._save_current_api_key()
        self.settings.set_value(
            Settings.Key.LOCALIZATION_COVER_ORIGINAL_SUBTITLES,
            self.cover_original_subtitles_checkbox.isChecked(),
        )

        output_directory = Path(self.output_edit.text())
        output_directory.mkdir(parents=True, exist_ok=True)
        preflight = run_localization_preflight(
            self.source_edit.text(),
            "",
            output_directory,
            translation_api_key=self.api_key_edit.text().strip(),
            translation_model=self.translation_model_edit.text().strip(),
            translation_provider=self.translation_provider_combo.currentData(),
            source_language=self.language_combo.currentData(),
            require_edge_tts=True,
            use_background_separation=self.background_checkbox.isChecked(),
            asr_provider="paraformer-zh",
        )
        write_localization_diagnostics(
            preflight,
            output_directory / "localization_diagnostics.json",
        )
        if not preflight.ok:
            QMessageBox.critical(
                self,
                "Localization preflight failed",
                "\n".join(preflight.errors),
            )
            return
        if preflight.warnings:
            self.status_label.setText("Preflight warning: " + "; ".join(preflight.warnings))

        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.worker_thread = QThread(self)
        self.worker = LocalizationWorker(
            source_video=self.source_edit.text(),
            output_directory=self.output_edit.text(),
            source_language=self.language_combo.currentData(),
            translation_provider_name=self.translation_provider_combo.currentData(),
            translation_base_url=self.base_url_edit.text().strip() or None,
            translation_api_key=self.api_key_edit.text().strip(),
            translation_model=self.translation_model_edit.text().strip(),
            tts_voice=self.voice_combo.currentData(),
            subtitle_mode=self.subtitle_mode_combo.currentData(),
            cover_original_subtitles=(
                self.cover_original_subtitles_checkbox.isChecked()
                and self.subtitle_mode_combo.currentData() == "burn"
            ),
            use_background_separation=self.background_checkbox.isChecked(),
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def cancel_localization(self):
        if self.worker is not None:
            self.status_label.setText("Cancelling after current stage...")
            self.cancel_button.setEnabled(False)
            self.worker.request_cancel()

    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def _on_completed(self, result):
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Completed: {result.final_video.video_file}")
        QMessageBox.information(
            self,
            "Localization complete",
            f"Final video:\n{result.final_video.video_file}",
        )

    def _on_failed(self, error: str):
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Localization failed", error)

    def _on_cancelled(self):
        self.status_label.setText("Cancelled")

    def _on_worker_finished(self):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
        self.worker_thread = None
