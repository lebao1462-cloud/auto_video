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
from buzz.localization.preflight import (
    run_localization_preflight,
    write_localization_diagnostics,
)
from buzz.localization.providers import (
    ArgosTranslationProvider,
    EdgeTTSProvider,
    GeminiTranslationProvider,
    OpenAICompatibleTranslationProvider,
)
from buzz.localization.workflow import (
    LocalizationCancelled,
    LocalizationProgress,
    LocalizationWorkflowOptions,
    localize_video,
)
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key, get_password
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
        model_path: str,
        translation_provider_name: str,
        translation_base_url: str | None,
        translation_api_key: str,
        translation_model: str,
        tts_voice: str,
        subtitle_mode: str,
        use_background_separation: bool,
    ):
        super().__init__()
        self.source_video = source_video
        self.output_directory = output_directory
        self.source_language = source_language
        self.model_path = model_path
        self.translation_provider_name = translation_provider_name
        self.translation_base_url = translation_base_url
        self.translation_api_key = translation_api_key
        self.translation_model = translation_model
        self.tts_voice = tts_voice
        self.subtitle_mode = subtitle_mode
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
                self.model_path,
                translation_provider,
                tts_provider,
                LocalizationWorkflowOptions(
                    output_directory=self.output_directory,
                    use_background_separation=self.use_background_separation,
                    tts_voice=self.tts_voice,
                    render_options=FinalRenderOptions(
                        subtitle_mode=self.subtitle_mode,
                    ),
                ),
                progress_callback=self._on_progress,
                cancel_event=self.cancel_event,
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
        self.model_path_edit = QLineEdit(self._default_model_path())

        self.translation_provider_combo = QComboBox()
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
            "argos",
        )
        saved_index = self.translation_provider_combo.findData(saved_provider)
        self.translation_provider_combo.setCurrentIndex(
            saved_index if saved_index >= 0 else 0
        )
        self.translation_provider_status = QLabel()

        self.base_url_edit = QLineEdit(
            os.getenv("BUZZ_TRANSLATION_API_BASE_URL", "")
        )
        self.api_key_edit = QLineEdit(
            os.getenv("BUZZ_TRANSLATION_API_KEY", get_password(Key.OPENAI_API_KEY) or "")
        )
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.translation_model_edit = QLineEdit(
            os.getenv("BUZZ_TRANSLATION_MODEL", "gemini-2.5-flash")
        )

        self.language_combo = QComboBox()
        self.language_combo.addItem("Auto detect (English/Chinese)", None)
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Chinese", "zh")

        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Hoai My - Female", "vi-VN-HoaiMyNeural")
        self.voice_combo.addItem("Nam Minh - Male", "vi-VN-NamMinhNeural")

        self.subtitle_mode_combo = QComboBox()
        self.subtitle_mode_combo.addItem("Soft subtitle", "soft")
        self.subtitle_mode_combo.addItem("Burn subtitle into video", "burn")
        self.subtitle_mode_combo.addItem("No subtitle in MP4", "none")

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
        model_row = self._path_row(self.model_path_edit, self._browse_model)

        form = QFormLayout()
        form.addRow("Input video:", source_row)
        form.addRow("Output folder:", output_row)
        form.addRow("Source language:", self.language_combo)
        form.addRow("Whisper model path:", model_row)
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
        self._on_translation_provider_changed()

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

    @staticmethod
    def _default_model_path() -> str:
        root = os.getenv("BUZZ_MODEL_ROOT")
        if root:
            candidate = Path(root) / "whisper" / "tiny.pt"
            if candidate.is_file():
                return str(candidate)
        return ""

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

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Whisper model",
            self.model_path_edit.text(),
            "Model Files (*.pt *.bin);;All Files (*)",
        )
        if path:
            self.model_path_edit.setText(path)

    def _on_translation_provider_changed(self):
        provider = self.translation_provider_combo.currentData()
        is_argos = provider == "argos"
        is_gemini = provider == "gemini"
        is_openai = provider == "openai-compatible"

        self.base_url_label.setVisible(is_openai)
        self.base_url_edit.setVisible(is_openai)
        self.api_key_label.setVisible(not is_argos)
        self.api_key_edit.setVisible(not is_argos)
        self.translation_model_label.setVisible(not is_argos)
        self.translation_model_edit.setVisible(not is_argos)

        if is_argos:
            self.translation_provider_status.setText(
                "Offline / free. Uses installed Argos language packages; no API key."
            )
        elif is_gemini:
            self.translation_provider_status.setText(
                "Online Gemini translation. API quota/network access may apply."
            )
            if self.translation_model_edit.text().strip() in {"", "gpt-4o-mini"}:
                self.translation_model_edit.setText(
                    os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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
        if not Path(self.model_path_edit.text()).exists():
            return "Please select a valid Whisper model path."
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
        error = self._validate_inputs()
        if error:
            QMessageBox.warning(self, "Localization", error)
            return

        output_directory = Path(self.output_edit.text())
        output_directory.mkdir(parents=True, exist_ok=True)
        preflight = run_localization_preflight(
            self.source_edit.text(),
            self.model_path_edit.text(),
            output_directory,
            translation_api_key=self.api_key_edit.text().strip(),
            translation_model=self.translation_model_edit.text().strip(),
            translation_provider=self.translation_provider_combo.currentData(),
            source_language=self.language_combo.currentData(),
            require_edge_tts=True,
            use_background_separation=self.background_checkbox.isChecked(),
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
            model_path=self.model_path_edit.text(),
            translation_provider_name=self.translation_provider_combo.currentData(),
            translation_base_url=self.base_url_edit.text().strip() or None,
            translation_api_key=self.api_key_edit.text().strip(),
            translation_model=self.translation_model_edit.text().strip(),
            tts_voice=self.voice_combo.currentData(),
            subtitle_mode=self.subtitle_mode_combo.currentData(),
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
