import os
import re
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
from buzz.localization.subtitles import SubtitleOptions
from buzz.localization.timing import TimingPolicy
from buzz.localization.preflight import (
    run_localization_preflight,
    write_localization_diagnostics,
)
from buzz.localization.providers import (
    ArgosTranslationProvider,
    EdgeTTSProvider,
    VieNeuTTSProvider,
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
from buzz.widgets.voice_preview import VoicePreviewControl
from buzz.localization.voices import TTS_VOICE_CHOICES, parse_tts_voice
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key, get_password, set_password
from buzz.transcriber.transcriber import Task, TranscriptionOptions


_OPENAI_COMPATIBLE_PROVIDER_DEFAULTS = {
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4.7",
        "key": Key.GROK_API_KEY,
        "environment_key": "XAI_API_KEY",
        "status": "xAI Grok via its OpenAI-compatible API. Network access and billing may apply.",
    },
    "cloudflare": {
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "key": Key.CLOUDFLARE_API_TOKEN,
        "environment_key": "CLOUDFLARE_API_TOKEN",
        "status": "Cloudflare Workers AI. Replace {account_id} in the base URL with your Cloudflare account ID.",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash",
        "key": Key.OPENROUTER_API_KEY,
        "environment_key": "OPENROUTER_API_KEY",
        "status": "OpenRouter's OpenAI-compatible API. Choose any supported model ID.",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "key": Key.MISTRAL_API_KEY,
        "environment_key": "MISTRAL_API_KEY",
        "status": "Mistral's OpenAI-compatible Chat Completions API.",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b",
        "key": Key.GROQ_API_KEY,
        "environment_key": "GROQ_API_KEY",
        "status": "Groq's OpenAI-compatible API. Free-tier rate limits may apply.",
    },
    "openai-compatible": {
        "base_url": "",
        "model": "",
        "key": Key.OPENAI_API_KEY,
        "environment_key": "BUZZ_TRANSLATION_API_KEY",
        "status": "Custom OpenAI-compatible endpoint (for example a private Grok gateway).",
    },
}


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
        blur_bilibili_watermark: bool,
        overlay_ben_review_logo: bool = True,
        portrait_subtitle_overlay: bool = False,
        use_background_separation: bool = False,
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
        self.blur_bilibili_watermark = blur_bilibili_watermark
        self.overlay_ben_review_logo = overlay_ben_review_logo
        self.portrait_subtitle_overlay = portrait_subtitle_overlay
        self.use_background_separation = use_background_separation
        self.cancel_event = threading.Event()

    def request_cancel(self):
        self.cancel_event.set()

    def _on_progress(self, progress: LocalizationProgress):
        self.progress.emit(round(progress.fraction * 100), progress.message)

    def run(self):
        tts_provider = None
        try:
            if self.translation_provider_name != "gemini":
                raise ValueError("Only Gemini translation is supported for new localization jobs.")
            gemini_models = tuple(
                part.strip() for part in re.split(r"[,\n\r]+", self.translation_model)
                if part.strip()
            )
            translation_provider = GeminiTranslationProvider(
                api_key=self.translation_api_key,
                model=gemini_models[0],
                rate_limit_fallback_models=gemini_models[1:],
            )
            tts_provider_name, tts_voice = parse_tts_voice(self.tts_voice)
            tts_provider = (VieNeuTTSProvider(default_voice=tts_voice)
                            if tts_provider_name == "vieneu" else EdgeTTSProvider(
                                default_voice=tts_voice, batch_enabled=False))
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
                    tts_voice=tts_voice,
                    timing_policy=TimingPolicy(),
                    subtitle_options=SubtitleOptions(
                        max_line_length=None if self.portrait_subtitle_overlay else 42,
                        max_lines=1 if self.portrait_subtitle_overlay else 2,
                    ),
                    render_options=FinalRenderOptions(
                        subtitle_mode=self.subtitle_mode,
                        cover_original_subtitles=self.cover_original_subtitles,
                        blur_bilibili_watermark=self.blur_bilibili_watermark,
                        overlay_ben_review_logo=self.overlay_ben_review_logo,
                        portrait_subtitle_overlay=self.portrait_subtitle_overlay,
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
            close_tts = getattr(tts_provider, "close", None)
            if callable(close_tts):
                close_tts()
            self.finished.emit()


class LocalizationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Dịch & lồng tiếng video")
        self.resize(640, 520)
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
            "Grok / xAI - Online", "grok"
        )
        self.translation_provider_combo.addItem(
            "Cloudflare Workers AI - Online", "cloudflare"
        )
        self.translation_provider_combo.addItem(
            "OpenRouter - Online", "openrouter"
        )
        self.translation_provider_combo.addItem(
            "Mistral API - Online", "mistral"
        )
        self.translation_provider_combo.addItem(
            "Groq - Online / Free tier", "groq"
        )
        self.translation_provider_combo.addItem(
            "OpenAI-compatible - Advanced", "openai-compatible"
        )
        self.translation_provider_combo.clear()
        self.translation_provider_combo.addItem("Gemini API", "gemini")
        self.translation_provider_status = QLabel()

        self._translation_api_keys = {
            "gemini": (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("BUZZ_TRANSLATION_API_KEY")
                or get_password(Key.GEMINI_API_KEY)
                or ""
            ),
        }
        self._translation_base_urls = {}
        self._translation_models = {
            "gemini": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
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
        self.base_url_edit = QLineEdit(
            self._translation_base_urls.get(self._active_translation_provider, "")
        )
        self.translation_model_edit = QLineEdit(
            self._translation_models.get(self._active_translation_provider, "")
        )
        self.translation_model_edit.editingFinished.connect(
            self._normalize_translation_model
        )
        self.translation_model_edit.setToolTip(
            "Một model hoặc danh sách ưu tiên, ngăn cách bằng dấu phẩy. "
            "Khi hết quota, app sẽ thử model tiếp theo."
        )

        self.language_combo = QComboBox()
        self.language_combo.addItem("Auto detect (Chinese)", None)
        self.language_combo.addItem("Chinese", "zh")

        self.voice_combo = QComboBox()
        for label, value in TTS_VOICE_CHOICES:
            self.voice_combo.addItem(label, value)

        self.voice_preview_control = VoicePreviewControl(self.voice_combo, self)
        voice_row = QWidget(self)
        voice_row_layout = QHBoxLayout(voice_row)
        voice_row_layout.setContentsMargins(0, 0, 0, 0)
        voice_row_layout.addWidget(self.voice_combo, 3)
        voice_row_layout.addWidget(self.voice_preview_control, 2)

        self.subtitle_mode_combo = QComboBox()
        self.subtitle_mode_combo.addItem("Soft subtitle", "soft")
        self.subtitle_mode_combo.addItem("Burn subtitle into video", "burn")
        self.subtitle_mode_combo.addItem("No subtitle in MP4", "none")
        self.subtitle_mode_combo.setCurrentIndex(
            self.subtitle_mode_combo.findData("burn")
        )

        self.cover_original_subtitles_checkbox = QCheckBox(
            "Blur original subtitles area (bottom 18%)"
        )
        # Product default: always hide hardcoded source subtitles before
        # burning Vietnamese subtitles on top.
        self.cover_original_subtitles_checkbox.setChecked(True)
        self.cover_original_subtitles_checkbox.setToolTip(
            "Blurs the lower 18% of the video, adds a light dark overlay, then burns Vietnamese subtitles on top."
        )
        self.blur_bilibili_watermark_checkbox = QCheckBox(
            "Làm mờ logo Bilibili (góc phải trên)"
        )
        self.blur_bilibili_watermark_checkbox.setChecked(True)
        self.blur_bilibili_watermark_checkbox.setToolTip(
            "Blurs and lightly darkens the upper-right 24% × 15% watermark region."
        )

        self.overlay_ben_review_logo_checkbox = QCheckBox("Chèn logo Ben review")
        self.overlay_ben_review_logo_checkbox.setChecked(True)
        self.overlay_ben_review_logo_checkbox.setToolTip(
            "Adds the approved Ben review logo at the upper-right of the output video."
        )

        self.portrait_subtitle_overlay_checkbox = QCheckBox(
            "Phim dọc: phụ đề Việt 1 dòng đè lên chữ Trung (làm mờ 3 vùng)"
        )
        self.portrait_subtitle_overlay_checkbox.setChecked(False)
        self.portrait_subtitle_overlay_checkbox.setToolTip(
            "Dành cho video 9:16: làm mờ watermark trên trái, cảnh báo dưới, "
            "và phụ đề Trung ở độ cao 68-71%, rồi ghi phụ đề Việt tại đó."
        )

        self.background_checkbox = QCheckBox(
            "Separate original vocals and preserve background/music (Demucs)"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Sẵn sàng")

        self.start_button = QPushButton("Bắt đầu")
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.setEnabled(False)

        source_row = self._path_row(self.source_edit, self._browse_source)

        form = QFormLayout()
        form.addRow("Video:", source_row)
        form.addRow("Translation provider:", self.translation_provider_combo)
        self.translation_provider_status.setWordWrap(True)
        form.addRow("", self.translation_provider_status)
        self.base_url_label = QLabel("Translation API base URL:")
        form.addRow(self.base_url_label, self.base_url_edit)
        self.api_key_label = QLabel("Translation API key:")
        self.translation_model_label = QLabel("Translation model(s):")
        form.addRow(self.api_key_label, self.api_key_edit)
        form.addRow(self.translation_model_label, self.translation_model_edit)
        form.addRow("Giọng Việt:", voice_row)
        form.addRow("", self.blur_bilibili_watermark_checkbox)
        form.addRow("", self.overlay_ben_review_logo_checkbox)
        form.addRow("", self.portrait_subtitle_overlay_checkbox)

        pipeline_label = QLabel("Tự động: Paraformer-zh → dịch thuật → Edge / VieNeu TTS → phụ đề Việt")
        pipeline_label.setWordWrap(True)
        form.addRow("", pipeline_label)

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
        # The legacy dialog retains source/start controls, but Gemini is fixed.
        self.translation_provider_combo.hide()
        self.translation_provider_status.hide()
        self.translation_model_label.hide()
        self.translation_model_edit.hide()

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
            normalized = ", ".join(
                value for value in (
                    normalize_gemini_model_name(item)
                    for item in re.split(r"[,\n\r]+", self.translation_model_edit.text())
                ) if value
            )
            if normalized:
                self.translation_model_edit.setText(normalized)

    def _save_current_api_key(self):
        provider = self.translation_provider_combo.currentData()
        api_key = self.api_key_edit.text().strip()
        if provider not in {"gemini", *_OPENAI_COMPATIBLE_PROVIDER_DEFAULTS} or not api_key:
            return
        self._translation_api_keys[provider] = api_key
        key = Key.GEMINI_API_KEY if provider == "gemini" else _OPENAI_COMPATIBLE_PROVIDER_DEFAULTS[provider]["key"]
        try:
            set_password(key, api_key)
        except Exception:
            # A keyring failure must not prevent localization from running.
            return

    def _on_translation_provider_changed(self):
        provider = self.translation_provider_combo.currentData()
        previous = getattr(self, "_active_translation_provider", None)
        if hasattr(self, "api_key_edit") and previous != provider:
            if previous in {"gemini", *_OPENAI_COMPATIBLE_PROVIDER_DEFAULTS}:
                self._translation_api_keys[previous] = (
                    self.api_key_edit.text().strip()
                )
                self._translation_base_urls[previous] = self.base_url_edit.text().strip()
                self._translation_models[previous] = self.translation_model_edit.text().strip()
            self.api_key_edit.setText(
                self._translation_api_keys.get(provider, "")
            )
            self.base_url_edit.setText(self._translation_base_urls.get(provider, ""))
            self.translation_model_edit.setText(self._translation_models.get(provider, ""))
            self._active_translation_provider = provider

        is_offline = provider in {"argos", "nllb"}
        is_argos = provider == "argos"
        is_nllb = provider == "nllb"
        is_gemini = provider == "gemini"
        is_openai_compatible = provider in _OPENAI_COMPATIBLE_PROVIDER_DEFAULTS

        self.base_url_label.setVisible(is_openai_compatible)
        self.base_url_edit.setVisible(is_openai_compatible)
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
                _OPENAI_COMPATIBLE_PROVIDER_DEFAULTS[provider]["status"]
            )
        self.api_key_label.setText(
            "Gemini API key:" if is_gemini else "Translation API key:"
        )
        self.api_key_edit.setPlaceholderText(
            "Nhập Gemini API key" if is_gemini else "Nhập API key / token"
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
        elif provider in _OPENAI_COMPATIBLE_PROVIDER_DEFAULTS:
            if not self.api_key_edit.text().strip():
                return "Translation API key is required."
            if not self.translation_model_edit.text().strip():
                return "Translation model is required."
            base_url = self.base_url_edit.text().strip()
            if not base_url:
                return "Translation API base URL is required."
            if provider == "cloudflare" and "{account_id}" in base_url:
                return "Replace {account_id} in the Cloudflare API base URL."
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
            blur_bilibili_watermark=self.blur_bilibili_watermark_checkbox.isChecked(),
            overlay_ben_review_logo=self.overlay_ben_review_logo_checkbox.isChecked(),
            portrait_subtitle_overlay=(self.portrait_subtitle_overlay_checkbox.isChecked() and self.subtitle_mode_combo.currentData() == "burn"),
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
