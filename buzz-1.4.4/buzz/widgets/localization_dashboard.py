"""Five-slot FIFO localization dashboard for the Auto Video home screen."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

from PyQt6.QtCore import QObject, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton, QVBoxLayout, QWidget
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key, get_password, set_password
from buzz.localization.paraformer import reset_paraformer_model_cache
from buzz.localization.final_render import (
    FinalRenderError,
    refresh_dubbed_portrait_subtitles,
    subtitle_refresh_output_path,
)
from buzz.widgets.localization_dialog import LocalizationWorker
from buzz.widgets.voice_preview import VoicePreviewControl
from buzz.localization.voices import TTS_VOICE_CHOICES

MAX_LOCALIZATION_JOBS = 5
MEDIA_FILTER = "Media Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)"
WAITING, PROCESSING, CANCELLING, CANCELLED, COMPLETED, FAILED = ("Đang chờ", "Đang xử lý", "Đang hủy", "Đã hủy", "Hoàn thành", "Lỗi")

@dataclass
class LocalizationJob:
    source_video: str
    output_directory: str
    status: str = WAITING
    progress: int = 0
    detail: str = ""
    worker: LocalizationWorker | None = None
    thread: QThread | None = None
    blur_bilibili_watermark: bool = True
    overlay_ben_review_logo: bool = True
    portrait_subtitle_overlay: bool = False


_LIVE_LOCALIZATION_JOBS: list[LocalizationJob] = []


class SubtitleRefreshWorker(QObject):
    """A cancellable, isolated refresh of subtitles on an already dubbed part."""
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, source_video: str, subtitle_file: str | None, output_file: str):
        super().__init__()
        import threading
        self.source_video, self.subtitle_file, self.output_file = source_video, subtitle_file, output_file
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            self.progress.emit(5, "Dang tao phu de doc de va render lai...")
            result = refresh_dubbed_portrait_subtitles(
                self.source_video, self.output_file, subtitle_file=self.subtitle_file,
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.progress.emit(100, "Da render xong phu de moi.")
                self.completed.emit(result.video_file)
        except Exception as exc:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()

class LocalizationJobRow(QWidget):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.name_label, self.status_label, self.detail_label = QLabel(), QLabel(), QLabel()
        self.detail_label.setWordWrap(True)
        self.progress_bar, self.action_button, self.retry_button, self.open_button = QProgressBar(), QPushButton(), QPushButton(), QPushButton()
        self.progress_bar.setRange(0, 100)
        self.action_button.setFixedWidth(70)
        self.retry_button.setFixedWidth(70)
        self.open_button.setFixedWidth(70)
        header = QHBoxLayout()
        header.addWidget(self.name_label, 1); header.addWidget(self.status_label); header.addWidget(self.open_button); header.addWidget(self.retry_button); header.addWidget(self.action_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header); layout.addWidget(self.progress_bar); layout.addWidget(self.detail_label)
        self.show_empty()
    def show_empty(self):
        self.name_label.setText(f"{self.index + 1}. Trống"); self.status_label.clear(); self.detail_label.clear()
        self.progress_bar.setValue(0); self.progress_bar.hide(); self.action_button.hide(); self.retry_button.hide(); self.open_button.hide()
    def show_job(self, job):
        self.name_label.setText(f"{self.index + 1}. {Path(job.source_video).name}")
        self.status_label.setText(job.status); self.detail_label.setText(job.detail)
        self.progress_bar.setValue(job.progress); self.progress_bar.show()
        if job.status in {WAITING, PROCESSING, CANCELLING}:
            self.action_button.setText("Hủy")
        else:
            self.action_button.setText("Xóa")
        self.open_button.setText("Mở")
        self.open_button.setVisible(job.status == COMPLETED)
        can_retry = job.status in {FAILED, CANCELLED}
        self.retry_button.setText("Thử lại")
        self.retry_button.setVisible(can_retry)
        self.retry_button.setEnabled(can_retry and job.worker is None and job.thread is None)
        self.action_button.setEnabled(not can_retry or (job.worker is None and job.thread is None))
        self.action_button.show()

class LocalizationDashboard(QWidget):
    """Five independent UI slots backed by concurrently active workers."""
    def __init__(self, parent=None, worker_factory: Callable[..., LocalizationWorker] = LocalizationWorker, *, persist_api_key: bool = True):
        super().__init__(parent)
        self.settings, self.worker_factory = Settings(), worker_factory
        self._persist_api_key = persist_api_key
        self.jobs = [None] * MAX_LOCALIZATION_JOBS
        self.active_jobs = []
        self._shutting_down = False
        self._build_ui()
    def _build_ui(self):
        default = "D:/AutoVideoOutput" if Path("D:/").exists() else str(Path.home() / "AutoVideoOutput")
        # Gemini is the only new-job translation path.  GEMINI_MODEL remains
        # an advanced environment override, while provider fallback stays internal.
        self._gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.api_key_edit = QLineEdit(
            get_password(Key.GEMINI_API_KEY) if self._persist_api_key else ""
        )
        # Kept as one-item implementation details for compatibility with
        # callers which inspect the dashboard, but never shown to the user.
        self.translation_provider_combo = QComboBox()
        self.translation_provider_combo.addItem("Gemini API", "gemini")
        self.translation_base_url_edit = QLineEdit()
        self.translation_model_edit = QLineEdit(self._gemini_model)
        self.translation_status_label = QLabel()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password); self.api_key_edit.editingFinished.connect(self.save_api_key)
        self.voice_combo = QComboBox()
        for label, value in TTS_VOICE_CHOICES: self.voice_combo.addItem(label, value)
        self.voice_preview_control = VoicePreviewControl(self.voice_combo, self)
        voice_row = QWidget(self)
        voice_row_layout = QHBoxLayout(voice_row)
        voice_row_layout.setContentsMargins(0, 0, 0, 0)
        voice_row_layout.addWidget(self.voice_combo, 3)
        voice_row_layout.addWidget(self.voice_preview_control, 2)
        self.blur_bilibili_watermark_checkbox = QCheckBox("Làm mờ logo Bilibili (góc phải trên)")
        self.blur_bilibili_watermark_checkbox.setChecked(True)
        self.blur_bilibili_watermark_checkbox.setToolTip("Làm mờ và phủ tối nhẹ vùng logo ở góc phải trên.")
        self.overlay_ben_review_logo_checkbox = QCheckBox("Chèn logo Ben review")
        self.overlay_ben_review_logo_checkbox.setChecked(True)
        self.overlay_ben_review_logo_checkbox.setToolTip("Chèn logo Ben review đã duyệt ở góc phải trên.")
        self.landscape_video_radio = QRadioButton("Video ngang (16:9)")
        self.portrait_subtitle_overlay_checkbox = QRadioButton("Video dọc (9:16)")
        self.video_orientation_group = QButtonGroup(self)
        self.video_orientation_group.addButton(self.landscape_video_radio)
        self.video_orientation_group.addButton(self.portrait_subtitle_overlay_checkbox)
        self.landscape_video_radio.setChecked(True)
        self.portrait_subtitle_overlay_checkbox.setToolTip(
            "Cho video 9:16; che logo trái trên, cảnh báo dưới và phụ đề Trung 68-71% chiều cao."
        )
        orientation_row = QWidget(self)
        orientation_layout = QHBoxLayout(orientation_row)
        orientation_layout.setContentsMargins(0, 0, 0, 0)
        orientation_layout.addWidget(self.landscape_video_radio)
        orientation_layout.addWidget(self.portrait_subtitle_overlay_checkbox)
        orientation_layout.addStretch()
        self.subtitle_refresh_source_edit = QLineEdit()
        self.subtitle_refresh_srt_edit = QLineEdit()
        self.subtitle_refresh_source_button = QPushButton("Select dubbed MP4")
        self.subtitle_refresh_srt_button = QPushButton("Select SRT")
        self.subtitle_refresh_button = QPushButton("Render lai phu de")
        self.subtitle_refresh_status = QLabel()
        self.subtitle_refresh_status.setWordWrap(True)
        self.subtitle_refresh_source_button.clicked.connect(self.browse_subtitle_refresh_source)
        self.subtitle_refresh_srt_button.clicked.connect(self.browse_subtitle_refresh_srt)
        self.subtitle_refresh_button.clicked.connect(self.start_subtitle_refresh)
        self._subtitle_refresh_worker = None
        self._subtitle_refresh_thread = None
        self.output_edit = QLineEdit(self._configured_output_directory(default))
        self.output_button = QPushButton("Chọn thư mục"); self.output_button.clicked.connect(self.browse_output)
        self.add_video_button = QPushButton("+ Thêm video"); self.add_video_button.clicked.connect(self.browse_videos)
        output_row = QWidget(); output_layout = QHBoxLayout(output_row); output_layout.setContentsMargins(0, 0, 0, 0); output_layout.addWidget(self.output_edit); output_layout.addWidget(self.output_button)
        form = QFormLayout()
        form.addRow("Dịch bằng:", self.translation_provider_combo)
        form.addRow("", self.translation_status_label)
        self.translation_base_url_label = QLabel("Translation API base URL:")
        self.translation_api_key_label = QLabel("Gemini API key:")
        self.translation_model_label = QLabel("Translation model(s):")
        self.translation_model_edit.setToolTip(
            "Một model hoặc danh sách ưu tiên, ngăn cách bằng dấu phẩy. "
            "Khi hết quota, app sẽ thử model tiếp theo."
        )
        form.addRow(self.translation_base_url_label, self.translation_base_url_edit)
        form.addRow(self.translation_api_key_label, self.api_key_edit)
        form.addRow(self.translation_model_label, self.translation_model_edit)
        form.addRow("Giọng Việt:", voice_row)
        form.addRow("Loại video:", orientation_row)
        form.addRow("Thư mục đầu ra:", output_row)
        form.addRow("", self.blur_bilibili_watermark_checkbox)
        form.addRow("", self.overlay_ben_review_logo_checkbox)
        refresh_source_row = QWidget(); refresh_source_layout = QHBoxLayout(refresh_source_row); refresh_source_layout.setContentsMargins(0, 0, 0, 0); refresh_source_layout.addWidget(self.subtitle_refresh_source_edit); refresh_source_layout.addWidget(self.subtitle_refresh_source_button)
        refresh_srt_row = QWidget(); refresh_srt_layout = QHBoxLayout(refresh_srt_row); refresh_srt_layout.setContentsMargins(0, 0, 0, 0); refresh_srt_layout.addWidget(self.subtitle_refresh_srt_edit); refresh_srt_layout.addWidget(self.subtitle_refresh_srt_button)
        form.addRow("Dubbed MP4:", refresh_source_row)
        form.addRow("SRT (optional):", refresh_srt_row)
        advanced_form = QFormLayout()
        for widget in (
            self.translation_provider_combo, self.translation_status_label,
            self.translation_base_url_edit, self.translation_model_edit,
            output_row, self.blur_bilibili_watermark_checkbox,
            self.overlay_ben_review_logo_checkbox, refresh_source_row,
            refresh_srt_row,
        ):
            row = form.takeRow(widget)
            if widget in {
                self.translation_provider_combo, self.translation_status_label,
                self.translation_base_url_edit, self.translation_model_edit,
            }:
                continue
            if row.fieldItem is None:
                continue
            field = row.fieldItem.widget()
            label = row.labelItem.widget() if row.labelItem is not None else None
            if label is not None:
                advanced_form.addRow(label, field)
            else:
                advanced_form.addRow(field)
        advanced_form.addRow(self.subtitle_refresh_button)
        advanced_form.addRow(self.subtitle_refresh_status)
        self.advanced_options_button = QPushButton("Advanced options")
        self.advanced_options_button.setCheckable(True)
        self.advanced_options = QWidget(self)
        self.advanced_options.setLayout(advanced_form)
        self.advanced_options.hide()
        self.advanced_options_button.toggled.connect(self.advanced_options.setVisible)
        self.rows = [LocalizationJobRow(index, self) for index in range(MAX_LOCALIZATION_JOBS)]
        for row in self.rows:
            row.action_button.clicked.connect(lambda _=False, row=row: self.on_row_action(row.index))
            row.retry_button.clicked.connect(lambda _=False, row=row: self.on_retry(row.index))
            row.open_button.clicked.connect(lambda _=False, row=row: self.open_job_output(row.index))
        layout = QVBoxLayout(self); layout.addWidget(QLabel("Dịch & lồng tiếng video")); layout.addLayout(form)
        pipeline = QLabel("Tự động: Paraformer-zh → dịch thuật → Edge / VieNeu TTS → đồng bộ tự nhiên → làm mờ phụ đề gốc → phụ đề Việt"); pipeline.setWordWrap(True)
        layout.addWidget(pipeline); layout.addWidget(self.add_video_button)
        layout.addWidget(self.advanced_options_button); layout.addWidget(self.advanced_options)
        for row in self.rows: layout.addWidget(row)
        layout.addStretch()

    def browse_subtitle_refresh_source(self):
        source, _ = QFileDialog.getOpenFileName(self, "Select dubbed MP4", "", "MP4 Files (*.mp4)")
        if source:
            source_path = Path(source)
            self.subtitle_refresh_source_edit.setText(source)
            adjacent = source_path.with_suffix(".srt")
            self.subtitle_refresh_srt_edit.setText(str(adjacent) if adjacent.is_file() else "")

    def browse_subtitle_refresh_srt(self):
        subtitle, _ = QFileDialog.getOpenFileName(self, "Select Vietnamese subtitles", "", "Subtitle Files (*.srt *.vtt)")
        if subtitle:
            self.subtitle_refresh_srt_edit.setText(subtitle)

    def start_subtitle_refresh(self):
        if self._subtitle_refresh_thread is not None:
            return
        source = Path(self.subtitle_refresh_source_edit.text().strip())
        subtitle_text = self.subtitle_refresh_srt_edit.text().strip()
        subtitle = Path(subtitle_text) if subtitle_text else source.with_suffix(".srt")
        output = subtitle_refresh_output_path(source)
        if not source.is_file() or source.suffix.lower() != ".mp4":
            self.subtitle_refresh_status.setText("Select an existing dubbed MP4.")
            return
        if not subtitle.is_file() or subtitle.suffix.lower() not in {".srt", ".vtt"}:
            self.subtitle_refresh_status.setText("Select an adjacent or explicit SRT/VTT.")
            return
        if output.exists():
            self.subtitle_refresh_status.setText(f"Refusing to overwrite existing output: {output}")
            return
        worker = SubtitleRefreshWorker(str(source), str(subtitle), str(output))
        thread = QThread()
        self._subtitle_refresh_worker, self._subtitle_refresh_thread = worker, thread
        self.subtitle_refresh_button.setText("Cancel subtitle render")
        self.subtitle_refresh_button.clicked.disconnect()
        self.subtitle_refresh_button.clicked.connect(self.cancel_subtitle_refresh)
        worker.moveToThread(thread); thread.started.connect(worker.run)
        worker.progress.connect(lambda progress, detail: self.subtitle_refresh_status.setText(f"{progress}% - {detail}"))
        worker.completed.connect(lambda path: self.subtitle_refresh_status.setText(f"Completed: {path}"))
        worker.failed.connect(lambda error: self.subtitle_refresh_status.setText(f"Failed: {error}"))
        worker.cancelled.connect(lambda: self.subtitle_refresh_status.setText("Cancelled; partial output removed."))
        worker.finished.connect(thread.quit); worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._finish_subtitle_refresh)
        thread.start()

    def cancel_subtitle_refresh(self):
        if self._subtitle_refresh_worker is not None:
            self.subtitle_refresh_status.setText("Cancelling subtitle render...")
            self._subtitle_refresh_worker.request_cancel()

    def _finish_subtitle_refresh(self):
        thread = self._subtitle_refresh_thread
        if thread is not None:
            thread.deleteLater()
        self._subtitle_refresh_worker = self._subtitle_refresh_thread = None
        self.subtitle_refresh_button.setText("Render lai phu de")
        self.subtitle_refresh_button.clicked.disconnect()
        self.subtitle_refresh_button.clicked.connect(self.start_subtitle_refresh)

    def _translation_configuration_error(self) -> str | None:
        if not self.api_key_edit.text().strip():
            return "Vui lòng nhập Gemini API key trước khi thêm video."
        if not self.translation_model_edit.text().strip():
            return "Vui lòng nhập model dịch thuật."
        return None
    def save_api_key(self):
        if not self._persist_api_key:
            return
        if api_key := self.api_key_edit.text().strip():
            try: set_password(Key.GEMINI_API_KEY, api_key)
            except Exception: pass
    def browse_output(self):
        selected = QFileDialog.getExistingDirectory(self, "Chọn thư mục đầu ra", self.output_edit.text())
        if selected: self.output_edit.setText(selected); self.settings.set_value(Settings.Key.LOCALIZATION_OUTPUT_DIRECTORY, selected)
    def browse_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Chọn video", "", MEDIA_FILTER); self.add_videos(paths)
    @staticmethod
    def _is_writable_directory(directory: Path) -> bool:
        """Create and probe an output directory before a job relies on it."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".autovideo-write-", delete=True):
                pass
        except OSError:
            return False
        return True

    def _configured_output_directory(self, default: str) -> str:
        configured = str(self.settings.value(
            Settings.Key.LOCALIZATION_OUTPUT_DIRECTORY, ""
        ) or "").strip()
        if not configured:
            self.settings.set_value(Settings.Key.LOCALIZATION_OUTPUT_DIRECTORY, default)
            return default
        if self._is_writable_directory(Path(configured)):
            return configured

        fallback = Path(default)
        if self._is_writable_directory(fallback):
            self.settings.set_value(
                Settings.Key.LOCALIZATION_OUTPUT_DIRECTORY, str(fallback)
            )
        return str(fallback)

    def _show_output_directory_error(self, directory: str) -> None:
        QMessageBox.critical(
            self,
            "Thư mục đầu ra",
            "Không thể tạo hoặc ghi vào thư mục đầu ra:\n"
            f"{directory}\n\nVui lòng chọn một thư mục khác mà bạn có quyền ghi.",
        )

    def add_videos(self, paths):
        # QFileDialog normally returns unique paths, but drag/drop and callers
        # can submit duplicates.  Canonical paths prevent a second job from
        # allocating a "-2" output while the first one owns the checkpoint.
        paths = list(dict.fromkeys(os.path.normcase(str(Path(path).resolve())) for path in paths))
        existing_by_source = {
            os.path.normcase(str(Path(job.source_video).resolve())): job
            for job in self.jobs if job
        }
        retryable = [path for path in paths if (
            (job := existing_by_source.get(path)) is not None
            and job.status in {FAILED, CANCELLED}
            and job.worker is None and job.thread is None
        )]
        available = self.jobs.count(None)
        accepted = retryable + [path for path in paths
                                if path not in retryable and path not in existing_by_source][:available]
        if len(paths) > available: QMessageBox.warning(self, "Hàng đợi video", f"Chỉ có thể có tối đa {MAX_LOCALIZATION_JOBS} video. Đã thêm {len(accepted)} video.")
        if not accepted: return []
        root = self.output_edit.text().strip()
        if not root: QMessageBox.warning(self, "Hàng đợi video", "Vui lòng chọn thư mục đầu ra."); return []
        root_path = Path(root)
        if not self._is_writable_directory(root_path):
            self._show_output_directory_error(root)
            return []
        self.save_api_key(); self.settings.set_value(Settings.Key.LOCALIZATION_OUTPUT_DIRECTORY, root); added = []
        for path in accepted:
            retry = next((job for job in self.jobs if job and os.path.normcase(
                str(Path(job.source_video).resolve())) == path
                          and job.status in {FAILED, CANCELLED}
                          and job.worker is None and job.thread is None), None)
            if retry:
                retry.status, retry.progress, retry.detail = WAITING, 0, "Tiếp tục từ checkpoint hiện có..."
                self.refresh_job(retry); added.append(retry)
                continue
            if self.jobs.count(None) == 0:
                continue
            job = LocalizationJob(
                path, str(self._job_output_directory(root_path, Path(path))),
                blur_bilibili_watermark=self.blur_bilibili_watermark_checkbox.isChecked(),
                overlay_ben_review_logo=self.overlay_ben_review_logo_checkbox.isChecked(),
                portrait_subtitle_overlay=self.portrait_subtitle_overlay_checkbox.isChecked(),
            )
            self.jobs[self.jobs.index(None)] = job; self.refresh_job(job); added.append(job)
        self.start_next_job(); return added

    def _job_output_directory(self, root, source):
        used = {Path(job.output_directory) for job in self.jobs if job}
        pattern = re.compile(rf"{re.escape(source.stem)}(?:-(\d+))?")
        try:
            incomplete = [directory for directory in root.iterdir()
                          if directory.is_dir() and pattern.fullmatch(directory.name)
                          and directory not in used
                          and (directory / f"{source.stem}_localization_work").is_dir()
                          and not (directory / f"{source.stem}.vi.mp4").exists()]
        except OSError:
            incomplete = []
        if incomplete:
            def resume_priority(directory):
                # A newly created "-2" folder must never win over an earlier
                # folder with durable part-level checkpoints just by suffix.
                work = directory / f"{source.stem}_localization_work"
                progress = 0
                for part in work.glob("part_[0-9][0-9][0-9][0-9]"):
                    if ((part / "localization-part-complete.json").is_file()
                            or (part / "localization-asr-checkpoint.json").is_file()
                            or (part / "source_localization_work"
                                / "timing-recovery-checkpoint.json").is_file()
                            or (part / "source_localization_work"
                                / "translation-checkpoint.json").is_file()):
                        progress += 1
                generation = int(pattern.fullmatch(directory.name).group(1) or 1)
                return (progress, generation)
            return max(incomplete, key=resume_priority)
        candidate, suffix = root / source.stem, 2
        while candidate in used or candidate.exists(): candidate = root / f"{source.stem}-{suffix}"; suffix += 1
        return candidate
    def start_next_job(self):
        if self._shutting_down: return
        jobs = [item for item in self.jobs if item and item.status == WAITING]
        for job in jobs:
            self._start_job(job)

    def _start_job(self, job):
        if error := self._translation_configuration_error():
            job.status, job.detail = FAILED, error; self.refresh_job(job); self.start_next_job(); return
        try:
            Path(job.output_directory).mkdir(parents=True, exist_ok=True)
        except OSError:
            job.status, job.detail = FAILED, "Không thể tạo thư mục đầu ra."
            self.refresh_job(job)
            self._show_output_directory_error(job.output_directory)
            return
        job.status, job.detail, job.progress = PROCESSING, "Đang khởi tạo...", 0; self.active_jobs.append(job); self.refresh_job(job)
        worker = self.worker_factory(source_video=job.source_video, output_directory=job.output_directory, source_language=None, translation_provider_name="gemini", translation_base_url=None, translation_api_key=self.api_key_edit.text().strip(), translation_model=self._gemini_model, tts_voice=self.voice_combo.currentData(), subtitle_mode="burn", cover_original_subtitles=True, blur_bilibili_watermark=job.blur_bilibili_watermark, overlay_ben_review_logo=job.overlay_ben_review_logo, portrait_subtitle_overlay=job.portrait_subtitle_overlay, use_background_separation=False)
        # This thread has no widget parent.  When a window closes while a job
        # is cancelling, the registry keeps the QObject graph alive until the
        # worker emits ``finished`` and the QThread has actually stopped.
        thread = QThread(); job.worker, job.thread = worker, thread; _LIVE_LOCALIZATION_JOBS.append(job); worker.moveToThread(thread); thread.started.connect(worker.run)
        worker.progress.connect(lambda progress, detail, job=job: self.on_progress(job, progress, detail)); worker.completed.connect(lambda result, job=job: self.on_completed(job, result)); worker.failed.connect(lambda error, job=job: self.on_failed(job, error)); worker.cancelled.connect(lambda job=job: self.on_cancelled(job)); worker.finished.connect(thread.quit); worker.finished.connect(worker.deleteLater); thread.finished.connect(lambda job=job: self.on_thread_finished(job)); thread.start()
    def on_progress(self, job, progress, detail): job.progress, job.detail = progress, detail; self.refresh_job(job)
    def on_completed(self, job, result): job.status, job.progress, job.detail = COMPLETED, 100, str(getattr(getattr(result, "final_video", None), "video_file", job.output_directory)); self.refresh_job(job)
    def on_failed(self, job, error): job.status, job.detail = FAILED, str(error); self.refresh_job(job)
    def on_cancelled(self, job): job.status, job.detail = CANCELLED, "Đã hủy theo yêu cầu."; self.refresh_job(job)
    def on_thread_finished(self, job):
        thread = job.thread
        if thread is None:
            return
        # QThread.finished can arrive before the native thread has returned.
        # Keep both wrappers alive until a nonblocking wait confirms it has.
        if not thread.wait(0):
            QTimer.singleShot(75, lambda job=job, thread=thread: self._finalize_thread(job, thread))
            return
        self._finalize_thread(job, thread)

    def _finalize_thread(self, job, thread):
        if job.thread is not thread:
            return
        if not thread.wait(0):
            QTimer.singleShot(75, lambda job=job, thread=thread: self._finalize_thread(job, thread))
            return
        thread.deleteLater()
        job.worker, job.thread = None, None
        if job in _LIVE_LOCALIZATION_JOBS:
            _LIVE_LOCALIZATION_JOBS.remove(job)
        if job in self.active_jobs: self.active_jobs.remove(job)
        if job in self.jobs:
            self.refresh_job(job)
        if (not self.active_jobs and not any(
            item and item.status in {WAITING, PROCESSING, CANCELLING}
            for item in self.jobs
        )):
            reset_paraformer_model_cache()
    def on_retry(self, index):
        job = self.jobs[index]
        if (self._shutting_down or not job or job.status not in {FAILED, CANCELLED}
                or job.worker is not None or job.thread is not None):
            return
        if error := self._translation_configuration_error():
            QMessageBox.warning(self, "Gemini API key", error)
            return
        output_path = Path(job.output_directory)
        if not output_path.is_dir() or not self._is_writable_directory(output_path):
            self._show_output_directory_error(job.output_directory)
            return
        self.save_api_key()
        job.status, job.progress, job.detail = WAITING, 0, "Tiếp tục từ checkpoint hiện có..."
        self.refresh_job(job)
        self.start_next_job()
    def on_row_action(self, index):
        job = self.jobs[index]
        if not job: return
        if job.status == WAITING: job.status, job.detail = CANCELLED, "Đã hủy theo yêu cầu."; self.refresh_row(index)
        elif job.status in {PROCESSING, CANCELLING}: job.status, job.detail = CANCELLING, "Đang gửi yêu cầu hủy..."; job.worker and job.worker.request_cancel(); self.refresh_row(index)
        elif job.worker is None and job.thread is None:
            self.jobs[index] = None; self.refresh_row(index)
    def open_job_output(self, index):
        job = self.jobs[index]
        if not job or job.status != COMPLETED:
            return
        target = Path(job.detail)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target if target.exists() else Path(job.output_directory))))
    def shutdown(self):
        self._shutting_down = True
        self.cancel_subtitle_refresh()
        for job in tuple(self.active_jobs):
            if job.worker: job.worker.request_cancel()
    def refresh_job(self, job): self.refresh_row(self.jobs.index(job))
    def refresh_row(self, index): self.rows[index].show_empty() if self.jobs[index] is None else self.rows[index].show_job(self.jobs[index])
