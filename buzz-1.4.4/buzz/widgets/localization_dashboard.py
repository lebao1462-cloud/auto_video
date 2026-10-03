"""Five-slot FIFO localization dashboard for the Auto Video home screen."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable

from PyQt6.QtCore import QThread, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key, get_password, set_password
from buzz.localization.paraformer import reset_paraformer_model_cache
from buzz.widgets.localization_dialog import LocalizationWorker

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


_LIVE_LOCALIZATION_JOBS: list[LocalizationJob] = []

class LocalizationJobRow(QWidget):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.name_label, self.status_label, self.detail_label = QLabel(), QLabel(), QLabel()
        self.detail_label.setWordWrap(True)
        self.progress_bar, self.action_button, self.open_button = QProgressBar(), QPushButton(), QPushButton()
        self.progress_bar.setRange(0, 100)
        self.action_button.setFixedWidth(70)
        self.open_button.setFixedWidth(70)
        header = QHBoxLayout()
        header.addWidget(self.name_label, 1); header.addWidget(self.status_label); header.addWidget(self.open_button); header.addWidget(self.action_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header); layout.addWidget(self.progress_bar); layout.addWidget(self.detail_label)
        self.show_empty()
    def show_empty(self):
        self.name_label.setText(f"{self.index + 1}. Trống"); self.status_label.clear(); self.detail_label.clear()
        self.progress_bar.setValue(0); self.progress_bar.hide(); self.action_button.hide(); self.open_button.hide()
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
        self.action_button.show()

class LocalizationDashboard(QWidget):
    """Five independent UI slots backed by concurrently active workers."""
    def __init__(self, parent=None, worker_factory: Callable[..., LocalizationWorker] = LocalizationWorker):
        super().__init__(parent)
        self.settings, self.worker_factory = Settings(), worker_factory
        self.jobs = [None] * MAX_LOCALIZATION_JOBS
        self.active_jobs = []
        self._shutting_down = False
        self._build_ui()
    def _build_ui(self):
        default = "D:/AutoVideoOutput" if Path("D:/").exists() else str(Path.home() / "AutoVideoOutput")
        self.api_key_edit = QLineEdit(get_password(Key.GEMINI_API_KEY) or "")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password); self.api_key_edit.editingFinished.connect(self.save_api_key)
        self.voice_combo = QComboBox(); self.voice_combo.addItem("Hoai My", "vi-VN-HoaiMyNeural"); self.voice_combo.addItem("Nam Minh", "vi-VN-NamMinhNeural")
        self.output_edit = QLineEdit(self._configured_output_directory(default))
        self.output_button = QPushButton("Chọn thư mục"); self.output_button.clicked.connect(self.browse_output)
        self.add_video_button = QPushButton("+ Thêm video"); self.add_video_button.clicked.connect(self.browse_videos)
        output_row = QWidget(); output_layout = QHBoxLayout(output_row); output_layout.setContentsMargins(0, 0, 0, 0); output_layout.addWidget(self.output_edit); output_layout.addWidget(self.output_button)
        form = QFormLayout(); form.addRow("Gemini API key:", self.api_key_edit); form.addRow("Giọng Việt:", self.voice_combo); form.addRow("Thư mục đầu ra:", output_row)
        self.rows = [LocalizationJobRow(index, self) for index in range(MAX_LOCALIZATION_JOBS)]
        for row in self.rows:
            row.action_button.clicked.connect(lambda _=False, row=row: self.on_row_action(row.index))
            row.open_button.clicked.connect(lambda _=False, row=row: self.open_job_output(row.index))
        layout = QVBoxLayout(self); layout.addWidget(QLabel("Dịch & lồng tiếng video")); layout.addLayout(form)
        pipeline = QLabel("Tự động: Paraformer-zh → Gemini → Edge TTS → đồng bộ tự nhiên → làm mờ phụ đề gốc → phụ đề Việt"); pipeline.setWordWrap(True)
        layout.addWidget(pipeline); layout.addWidget(self.add_video_button)
        for row in self.rows: layout.addWidget(row)
        layout.addStretch()
    def save_api_key(self):
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
        if not configured or self._is_writable_directory(Path(configured)):
            return configured or default

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
        available, accepted = self.jobs.count(None), list(paths)[:self.jobs.count(None)]
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
            job = LocalizationJob(path, str(self._job_output_directory(root_path, Path(path))))
            self.jobs[self.jobs.index(None)] = job; self.refresh_job(job); added.append(job)
        self.start_next_job(); return added
    def _job_output_directory(self, root, source):
        candidate, suffix = root / source.stem, 2; used = {Path(job.output_directory) for job in self.jobs if job}
        while candidate in used or candidate.exists(): candidate = root / f"{source.stem}-{suffix}"; suffix += 1
        return candidate
    def start_next_job(self):
        if self._shutting_down: return
        jobs = [item for item in self.jobs if item and item.status == WAITING]
        for job in jobs:
            self._start_job(job)

    def _start_job(self, job):
        if not self.api_key_edit.text().strip(): job.status, job.detail = FAILED, "Cần Gemini API key để bắt đầu."; self.refresh_job(job); self.start_next_job(); return
        try:
            Path(job.output_directory).mkdir(parents=True, exist_ok=True)
        except OSError:
            job.status, job.detail = FAILED, "Không thể tạo thư mục đầu ra."
            self.refresh_job(job)
            self._show_output_directory_error(job.output_directory)
            return
        job.status, job.detail, job.progress = PROCESSING, "Đang khởi tạo...", 0; self.active_jobs.append(job); self.refresh_job(job)
        worker = self.worker_factory(source_video=job.source_video, output_directory=job.output_directory, source_language=None, translation_provider_name="gemini", translation_base_url=None, translation_api_key=self.api_key_edit.text().strip(), translation_model="gemini-3.5-flash-lite", tts_voice=self.voice_combo.currentData(), subtitle_mode="burn", cover_original_subtitles=True, use_background_separation=False)
        # This thread has no widget parent.  When a window closes while a job
        # is cancelling, the registry keeps the QObject graph alive until the
        # worker emits ``finished`` and the QThread has actually stopped.
        thread = QThread(); job.worker, job.thread = worker, thread; _LIVE_LOCALIZATION_JOBS.append(job); worker.moveToThread(thread); thread.started.connect(worker.run)
        worker.progress.connect(lambda progress, detail, job=job: self.on_progress(job, progress, detail)); worker.completed.connect(lambda result, job=job: self.on_completed(job, result)); worker.failed.connect(lambda error, job=job: self.on_failed(job, error)); worker.cancelled.connect(lambda job=job: self.on_cancelled(job)); worker.finished.connect(thread.quit); worker.finished.connect(worker.deleteLater); thread.finished.connect(lambda job=job: self.on_thread_finished(job)); thread.finished.connect(thread.deleteLater); thread.start()
    def on_progress(self, job, progress, detail): job.progress, job.detail = progress, detail; self.refresh_job(job)
    def on_completed(self, job, result): job.status, job.progress, job.detail = COMPLETED, 100, str(getattr(getattr(result, "final_video", None), "video_file", job.output_directory)); self.refresh_job(job)
    def on_failed(self, job, error): job.status, job.detail = FAILED, str(error); self.refresh_job(job)
    def on_cancelled(self, job): job.status, job.detail = CANCELLED, "Đã hủy theo yêu cầu."; self.refresh_job(job)
    def on_thread_finished(self, job):
        job.worker, job.thread = None, None
        if job in _LIVE_LOCALIZATION_JOBS:
            _LIVE_LOCALIZATION_JOBS.remove(job)
        if job in self.active_jobs: self.active_jobs.remove(job)
        if (not self.active_jobs and not any(
            item and item.status in {WAITING, PROCESSING, CANCELLING}
            for item in self.jobs
        )):
            reset_paraformer_model_cache()
    def on_row_action(self, index):
        job = self.jobs[index]
        if not job: return
        if job.status == WAITING: job.status, job.detail = CANCELLED, "Đã hủy theo yêu cầu."; self.refresh_row(index)
        elif job.status in {PROCESSING, CANCELLING}: job.status, job.detail = CANCELLING, "Đang gửi yêu cầu hủy..."; job.worker and job.worker.request_cancel(); self.refresh_row(index)
        else: self.jobs[index] = None; self.refresh_row(index)
    def open_job_output(self, index):
        job = self.jobs[index]
        if not job or job.status != COMPLETED:
            return
        target = Path(job.detail)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target if target.exists() else Path(job.output_directory))))
    def shutdown(self):
        self._shutting_down = True
        for job in tuple(self.active_jobs):
            if job.worker: job.worker.request_cancel()
    def refresh_job(self, job): self.refresh_row(self.jobs.index(job))
    def refresh_row(self, index): self.rows[index].show_empty() if self.jobs[index] is None else self.rows[index].show_job(self.jobs[index])
