from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from buzz.settings.settings import Settings as RealSettings
from buzz.widgets.localization_dashboard import (
    CANCELLED, COMPLETED, PROCESSING, WAITING, LocalizationDashboard,
)


class FakeSettings:
    """Test-local settings store; never instantiate QSettings in these tests."""
    Key = RealSettings.Key
    values = {}
    instances = []

    def __init__(self):
        self.instances.append(self)

    def value(self, key, default_value, value_type=None):
        return self.values.get(key, default_value)

    def set_value(self, key, value):
        self.values[key] = value


@pytest.fixture(scope="session")
def qapp_cls():
    """Dashboard tests do not need the application database."""
    return QApplication


@pytest.fixture(autouse=True)
def isolate_dashboard_storage(monkeypatch):
    """Never read or overwrite the user's keyring or QSettings in tests."""
    FakeSettings.values = {}
    FakeSettings.instances = []
    monkeypatch.setattr("buzz.widgets.localization_dashboard.Settings", FakeSettings)
    monkeypatch.setattr("buzz.widgets.localization_dashboard.get_password", lambda key: "")
    monkeypatch.setattr("buzz.widgets.localization_dashboard.set_password", lambda key, value: None)


class FakeWorker(QObject):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    finished = pyqtSignal()
    instances = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.cancel_requested = False
        self.instances.append(self)

    def run(self):
        pass

    def request_cancel(self):
        self.cancel_requested = True


def dashboard(qtbot, tmp_path):
    FakeWorker.instances.clear()
    result = LocalizationDashboard(worker_factory=FakeWorker)
    result.output_edit.setText(str(tmp_path / "output"))
    result.api_key_edit.setText("test-key")
    qtbot.add_widget(result)
    return result


def finish(qtbot, worker, job):
    worker.finished.emit()
    qtbot.waitUntil(lambda: job.thread is None, timeout=1000)


def test_initial_five_empty_slots(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    assert len(view.rows) == 5
    assert all(job is None for job in view.jobs)
    assert all("Trống" in row.name_label.text() for row in view.rows)


def test_dashboard_uses_fake_settings_only(qtbot, tmp_path):
    dashboard(qtbot, tmp_path)
    assert len(FakeSettings.instances) == 1


def test_invalid_configured_output_directory_falls_back_to_default(qtbot, tmp_path, monkeypatch):
    invalid = tmp_path / "not-writable"
    fallback = tmp_path / "AutoVideoOutput"
    FakeSettings.values[FakeSettings.Key.LOCALIZATION_OUTPUT_DIRECTORY] = str(invalid)
    monkeypatch.setattr(
        LocalizationDashboard,
        "_is_writable_directory",
        lambda self, directory: directory == fallback,
    )
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.Path.exists",
        lambda self: False,
    )
    monkeypatch.setattr("buzz.widgets.localization_dashboard.Path.home", lambda: tmp_path)

    view = LocalizationDashboard(worker_factory=FakeWorker)
    qtbot.add_widget(view)

    assert view.output_edit.text() == str(fallback)
    assert FakeSettings.values[FakeSettings.Key.LOCALIZATION_OUTPUT_DIRECTORY] == str(fallback)


def test_add_videos_reports_output_directory_creation_failure(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    errors = []
    monkeypatch.setattr(
        LocalizationDashboard, "_is_writable_directory", lambda self, directory: False
    )
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.QMessageBox.critical",
        lambda *args: errors.append(args),
    )

    assert view.add_videos([str(tmp_path / "one.mp4")]) == []
    assert all(job is None for job in view.jobs)
    assert errors and "Không thể tạo hoặc ghi" in errors[0][2]


def test_max_five_and_all_five_workers_start(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    warnings = []
    monkeypatch.setattr("buzz.widgets.localization_dashboard.QMessageBox.warning", lambda *args: warnings.append(args))
    jobs = view.add_videos([str(tmp_path / f"video-{i}.mp4") for i in range(6)])
    assert len(jobs) == 5 and len(FakeWorker.instances) == 5
    assert all(job.status == PROCESSING for job in jobs)
    assert warnings
    view.shutdown()
    for job in jobs:
        finish(qtbot, job.worker, job)


def test_progress_queued_cancel_and_active_cancel(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    first, second = view.add_videos([str(tmp_path / "one.mp4"), str(tmp_path / "two.mp4")])
    first.worker.progress.emit(37, "Đang dịch")
    assert view.rows[0].progress_bar.value() == 37
    assert view.rows[0].detail_label.text() == "Đang dịch"
    view.on_row_action(1)
    assert second.worker.cancel_requested
    second.worker.cancelled.emit()
    assert second.status == CANCELLED
    view.on_row_action(0)
    assert first.worker.cancel_requested
    first.worker.cancelled.emit()
    assert first.status == CANCELLED
    finish(qtbot, first.worker, first)


def test_adding_while_running_starts_another_row(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    first = view.add_videos([str(tmp_path / "one.mp4")])[0]
    second = view.add_videos([str(tmp_path / "two.mp4")])[0]
    assert first.status == PROCESSING and second.status == PROCESSING
    assert len(view.active_jobs) == 2
    view.shutdown()
    finish(qtbot, first.worker, first)
    finish(qtbot, second.worker, second)


def test_waiting_progress_and_cancel_while_waiting(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    first = view.add_videos([str(tmp_path / "one.mp4")])[0]
    first.worker.progress.emit(12, "Đang chờ tài nguyên: Gemini")
    assert first.status == PROCESSING
    assert first.progress == 12
    assert "chờ" in view.rows[0].detail_label.text()
    view.on_row_action(0)
    assert first.worker.cancel_requested
    first.worker.cancelled.emit()
    assert first.status == CANCELLED
    finish(qtbot, first.worker, first)


def test_completed_row_can_open_then_clear_a_slot(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    view.voice_combo.setCurrentIndex(1)
    first = view.add_videos([str(tmp_path / "one.mp4")])[0]
    assert first.worker.kwargs["translation_provider_name"] == "gemini"
    assert first.worker.kwargs["translation_model"] == "gemini-3.5-flash-lite"
    assert first.worker.kwargs["tts_voice"] == "vi-VN-NamMinhNeural"
    assert first.worker.kwargs["subtitle_mode"] == "burn"
    assert first.worker.kwargs["cover_original_subtitles"] is True
    first.worker.completed.emit(SimpleNamespace(final_video=SimpleNamespace(video_file="D:/final.mp4")))
    finish(qtbot, first.worker, first)
    opened = []
    monkeypatch.setattr("buzz.widgets.localization_dashboard.QDesktopServices.openUrl", lambda url: opened.append(url))
    assert not view.rows[0].open_button.isHidden()
    view.open_job_output(0)
    assert opened and view.jobs[0] is first
    view.on_row_action(0)
    assert view.jobs[0] is None
    assert view.add_videos([str(tmp_path / "two.mp4")])
    view.shutdown()
    active_job = view.active_jobs[0]
    finish(qtbot, active_job.worker, active_job)


def test_paraformer_cache_resets_only_after_final_active_thread(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    resets = []
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.reset_paraformer_model_cache",
        lambda: resets.append("reset"),
    )
    first, second = view.add_videos([str(tmp_path / "one.mp4"), str(tmp_path / "two.mp4")])
    first.worker.cancelled.emit()
    finish(qtbot, first.worker, first)
    assert resets == []
    second.worker.cancelled.emit()
    finish(qtbot, second.worker, second)
    assert resets == ["reset"]
