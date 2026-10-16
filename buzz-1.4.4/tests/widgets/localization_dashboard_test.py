from types import SimpleNamespace
from pathlib import Path
import os

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from buzz.settings.settings import Settings as RealSettings
from buzz.store.keyring_store import Key
from buzz.widgets.localization_dashboard import (
    CANCELLED, COMPLETED, FAILED, PROCESSING, WAITING, LocalizationDashboard, LocalizationJob,
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


def test_dashboard_is_gemini_only_and_hides_advanced_options(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    assert view.translation_provider_combo.count() == 1
    assert view.translation_provider_combo.currentData() == "gemini"
    assert view.advanced_options.isHidden()
    job = view.add_videos([str(tmp_path / "one.mp4")])[0]
    assert job.worker.kwargs["translation_provider_name"] == "gemini"
    assert job.worker.kwargs["translation_base_url"] is None
    view.shutdown()
    finish(qtbot, job.worker, job)


def test_dashboard_keeps_gemini_key_secure(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    stored = []
    monkeypatch.setattr("buzz.widgets.localization_dashboard.set_password", lambda *args: stored.append(args))
    view.api_key_edit.setText("gemini-secret")
    view.save_api_key()
    assert view.api_key_edit.echoMode() == view.api_key_edit.EchoMode.Password
    assert stored == [(Key.GEMINI_API_KEY, "gemini-secret")]


def test_non_persistent_dashboard_never_reads_or_writes_keyring(qtbot, monkeypatch):
    reads, writes = [], []
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.get_password",
        lambda key: reads.append(key) or "real-secret-must-not-be-read",
    )
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.set_password",
        lambda key, value: writes.append((key, value)),
    )
    view = LocalizationDashboard(worker_factory=FakeWorker, persist_api_key=False)
    qtbot.add_widget(view)
    view.api_key_edit.setText("scheduler-test-dummy-key")
    view.save_api_key()
    assert reads == []
    assert writes == []


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
    assert first.worker.kwargs["blur_bilibili_watermark"] is True
    assert first.worker.kwargs["overlay_ben_review_logo"] is True
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


def test_logo_choice_is_snapshotted_when_a_job_is_added(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    view.overlay_ben_review_logo_checkbox.setChecked(False)
    job = view.add_videos([str(tmp_path / "one.mp4")])[0]
    assert job.overlay_ben_review_logo is False
    assert job.worker.kwargs["overlay_ben_review_logo"] is False
    view.shutdown()
    finish(qtbot, job.worker, job)


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


def test_thread_cleanup_waits_for_native_thread_before_releasing_refs(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)

    class ThreadStillStopping:
        def __init__(self):
            self.wait_results = [False, True]
            self.deleted = 0

        def wait(self, timeout):
            assert timeout == 0
            return self.wait_results.pop(0)

        def deleteLater(self):
            self.deleted += 1

    thread = ThreadStillStopping()
    job = LocalizationJob("p01.mp4", str(tmp_path), status=FAILED, worker=object(), thread=thread)
    view.jobs[0] = job
    view.active_jobs.append(job)

    view.on_thread_finished(job)

    assert job.thread is thread and job.worker is not None
    assert job in view.active_jobs
    qtbot.waitUntil(lambda: job.thread is None, timeout=1000)

    assert job.thread is None and job.worker is None
    assert job not in view.active_jobs
    assert thread.deleted == 1


def test_output_directory_prefers_existing_part_checkpoints_over_new_empty_suffix(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    root = tmp_path / "output"
    original = root / "p01"
    accidental_duplicate = root / "p01-2"
    (original / "p01_localization_work" / "part_0002"
     / "source_localization_work").mkdir(parents=True)
    (original / "p01_localization_work" / "part_0002"
     / "source_localization_work" / "timing-recovery-checkpoint.json").write_text("{}")
    (accidental_duplicate / "p01_localization_work" / "part_0001").mkdir(parents=True)
    (accidental_duplicate / "p01_localization_work" / "part_0001"
     / "source.mp4").write_bytes(b"incomplete")

    assert view._job_output_directory(root, Path("p01.mp4")) == original


def test_output_directory_prefers_highest_generation_over_newer_empty_checkpoint(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    root = tmp_path / "output"
    checkpoint, empty_newer_attempt = root / "p01-3", root / "p01"
    for directory in (checkpoint, empty_newer_attempt):
        (directory / "p01_localization_work").mkdir(parents=True)
    (checkpoint / "p01_localization_work" / "tts-167.cache").touch()
    os.utime(checkpoint, (1, 1))
    os.utime(empty_newer_attempt, (2, 2))

    assert view._job_output_directory(root, Path("p01.mp4")) == checkpoint


def test_output_directory_prefers_highest_generation_regardless_of_mtime_or_file_count(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    root = tmp_path / "output"
    older_generation, current_generation = root / "p02-3", root / "p02-4"
    for directory in (older_generation, current_generation):
        (directory / "p02_localization_work").mkdir(parents=True)
    for index in range(20):
        (older_generation / "p02_localization_work" / f"dummy-{index}").touch()
    os.utime(older_generation, (2, 2))
    os.utime(current_generation, (1, 1))

    assert view._job_output_directory(root, Path("p02.mp4")) == current_generation


def test_output_directory_skips_completed_and_similarly_named_directories(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    root = tmp_path / "output"
    completed = root / "p01"
    (completed / "p01_localization_work").mkdir(parents=True)
    (completed / "p01.vi.mp4").touch()
    for name in ("p01-extra", "p010", "p01-foo"):
        (root / name / "p01_localization_work").mkdir(parents=True)

    assert view._job_output_directory(root, Path("p01.mp4")) == root / "p01-2"


def test_output_directory_skips_completed_highest_generation(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    root = tmp_path / "output"
    incomplete, completed = root / "p04-2", root / "p04-3"
    for directory in (incomplete, completed):
        (directory / "p04_localization_work").mkdir(parents=True)
    (completed / "p04.vi.mp4").touch()

    assert view._job_output_directory(root, Path("p04.mp4")) == incomplete


def test_output_directory_skips_checkpoint_assigned_to_current_job(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    root = tmp_path / "output"
    checkpoint = root / "p01-2"
    (checkpoint / "p01_localization_work").mkdir(parents=True)
    view.jobs[0] = SimpleNamespace(output_directory=str(checkpoint))

    assert view._job_output_directory(root, Path("p01.mp4")) == root / "p01"


def test_adding_failed_source_retries_same_job_and_directory(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    source = str(tmp_path / "p01.mp4")
    job = view.add_videos([source])[0]
    directory = job.output_directory
    job.worker.failed.emit("network")
    finish(qtbot, job.worker, job)
    view.start_next_job = lambda: None

    assert view.add_videos([source]) == [job]
    assert job.output_directory == directory
    assert job.status == WAITING and job.progress == 0
    assert "Tiếp tục từ checkpoint hiện có" in job.detail


def test_add_videos_deduplicates_pending_source_without_allocating_second_output(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    source = str(tmp_path / "p01.mp4")

    first = view.add_videos([source])
    duplicate = view.add_videos([source])
    repeated_in_one_request = view.add_videos([str(tmp_path / "p02.mp4")] * 2)

    assert len(first) == 1
    assert duplicate == []
    assert len(repeated_in_one_request) == 1
    assert [job for job in view.jobs if job and Path(job.source_video).name == "p01.mp4"] == first
    assert not (tmp_path / "output" / "p01-2").exists()
    view.shutdown()
    for job in (first[0], repeated_in_one_request[0]):
        finish(qtbot, job.worker, job)


def test_retry_uses_changed_api_key_and_same_job_directory_and_slot(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    source = str(tmp_path / "p01.mp4")
    job = view.add_videos([source])[0]
    directory, first_worker = job.output_directory, job.worker
    first_worker.failed.emit("bad key")
    finish(qtbot, first_worker, job)

    view.api_key_edit.setText("replacement-key")
    saved_keys = []
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.set_password",
        lambda key, value: saved_keys.append((key, value)),
    )
    view.rows[0].retry_button.click()

    assert view.jobs[0] is job
    assert job.output_directory == directory
    assert job.status == PROCESSING
    assert job.worker is not first_worker
    assert os.path.normcase(job.worker.kwargs["source_video"]) == os.path.normcase(source)
    assert job.worker.kwargs["output_directory"] == directory
    assert job.worker.kwargs["translation_api_key"] == "replacement-key"
    assert saved_keys and saved_keys[0][1] == "replacement-key"
    view.shutdown()
    finish(qtbot, job.worker, job)


def test_retry_is_disabled_until_thread_cleanup(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    job = view.add_videos([str(tmp_path / "p01.mp4")])[0]
    worker = job.worker
    worker.failed.emit("network")

    assert not view.rows[0].retry_button.isHidden()
    assert not view.rows[0].retry_button.isEnabled()
    view.rows[0].retry_button.click()
    assert job.worker is worker

    finish(qtbot, worker, job)
    assert view.rows[0].retry_button.isEnabled()


def test_retry_without_key_preserves_cancelled_job(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    job = view.add_videos([str(tmp_path / "p01.mp4")])[0]
    worker = job.worker
    worker.cancelled.emit()
    finish(qtbot, worker, job)
    view.api_key_edit.clear()
    warnings = []
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    view.rows[0].retry_button.click()

    assert job.status == CANCELLED
    assert job.worker is None and job.thread is None
    assert warnings and "Gemini API key" in warnings[0][1]


def test_completed_rows_do_not_show_retry(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    job = view.add_videos([str(tmp_path / "p01.mp4")])[0]
    worker = job.worker
    worker.completed.emit(SimpleNamespace(final_video=SimpleNamespace(video_file="D:/final.mp4")))
    finish(qtbot, worker, job)

    assert job.status == COMPLETED
    assert view.rows[0].retry_button.isHidden()



def test_failed_job_cannot_be_removed_until_thread_finishes(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    job = view.add_videos([str(tmp_path / "p01.mp4")])[0]
    worker = job.worker
    worker.failed.emit("network")

    assert not view.rows[0].action_button.isEnabled()
    view.on_row_action(0)
    assert view.jobs[0] is job

    finish(qtbot, worker, job)
    assert view.rows[0].action_button.isEnabled()


def test_retry_missing_checkpoint_directory_does_not_start_over(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    job = view.add_videos([str(tmp_path / "p01.mp4")])[0]
    worker = job.worker
    worker.failed.emit("network")
    finish(qtbot, worker, job)
    directory = Path(job.output_directory)
    directory.rmdir()
    errors = []
    monkeypatch.setattr(
        "buzz.widgets.localization_dashboard.QMessageBox.critical",
        lambda *args: errors.append(args),
    )
    view.on_retry(0)

    assert job.status == FAILED
    assert job.worker is None
    assert not directory.exists()
    assert errors


def test_output_directory_without_checkpoint_allocates_unique_new_directory(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    (root / "p01").mkdir()

    assert view._job_output_directory(root, Path("p01.mp4")) == root / "p01-2"

def test_manual_video_orientation_is_visible_and_mutually_exclusive(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    assert view.portrait_subtitle_overlay_checkbox.isChecked() is False
    assert view.landscape_video_radio.isChecked() is True
    assert not view.advanced_options.isAncestorOf(view.landscape_video_radio)
    assert not view.advanced_options.isAncestorOf(view.portrait_subtitle_overlay_checkbox)
    view.portrait_subtitle_overlay_checkbox.setChecked(True)
    assert view.landscape_video_radio.isChecked() is False
    view.landscape_video_radio.setChecked(True)
    assert view.landscape_video_radio.isChecked() is True
    assert view.portrait_subtitle_overlay_checkbox.isChecked() is False


def test_manual_video_orientation_snapshots_on_add_and_ignores_mocked_media_probe(qtbot, tmp_path, monkeypatch):
    view = dashboard(qtbot, tmp_path)
    portrait_named_landscape = tmp_path / "landscape-1920x1080.mp4"
    landscape_named_portrait = tmp_path / "portrait-1080x1920.mp4"
    probe_calls = []

    def unexpected_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        raise AssertionError("Dashboard must not probe media orientation")

    monkeypatch.setattr("subprocess.run", unexpected_probe)
    view.portrait_subtitle_overlay_checkbox.setChecked(True)
    first = view.add_videos([str(portrait_named_landscape)])[0]
    assert first.portrait_subtitle_overlay is True
    assert first.worker.kwargs["portrait_subtitle_overlay"] is True
    view.landscape_video_radio.setChecked(True)
    second = view.add_videos([str(landscape_named_portrait)])[0]
    assert first.worker.kwargs["portrait_subtitle_overlay"] is True
    assert second.portrait_subtitle_overlay is False
    assert second.worker.kwargs["portrait_subtitle_overlay"] is False
    assert probe_calls == []
    view.shutdown()
    finish(qtbot, first.worker, first)
    finish(qtbot, second.worker, second)


def test_dashboard_has_no_auto_orientation_detection(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    assert not hasattr(view, "_is_portrait_video")
    assert not hasattr(view, "_should_apply_portrait_subtitles")
    assert view.portrait_subtitle_overlay_checkbox.isChecked() is False
    view.shutdown()


def test_portrait_option_snapshots_on_add_and_does_not_change_active_job(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    view.portrait_subtitle_overlay_checkbox.setChecked(True)
    first = view.add_videos([str(tmp_path / "portrait.mp4")])[0]
    assert first.portrait_subtitle_overlay is True
    assert first.worker.kwargs["portrait_subtitle_overlay"] is True
    view.landscape_video_radio.setChecked(True)
    second = view.add_videos([str(tmp_path / "landscape.mp4")])[0]
    assert first.worker.kwargs["portrait_subtitle_overlay"] is True
    assert second.portrait_subtitle_overlay is False
    assert second.worker.kwargs["portrait_subtitle_overlay"] is False
    view.shutdown()
    finish(qtbot, first.worker, first)
    finish(qtbot, second.worker, second)
def test_subtitle_refresh_action_validates_mp4_and_adjacent_subtitle(qtbot, tmp_path):
    view = dashboard(qtbot, tmp_path)
    view.subtitle_refresh_source_edit.setText(str(tmp_path / "missing.mp4"))
    view.start_subtitle_refresh()
    assert "existing dubbed MP4" in view.subtitle_refresh_status.text()

    video = tmp_path / "source.vi.mp4"
    video.write_bytes(b"not a real mp4")
    view.subtitle_refresh_source_edit.setText(str(video))
    view.start_subtitle_refresh()
    assert "SRT/VTT" in view.subtitle_refresh_status.text()
