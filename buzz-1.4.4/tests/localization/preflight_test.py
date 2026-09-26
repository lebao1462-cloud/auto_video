import json
from pathlib import Path

import pytest

from buzz.localization.preflight import (
    PreflightCheck,
    LocalizationPreflightReport,
    estimate_required_workspace_bytes,
    run_localization_preflight,
    write_localization_diagnostics,
)


def _make_inputs(tmp_path):
    source = tmp_path / "video.mp4"
    model = tmp_path / "tiny.pt"
    output = tmp_path / "output"
    source.write_bytes(b"x" * 1024)
    model.write_bytes(b"model")
    return source, model, output


def test_workspace_estimate_has_conservative_floor():
    assert estimate_required_workspace_bytes(1) == 512 * 1024 * 1024
    assert estimate_required_workspace_bytes(200 * 1024 * 1024) == 1200 * 1024 * 1024


def test_workspace_estimate_rejects_negative_size():
    with pytest.raises(ValueError):
        estimate_required_workspace_bytes(-1)


def test_preflight_passes_with_available_runtime(monkeypatch, tmp_path):
    source, model, output = _make_inputs(tmp_path)

    monkeypatch.setattr(
        "buzz.localization.preflight._find_bundled_or_path_executable",
        lambda name: f"C:/{name}.exe",
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    report = run_localization_preflight(
        source,
        model,
        output,
        translation_api_key="key",
        translation_model="model",
        use_background_separation=True,
    )

    assert report.ok is True
    assert report.errors == ()
    assert report.source_size_bytes == 1024


def test_preflight_reports_missing_required_capabilities(monkeypatch, tmp_path):
    source, model, output = _make_inputs(tmp_path)

    monkeypatch.setattr(
        "buzz.localization.preflight._find_bundled_or_path_executable",
        lambda name: None,
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.importlib.util.find_spec",
        lambda name: None,
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 1})(),
    )

    report = run_localization_preflight(
        source,
        model,
        output,
        translation_api_key="",
        translation_model="",
        use_background_separation=True,
    )

    assert report.ok is False
    joined = " ".join(report.errors)
    assert "ffmpeg" in joined
    assert "ffprobe" in joined
    assert "Edge TTS" in joined
    assert "Demucs" in joined
    assert "API key" in joined
    assert "Translation model" in joined
    assert "disk space" in joined


def test_missing_edge_tts_can_be_warning_when_not_required(monkeypatch, tmp_path):
    source, model, output = _make_inputs(tmp_path)
    monkeypatch.setattr(
        "buzz.localization.preflight._find_bundled_or_path_executable",
        lambda name: f"C:/{name}.exe",
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.importlib.util.find_spec",
        lambda name: None,
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    report = run_localization_preflight(
        source,
        model,
        output,
        translation_api_key="key",
        translation_model="model",
        require_edge_tts=False,
    )

    assert report.ok is True
    assert any("Edge TTS" in warning for warning in report.warnings)


def test_preflight_rejects_missing_source_and_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "buzz.localization.preflight._find_bundled_or_path_executable",
        lambda name: f"C:/{name}.exe",
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.importlib.util.find_spec",
        lambda name: object(),
    )

    report = run_localization_preflight(
        tmp_path / "missing.mp4",
        tmp_path / "missing.pt",
        tmp_path / "output",
        translation_api_key="key",
        translation_model="model",
    )

    assert report.ok is False
    assert "Input video does not exist." in report.errors
    assert "Whisper model path does not exist." in report.errors


def test_diagnostics_do_not_contain_api_key(monkeypatch, tmp_path):
    source, model, output = _make_inputs(tmp_path)
    monkeypatch.setattr(
        "buzz.localization.preflight._find_bundled_or_path_executable",
        lambda name: f"C:/{name}.exe",
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    secret = "SUPER-SECRET-KEY"
    report = run_localization_preflight(
        source,
        model,
        output,
        translation_api_key=secret,
        translation_model="model",
    )
    diagnostics = tmp_path / "diagnostics.json"
    write_localization_diagnostics(report, diagnostics)

    text = diagnostics.read_text(encoding="utf-8")
    assert secret not in text
    data = json.loads(text)
    assert data["ok"] is True
    assert "checks" in data


def test_report_properties_only_require_required_checks():
    report = LocalizationPreflightReport(
        checks=(
            PreflightCheck("required", True, "ok", required=True),
            PreflightCheck("warning", False, "warn", required=False),
        ),
        source_size_bytes=1,
        free_space_bytes=2,
        estimated_required_bytes=1,
    )

    assert report.ok is True
    assert report.errors == ()
    assert report.warnings == ("warn",)
