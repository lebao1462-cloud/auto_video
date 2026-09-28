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


@pytest.fixture(autouse=True)
def available_paraformer(monkeypatch):
    monkeypatch.setattr(
        "buzz.localization.paraformer.paraformer_runtime_available", lambda: True,
    )
    monkeypatch.setattr(
        "buzz.localization.paraformer.paraformer_model_is_cached", lambda cache: True,
    )


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
        "buzz.localization.paraformer.paraformer_runtime_available",
        lambda: True,
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


def test_preflight_rejects_missing_source_without_requiring_a_model(monkeypatch, tmp_path):
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
    assert not any("model path" in error.lower() for error in report.errors)


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


def test_argos_preflight_does_not_require_api_key_or_model(monkeypatch, tmp_path):
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
        "buzz.localization.providers.argos_route_available",
        lambda language: (True, f"Argos route available: {language} -> vi"),
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    report = run_localization_preflight(
        source,
        model,
        output,
        translation_provider="argos",
        source_language="zh",
        translation_api_key="",
        translation_model="",
    )

    assert report.ok is True
    assert not any(check.name == "translation_api_key" for check in report.checks)
    assert not any(check.name == "translation_model" for check in report.checks)


def test_nllb_preflight_is_network_free_and_warns_when_model_is_missing(monkeypatch, tmp_path):
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
        "buzz.localization.providers.nllb_model_is_available", lambda: False,
    )
    monkeypatch.setattr(
        "buzz.localization.providers.nllb_model_path", lambda: tmp_path / "nllb",
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    report = run_localization_preflight(
        source, model, output, translation_provider="nllb", translation_api_key="",
        translation_model="",
    )

    assert report.ok is True
    assert not any(check.name == "translation_api_key" for check in report.checks)
    assert "first use downloads" in " ".join(report.warnings)


def test_nllb_preflight_requires_all_runtime_packages(monkeypatch, tmp_path):
    source, model, output = _make_inputs(tmp_path)
    monkeypatch.setattr(
        "buzz.localization.preflight._find_bundled_or_path_executable",
        lambda name: f"C:/{name}.exe",
    )
    packages = {"torch", "transformers", "huggingface_hub"}
    monkeypatch.setattr(
        "buzz.localization.preflight.importlib.util.find_spec",
        lambda name: object() if name in packages else None,
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    report = run_localization_preflight(
        source, model, output, translation_provider="nllb", translation_api_key="",
        translation_model="",
    )

    runtime = next(check for check in report.checks if check.name == "nllb_runtime")
    assert runtime.ok is False
    assert "sentencepiece" in runtime.message


def test_argos_preflight_blocks_missing_route(monkeypatch, tmp_path):
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
        "buzz.localization.providers.argos_route_available",
        lambda language: (False, f"missing {language}->vi route"),
    )
    monkeypatch.setattr(
        "buzz.localization.preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    report = run_localization_preflight(
        source,
        model,
        output,
        translation_provider="argos",
        source_language="zh",
    )

    assert report.ok is False
    assert any("missing zh->vi route" in error for error in report.errors)


def test_argos_auto_language_defers_route_check_until_detection(monkeypatch, tmp_path):
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
        translation_provider="argos",
        source_language=None,
    )

    assert report.ok is True
    auto_check = next(c for c in report.checks if c.name == "argos_route_auto")
    assert "after source-language auto-detection" in auto_check.message


def test_gemini_preflight_requires_only_gemini_configuration(monkeypatch, tmp_path):
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

    missing = run_localization_preflight(
        source,
        model,
        output,
        translation_provider="gemini",
        translation_api_key="",
        translation_model="",
    )
    assert missing.ok is False
    assert "Gemini API key is required." in missing.errors
    assert "Gemini model is required." in missing.errors

    ok = run_localization_preflight(
        source,
        model,
        output,
        translation_provider="gemini",
        translation_api_key="key",
        translation_model="gemini-2.5-flash",
    )
    assert ok.ok is True


def test_preflight_rejects_unknown_translation_provider(tmp_path):
    source, model, output = _make_inputs(tmp_path)

    with pytest.raises(ValueError, match="Unsupported translation provider"):
        run_localization_preflight(
            source,
            model,
            output,
            translation_provider="unknown",
        )


def test_paraformer_preflight_is_network_free_and_rejects_english(monkeypatch, tmp_path):
    source, model, output = _make_inputs(tmp_path)
    monkeypatch.setattr("buzz.localization.preflight._find_bundled_or_path_executable", lambda name: f"C:/{name}.exe")
    monkeypatch.setattr("buzz.localization.preflight.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("buzz.localization.preflight.shutil.disk_usage", lambda path: type("Usage", (), {"free": 10 * 1024**3})())
    monkeypatch.setattr("buzz.localization.paraformer.paraformer_runtime_available", lambda: True)
    monkeypatch.setattr("buzz.localization.paraformer.paraformer_model_is_cached", lambda cache: False)

    report = run_localization_preflight(
        source, "", output, translation_provider="nllb", source_language="en",
        translation_api_key="", translation_model="", asr_provider="paraformer-zh",
    )

    assert report.ok is False
    assert any("only supports Chinese" in error for error in report.errors)
    assert any(check.name == "paraformer_model" and not check.required for check in report.checks)
