from pathlib import Path

from buzz.localization.timing import TimingPolicy
from buzz.store.keyring_store import Key
from buzz.widgets.localization_dialog import LocalizationDialog, LocalizationWorker


def _dialog(qtbot, monkeypatch, saved_key=""):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: saved_key if key == Key.GEMINI_API_KEY else "",
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    return dialog


def test_simple_ui_uses_fixed_pipeline_defaults(qtbot, monkeypatch):
    dialog = _dialog(qtbot, monkeypatch)

    assert dialog.windowTitle() == "Dịch & lồng tiếng video"
    assert dialog.translation_provider_combo.currentData() == "gemini"
    assert dialog.translation_model_edit.text() == "gemini-3.5-flash-lite"
    assert dialog.subtitle_mode_combo.currentData() == "burn"
    assert dialog.cover_original_subtitles_checkbox.isChecked() is True
    assert dialog.cover_original_subtitles_checkbox.isEnabled() is True
    assert dialog.background_checkbox.isChecked() is False
    assert dialog.start_button.text() == "Bắt đầu"
    assert dialog.cancel_button.text() == "Hủy"


def test_simple_ui_only_exposes_video_key_and_voice(qtbot, monkeypatch):
    dialog = _dialog(qtbot, monkeypatch)

    assert dialog.source_edit.parent() is not None
    assert dialog.api_key_edit.parent() is not None
    assert dialog.voice_combo.parent() is not None
    assert dialog.output_edit.parent() is None
    assert dialog.translation_provider_combo.parent() is None
    assert dialog.translation_model_edit.parent() is None
    assert dialog.subtitle_mode_combo.parent() is None
    assert dialog.cover_original_subtitles_checkbox.parent() is None
    assert dialog.background_checkbox.parent() is None


def test_validation_accepts_valid_video_and_gemini_key(qtbot, monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    dialog = _dialog(qtbot, monkeypatch)
    dialog.source_edit.setText(str(source))
    dialog.api_key_edit.setText("key")

    assert dialog._validate_inputs() is None


def test_validation_reports_missing_source(qtbot, monkeypatch):
    dialog = _dialog(qtbot, monkeypatch)
    dialog.api_key_edit.setText("key")

    assert dialog._validate_inputs() == "Please select a valid input video."


def test_validation_requires_gemini_key(qtbot, monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    dialog = _dialog(qtbot, monkeypatch)
    dialog.source_edit.setText(str(source))
    dialog.api_key_edit.setText("")

    assert dialog._validate_inputs() == "Gemini API key is required."


def test_gemini_api_key_loads_from_keyring(qtbot, monkeypatch):
    dialog = _dialog(qtbot, monkeypatch, saved_key="saved-gemini-key")
    assert dialog.api_key_edit.text() == "saved-gemini-key"


def test_gemini_api_key_saves_to_keyring(qtbot, monkeypatch):
    saved = []
    dialog = _dialog(qtbot, monkeypatch)
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.set_password",
        lambda key, value: saved.append((key, value)),
    )
    dialog.api_key_edit.setText("new-gemini-key")
    dialog._save_current_api_key()

    assert saved == [(Key.GEMINI_API_KEY, "new-gemini-key")]


def test_worker_uses_gemini_paraformer_and_timing_policy(monkeypatch):
    calls = []
    provider_args = []

    class FakeGemini:
        def __init__(self, **kwargs):
            provider_args.append(kwargs)

    monkeypatch.setattr("buzz.widgets.localization_dialog.GeminiTranslationProvider", FakeGemini)
    monkeypatch.setattr("buzz.widgets.localization_dialog.EdgeTTSProvider", lambda **kwargs: object())
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.localize_video",
        lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )

    worker = LocalizationWorker(
        source_video="source.mp4",
        output_directory="output",
        source_language=None,
        translation_provider_name="gemini",
        translation_base_url=None,
        translation_api_key="key",
        translation_model="gemini-3.5-flash-lite",
        tts_voice="vi-VN-HoaiMyNeural",
        subtitle_mode="burn",
        cover_original_subtitles=True,
        use_background_separation=False,
    )
    worker.run()

    assert provider_args == [
        {"api_key": "key", "model": "gemini-3.5-flash-lite"}
    ]
    args, kwargs = calls[0]
    assert kwargs["asr_provider"] == "paraformer-zh"
    assert args[5].timing_policy == TimingPolicy()
    assert args[5].render_options.subtitle_mode == "burn"
    assert args[5].render_options.cover_original_subtitles is True
    assert args[5].use_background_separation is False
