from pathlib import Path
from unittest.mock import Mock

from buzz.store.keyring_store import Key
from buzz.localization.timing import TimingPolicy
from buzz.widgets.localization_dialog import LocalizationDialog, LocalizationWorker


def test_localization_dialog_has_expected_defaults(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)

    assert dialog.voice_combo.count() == 2
    assert dialog.voice_combo.itemData(0) == "vi-VN-HoaiMyNeural"
    assert dialog.voice_combo.itemData(1) == "vi-VN-NamMinhNeural"
    assert dialog.subtitle_mode_combo.itemData(0) == "soft"
    assert dialog.subtitle_mode_combo.itemData(1) == "burn"
    assert dialog.subtitle_mode_combo.itemData(2) == "none"
    assert dialog.language_combo.itemData(0) is None
    assert dialog.language_combo.itemData(1) == "zh"
    assert dialog.translation_model_edit.text() == "gemini-3.5-flash-lite"
    assert dialog.asr_provider_label.text() == "Paraformer-zh - Chinese / Offline / Recommended"
    assert "Chinese only" in dialog.asr_provider_status.text()


def test_localization_dialog_only_offers_auto_or_chinese(qtbot, monkeypatch):
    monkeypatch.setattr("buzz.widgets.localization_dialog.get_password", lambda key: "")
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    assert dialog.language_combo.count() == 2
    assert dialog.language_combo.findData("en") == -1


def test_localization_dialog_validation_accepts_valid_required_fields(
    qtbot, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.source_edit.setText(str(source))
    dialog.output_edit.setText(str(tmp_path / "output"))
    dialog.api_key_edit.setText("key")
    dialog.translation_model_edit.setText("model")

    assert dialog._validate_inputs() is None


def test_localization_dialog_validation_reports_missing_source(
    qtbot, monkeypatch
):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)

    assert dialog._validate_inputs() == "Please select a valid input video."


def test_argos_is_default_and_hides_api_fields(qtbot, monkeypatch):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.Settings.value",
        lambda self, key, default, value_type=None: (
            "argos" if key.name == "LOCALIZATION_TRANSLATION_PROVIDER" else default
        ),
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)

    assert dialog.translation_provider_combo.currentData() == "argos"
    assert dialog.api_key_edit.isHidden()
    assert dialog.translation_model_edit.isHidden()
    assert dialog.base_url_edit.isHidden()
    assert "Offline / free" in dialog.translation_provider_status.text()


def test_nllb_is_fresh_default_and_hides_provider_fields(qtbot, monkeypatch):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password", lambda key: "",
    )
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.Settings.value",
        lambda self, key, default, value_type=None: default,
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)

    assert dialog.translation_provider_combo.currentData() == "nllb"
    assert dialog.translation_provider_combo.itemText(0) == "NLLB-200 - Offline / Free / No API key"
    assert dialog.api_key_edit.isHidden()
    assert dialog.translation_model_edit.isHidden()
    assert dialog.base_url_edit.isHidden()
    assert "First use downloads" in dialog.translation_provider_status.text()


def test_worker_selects_nllb_provider(monkeypatch):
    selected = []

    class FakeNLLB:
        pass

    monkeypatch.setattr("buzz.widgets.localization_dialog.NLLBTranslationProvider", FakeNLLB)
    monkeypatch.setattr("buzz.widgets.localization_dialog.EdgeTTSProvider", lambda **kwargs: object())
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.localize_video",
        lambda *args, **kwargs: selected.append(args[3]) or object(),
    )
    worker = LocalizationWorker(
        source_video="source.mp4", output_directory="output", source_language="zh",
        translation_provider_name="nllb",
        translation_base_url=None, translation_api_key="", translation_model="",
        tts_voice="voice", subtitle_mode="soft", cover_original_subtitles=False,
        use_background_separation=False,
    )

    worker.run()

    assert isinstance(selected[0], FakeNLLB)


def test_worker_always_uses_paraformer_timing_policy(monkeypatch):
    calls = []
    monkeypatch.setattr("buzz.widgets.localization_dialog.NLLBTranslationProvider", lambda: object())
    monkeypatch.setattr("buzz.widgets.localization_dialog.EdgeTTSProvider", lambda **kwargs: object())
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.localize_video",
        lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )
    worker = LocalizationWorker(
        source_video="source.mp4", output_directory="output", source_language="zh",
        translation_provider_name="nllb",
        translation_base_url=None, translation_api_key="", translation_model="",
        tts_voice="voice", subtitle_mode="soft", cover_original_subtitles=False,
        use_background_separation=False,
    )

    worker.run()

    args, kwargs = calls[0]
    assert kwargs["asr_provider"] == "paraformer-zh"
    assert args[2] == ""
    assert args[5].timing_policy == TimingPolicy()


def test_gemini_provider_shows_key_and_model_but_hides_base_url(qtbot, monkeypatch):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)

    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("gemini")
    )

    assert dialog.api_key_edit.isHidden() is False
    assert dialog.translation_model_edit.isHidden() is False
    assert dialog.base_url_edit.isHidden()
    assert "Gemini" in dialog.translation_provider_status.text()


def test_openai_provider_shows_all_translation_fields(qtbot, monkeypatch):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)

    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("openai-compatible")
    )

    assert dialog.api_key_edit.isHidden() is False
    assert dialog.translation_model_edit.isHidden() is False
    assert dialog.base_url_edit.isHidden() is False


def test_argos_validation_does_not_require_api_key(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("argos")
    )
    dialog.source_edit.setText(str(source))
    dialog.output_edit.setText(str(tmp_path / "output"))
    dialog.api_key_edit.setText("")
    dialog.translation_model_edit.setText("")

    assert dialog._validate_inputs() is None


def test_gemini_validation_requires_key(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("gemini")
    )
    dialog.source_edit.setText(str(source))
    dialog.output_edit.setText(str(tmp_path / "output"))
    dialog.api_key_edit.setText("")
    dialog.translation_model_edit.setText("gemini-2.5-flash")

    assert dialog._validate_inputs() == "Gemini API key is required."


def test_gemini_api_key_loads_from_keyring(qtbot, monkeypatch):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "saved-gemini-key" if key == Key.GEMINI_API_KEY else "",
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("gemini")
    )
    assert dialog.api_key_edit.text() == "saved-gemini-key"


def test_gemini_api_key_saves_to_keyring(qtbot, monkeypatch):
    saved = []
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.set_password",
        lambda key, value: saved.append((key, value)),
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("gemini")
    )
    dialog.api_key_edit.setText("new-gemini-key")
    dialog._save_current_api_key()

    assert saved == [(Key.GEMINI_API_KEY, "new-gemini-key")]


def test_cover_original_subtitles_option_only_enabled_for_burn(qtbot, monkeypatch):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)

    assert dialog.cover_original_subtitles_checkbox.isChecked() is True
    assert dialog.cover_original_subtitles_checkbox.isEnabled() is False

    dialog.subtitle_mode_combo.setCurrentIndex(
        dialog.subtitle_mode_combo.findData("burn")
    )
    assert dialog.cover_original_subtitles_checkbox.isEnabled() is True

    dialog.subtitle_mode_combo.setCurrentIndex(
        dialog.subtitle_mode_combo.findData("none")
    )
    assert dialog.cover_original_subtitles_checkbox.isEnabled() is False
