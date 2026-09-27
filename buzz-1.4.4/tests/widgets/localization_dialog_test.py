from pathlib import Path
from unittest.mock import Mock

from buzz.widgets.localization_dialog import LocalizationDialog


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
    assert dialog.language_combo.itemData(1) == "en"
    assert dialog.language_combo.itemData(2) == "zh"


def test_localization_dialog_validation_accepts_valid_required_fields(
    qtbot, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    source = tmp_path / "video.mp4"
    model = tmp_path / "tiny.pt"
    source.write_bytes(b"video")
    model.write_bytes(b"model")

    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.source_edit.setText(str(source))
    dialog.output_edit.setText(str(tmp_path / "output"))
    dialog.model_path_edit.setText(str(model))
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
    model = tmp_path / "tiny.pt"
    source.write_bytes(b"video")
    model.write_bytes(b"model")

    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("argos")
    )
    dialog.source_edit.setText(str(source))
    dialog.output_edit.setText(str(tmp_path / "output"))
    dialog.model_path_edit.setText(str(model))
    dialog.api_key_edit.setText("")
    dialog.translation_model_edit.setText("")

    assert dialog._validate_inputs() is None


def test_gemini_validation_requires_key(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "buzz.widgets.localization_dialog.get_password",
        lambda key: "",
    )
    source = tmp_path / "video.mp4"
    model = tmp_path / "tiny.pt"
    source.write_bytes(b"video")
    model.write_bytes(b"model")

    dialog = LocalizationDialog()
    qtbot.add_widget(dialog)
    dialog.translation_provider_combo.setCurrentIndex(
        dialog.translation_provider_combo.findData("gemini")
    )
    dialog.source_edit.setText(str(source))
    dialog.output_edit.setText(str(tmp_path / "output"))
    dialog.model_path_edit.setText(str(model))
    dialog.api_key_edit.setText("")
    dialog.translation_model_edit.setText("gemini-2.5-flash")

    assert dialog._validate_inputs() == "Gemini API key is required."
