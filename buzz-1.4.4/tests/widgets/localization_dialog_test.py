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
