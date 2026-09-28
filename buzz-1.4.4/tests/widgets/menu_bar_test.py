from pathlib import Path
from unittest.mock import patch, Mock

from PyQt6.QtCore import QSettings

from buzz.widgets.menu_bar import MenuBar
from buzz.widgets.preferences_dialog.models.preferences import Preferences
from buzz.widgets.preferences_dialog.preferences_dialog import PreferencesDialog


class TestMenuBar:
    def test_only_help_menu_is_exposed(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)
        assert len(menu_bar.actions()) == 1
        actions = menu_bar.actions()[0].menu().actions()
        assert menu_bar.preferences_action in actions
        assert menu_bar.import_action not in actions
        assert menu_bar.import_url_action not in actions
        assert menu_bar.import_folder_action not in actions
        assert menu_bar.localize_video_action not in actions
        assert len(actions) == 3

    def test_import_folder_action_emits_signal(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)

        signal_mock = Mock()
        menu_bar.import_folder_action_triggered.connect(signal_mock)
        menu_bar.import_folder_action.trigger()

        signal_mock.assert_called_once()

    def test_localize_video_action_emits_signal(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)

        signal_mock = Mock()
        menu_bar.localize_video_action_triggered.connect(signal_mock)
        menu_bar.localize_video_action.trigger()

        signal_mock.assert_called_once()

    def test_open_preferences_dialog(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)

        preferences_dialog = menu_bar.findChild(PreferencesDialog)
        assert preferences_dialog is None

        menu_bar.preferences_action.trigger()

        preferences_dialog = menu_bar.findChild(PreferencesDialog)
        assert isinstance(preferences_dialog, PreferencesDialog)

    def test_help_opens_project_instead_of_upstream_docs(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)

        with patch("buzz.widgets.menu_bar.webbrowser.open") as open_browser:
            menu_bar.on_help_action_triggered()

        open_browser.assert_called_once_with(
            (Path(__file__).resolve().parents[2] / "HELP.html").as_uri()
        )
